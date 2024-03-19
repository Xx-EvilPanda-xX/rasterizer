use std::{io::BufReader, fs::File, collections::HashMap, time::Instant, sync::Mutex};
use camera::Camera;
use image::{RgbImage, RgbaImage};
use math::*;
use config::Config;
use renderer::Clip;
use winit::{event_loop::EventLoop , window::{WindowBuilder, Fullscreen}, dpi::{PhysicalPosition, PhysicalSize}, event::{Event, DeviceEvent, WindowEvent}, keyboard::KeyCode};
use winit_input_helper::WinitInputHelper;
use pixels::{Pixels, SurfaceTexture};
use scoped_threadpool::Pool;

mod config;
mod math;
mod renderer;
mod camera;

// z coord of the "screen" (anything with a smaller z will not be shown)
const SCREEN_Z: f64 = -1.0;
const DEPTH_INIT: f64 = 1.0;
const COLOR_BUF_CHANNELS: usize = 3;
const FRAME_BUF_CHANNELS: usize = 4;

struct Buffer {
    color: Box<[u8]>,
    depth: Box<[f64]>,
    shadow_depth: Box<[f64]>,
}

pub struct SubBuffer<'a> {
    color: Option<&'a mut [u8]>,
    depth: &'a mut [f64],
    dims: (u32, u32),
    start_y: u32,
}

fn main() {
    let args_vec: Vec<_> = std::env::args().collect();
    let args = parse_args(&args_vec);
    let len = args.len();
    if len == 0 {
        println!("USAGE: {} [config_name] OPTIONAL: [obj_name] OPTIONAL: [out_name]", args_vec[0]);
        std::process::exit(-1);
    }

    let config_str = get_config(args.get("config_name").expect("config_name is a mandatory arg"));
    let obj = if let Some(obj_name) = args.get("obj_name") {
            (Some(std::fs::read_to_string(obj_name.clone() + ".obj").expect("Failed to locate obj file")),
            Some(std::fs::read_to_string(obj_name.clone() + ".mtl").expect("Failed to locate mtl file")))
    } else {
        (None, None)
    };

    // here, we leak a reference to our mtl data, allowing to live for the rest of the program (required by event_loop.run())
    let mtls = Box::leak(Box::new(load_mtl_data(&obj.1)));
    let config = Config::new(&config_str, &obj.0, mtls);

    if config.shadow_mode == 1 {
        println!("WARNING: Shadow rendering is incredibly slow and time to render will increase with the sqaure of the triangle count. (recommendation: do not exceed 10000 tris at 4k res)");
    }

    if let Some(out_name) = args.get("out_name") {
        render_to_image(&config, out_name);
    } else {
        start_interactive(config);
    }
}

const ARGS: [&'static str; 3] = ["config_name", "obj_name", "out_name"];

fn parse_args(args: &[String]) -> HashMap<String, String> {
    let mut out = HashMap::new();

    for arg in args.iter().skip(1) {
        if let Some(i) = arg.find(':') {
            let (key, value) = arg.split_at(i);
            let value = &value[1..];
            if !ARGS.contains(&key) {
                println!("Unknown arg `{}`!", key);
                std::process::exit(-1);
            }

            if out.contains_key(key) {
                println!("Duplicate arg! Please remove one of them.");
                std::process::exit(-1);
            }

            out.insert(key.to_string(), value.to_string());
        } else {
            println!("Invalid arg format! An arg should formatted like so: [key]:[value]");
            std::process::exit(-1);
        }
    }

    out
}

fn start_interactive(mut config: Config<'static>) {
    let event_loop = EventLoop::new().expect("couldn't create eventloop");
    let mut input = WinitInputHelper::new();

    let fullscreen = if config.fullscreen {
        Some(Fullscreen::Borderless(None))
    } else {
        None
    };

    let position = if config.fullscreen {
        PhysicalPosition::new(0, 0)
    } else {
        PhysicalPosition::new(50, 50)
    };

    let size = if config.fullscreen {
        let size = event_loop.primary_monitor().unwrap().size();
        config.width = size.width;
        config.height = size.height;
        size
    } else {
        PhysicalSize::new(config.width, config.height)
    };

    let window: winit::window::Window = WindowBuilder::new()
        .with_title("Renderer")
        .with_visible(false)
        .with_fullscreen(fullscreen)
        .with_position(position)
        .with_inner_size(size)
        .with_min_inner_size(size)
        .build(&event_loop)
        .unwrap();

    window.set_visible(true);
    window.set_cursor_visible(false);

    let mut frame_buffer = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);

        Pixels::new(config.width, config.height, surface_texture).expect("Failed to create pixel buffer")
    };

    let dims = (config.width, config.height);

    let light = get_light();

    let mut processed_tris = config.triangles.clone().into_boxed_slice();
    let mut processed_tris_shadow = config.triangles.clone().into_boxed_slice();
    let mut processed_light = light.clone().into_boxed_slice();
    // will contain purely the world space transform of the scene, used for shadow calculation
    let mut occlusion_testing = vec![[Point3d::default(); 3]; processed_tris.len()].into_boxed_slice();
    // this is kinda just a way around having to use an optional type in the signature of vertex_shader_pass. it's completely unused. FIXME
    let mut dummy1 = vec![[Point3d::default(); 3]; light.len()].into_boxed_slice();
    let mut dummy2 = light.clone().into_boxed_slice().clone();
    let mut buf = Buffer {
        color: vec![0; (config.width * config.height) as usize * COLOR_BUF_CHANNELS].into_boxed_slice(),
        depth: vec![DEPTH_INIT; (config.width * config.height) as usize].into_boxed_slice(),
        shadow_depth: vec![DEPTH_INIT; (config.shadow_buf_width * config.shadow_buf_height) as usize].into(),
    };

    let mut pool = Pool::new(config.render_threads);

    let mut last_instant = Instant::now();
    let mut last_frame = Instant::now();
    let mut hold_wireframe = false;
    let mut show_shadow_depth = false;

    let mut camera = Camera::new(Point3d::origin(), 0.0, 0.0);

    event_loop.run(move |event, window_target| {
        match &event {
            Event::WindowEvent { window_id: _, event: WindowEvent::RedrawRequested } => {
                let now = Instant::now();
                let frame_time = now.duration_since(last_frame).as_secs_f64();
                last_frame = now;
                window.set_title(&format!("Renderer | {} FPS", (1.0 / frame_time).trunc()));

                // vertex shader + misc
                let (model, view, proj, shadow_view, shadow_proj) = get_matrices(&config, Some(&camera));

                let uniforms = renderer::Uniforms {
                    model,
                    view,
                    proj,
                    shadow_view,
                    shadow_proj,
                    inv_view: view.inverse(),
                    inv_proj: proj.inverse(),
                    shadow_inv_view: shadow_view.inverse(),
                    shadow_inv_proj: shadow_proj.inverse(),
                    near_clipping_plane: config.n,
                    light_pos: config.light_pos,
                    cam_pos: camera.loc,
                    ambient: config.ambient,
                    diffuse: config.diffuse,
                    specular: config.specular,
                    light_intensity: config.light_intensity,
                    light_dissipation: config.light_dissipation,
                    shininess: config.shininess,
                    legacy: config.legacy,
                    shadow_mode: config.shadow_mode,
                    shadow_buf_width: config.shadow_buf_width,
                    shadow_buf_height: config.shadow_buf_height,
                    tan_theta: (config.fov.to_radians() / config.height as f64).tan(), // tan of the angle between two pixels
                    tex_sample_lerp: config.tex_sample_lerp,
                    wireframe: hold_wireframe,
                };

                // println!("{}", uniforms.tan_theta.atan().to_degrees());

                vertex_shader_pass(
                    &config.triangles,
                    &mut processed_tris,
                    &mut processed_tris_shadow,
                    &mut occlusion_testing,
                    &uniforms, dims,
                    Some(&mut pool), config.render_threads
                );

                update_light(
                    &light, &mut processed_light,
                    &mut dummy1, &mut dummy2, config.light_pos,
                    config.light_scale, &view,
                    &proj, config.n, dims
                );

                clear(&mut buf, config.clear_color);
                // end vertex shader + misc

                // rasterize
                let chunk_size_y = config.height / config.render_threads;
                let chunk_size_y_shadow = config.shadow_buf_height / config.render_threads;

                let shadow_depth_chunks = buf.shadow_depth.chunks_mut((chunk_size_y_shadow * config.shadow_buf_width) as usize);
                let occlusion_dummy = [];

                let color_chunks = buf.color.chunks_mut((chunk_size_y * dims.0) as usize * COLOR_BUF_CHANNELS);
                let depth_chunks = buf.depth.chunks_mut((chunk_size_y * dims.0) as usize);

                if config.shadow_mode == 2 {
                    pool.scoped(|spawner| {
                        for (i, shadow_depth) in shadow_depth_chunks.enumerate() {
                            let chunk_height = shadow_depth.len() as u32 / config.shadow_buf_width;
                            let mut sub_buf = SubBuffer {
                                color: None,
                                depth: shadow_depth,
                                dims: (config.shadow_buf_width, chunk_height),
                                start_y: i as u32 * chunk_size_y_shadow,
                            };

                            let processed_tris_shadow = processed_tris_shadow.as_ref();
                            let uniforms = &uniforms;
                            spawner.execute(move || {
                                for tri in processed_tris_shadow {
                                    match &tri.clip {
                                        Clip::Zero => {},
                                        Clip::One(second_triangle) => {
                                            renderer::rasterize(&mut sub_buf, &second_triangle, &occlusion_dummy, None, &uniforms.shadow_inv_view, &uniforms.shadow_inv_proj, uniforms);
                                        }
                                        Clip::Two => {}, // the clipped triangle has replaced the original, we can continue normally
                                        Clip::Three => continue,
                                    }

                                    renderer::rasterize(&mut sub_buf, tri, &occlusion_dummy, None, &uniforms.shadow_inv_view, &uniforms.shadow_inv_proj, uniforms);
                                }
                            });
                        }
                    });
                }

                // separate the loops to allow for the automatic drop of `shadow_depth_chunks`, we must borrow it immutably

                pool.scoped(|spawner| {
                    for (i, (color, depth)) in color_chunks.zip(depth_chunks).enumerate() {
                        let chunk_height = depth.len() as u32 / dims.0;
                        let mut sub_buf = SubBuffer {
                            color: Some(color),
                            depth,
                            dims: (dims.0, chunk_height), // all chunks are the same width, but not neccassarily the same height
                            start_y: i as u32 * chunk_size_y,
                        };

                        let processed_tris = processed_tris.as_ref();
                        let processed_light = processed_light.as_ref();
                        let occlusion_testing = occlusion_testing.as_ref();
                        let shadow_depth = &buf.shadow_depth;
                        let uniforms = &uniforms;
                        spawner.execute(move || {
                            for tri in processed_tris.iter().chain(processed_light.iter()) {
                                match &tri.clip {
                                    Clip::Zero => {},
                                    Clip::One(second_triangle) => {
                                        renderer::rasterize(&mut sub_buf, &second_triangle, &occlusion_testing, Some(shadow_depth), &uniforms.inv_view, &uniforms.inv_proj, uniforms);
                                    }
                                    Clip::Two => {}, // the clipped triangle has replaced the original, we can continue normally
                                    Clip::Three => continue,
                                }

                                renderer::rasterize(&mut sub_buf, tri, &occlusion_testing, Some(shadow_depth), &uniforms.inv_view, &uniforms.inv_proj, uniforms);
                            }
                        });
                    }
                });
                // end rasterize

                // present
                let pixels = frame_buffer.frame_mut();
                if show_shadow_depth {
                    color_from_shadow(&buf, pixels, (config.shadow_buf_width, config.shadow_buf_height));
                } else {
                    flip_and_copy(&buf, pixels, dims);
                }

                if let Err(e) = frame_buffer.render() {
                    println!("{}", e);
                    window_target.exit();
                }
                // end present
            }
            Event::DeviceEvent { event, .. } => {
                if let DeviceEvent::MouseMotion { delta } = event {
                    camera.update_look(*delta);
                    window.set_cursor_position(PhysicalPosition::new(config.width / 2, config.height / 2)).expect("Failed to set cursor position");
                }
            }
            _ => {}
        }

        if input.update(&event) {
            let now = Instant::now();
            let dt = now.duration_since(last_instant).as_secs_f64();
            last_instant = now;

            // Close events
            if input.key_pressed(KeyCode::Escape) || input.close_requested() {
                println!("Exiting.");
                window_target.exit();
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                if let Err(e) = frame_buffer.resize_surface(size.width, size.height) {
                    println!("{}", e);
                    window_target.exit();
                }
            }

            const ROT_SPEED: f64 = 3.0;
            if config.do_rotation {
                config.rot_y += dt * ROT_SPEED;
            }

            let (_, scroll) = input.scroll_diff();
            config.fov -= scroll as f64;

            camera.update_pos(dt, &input);
            window.request_redraw();

            if input.key_held(KeyCode::KeyR) {
                config.light_pos = camera.loc;
                config.light_pos.y -= config.light_scale;
                config.light_yaw = camera.yaw;
                config.light_pitch = camera.pitch;
            }

            if input.key_held(KeyCode::KeyF) {
                println!("Camera position: ({}, {}, {})", camera.loc.x, camera.loc.y, camera.loc.z);
            }

            if input.key_held(KeyCode::KeyV) {
                show_shadow_depth = true;
            } else {
                show_shadow_depth = false;
            }

            hold_wireframe = if config.wireframe { !input.key_held(KeyCode::KeyC) } else { input.key_held(KeyCode::KeyC) };
        }
    }).expect("Couldn't run event loop");
}

// set color buffer to the clear color
fn clear(buf: &mut Buffer, color: [u8; 3]) {
    for pix in buf.color.chunks_mut(COLOR_BUF_CHANNELS) {
        pix[0] = color[0];
        pix[1] = color[1];
        pix[2] = color[2];
    }

    for pix in buf.depth.iter_mut() {
        *pix = DEPTH_INIT;
    }

    for pix in buf.shadow_depth.iter_mut() {
        *pix = DEPTH_INIT;
    }
}

fn color_from_shadow(buf: &Buffer, frame: &mut [u8], dims: (u32, u32)) {
    for y in 0..dims.1 - 10 {
        for x in 0..dims.0 - 10 {
            if x % 2 == 0 && y % 2 == 0 {
                let index1 = (((y) * dims.0) + (x)) as usize;
                let index = (((y / 4) * dims.0) + (x / 2)) as usize;

                if index >= 1000000 {
                    continue;
                }
                frame[index * 4] = (buf.shadow_depth[index1] * 255.0) as u8;
                frame[index * 4 + 1] = (buf.shadow_depth[index1] * 255.0) as u8;
                frame[index * 4 + 2] = (buf.shadow_depth[index1] * 255.0) as u8;
                frame[index * 4 + 3] = renderer::FULLY_OPAQUE;
            }
        }
    }
}

// flip color buffer vertically, then copy into frame buffer
fn flip_and_copy(buf: &Buffer, frame: &mut [u8], dims: (u32, u32)) {
    let (width, height) = dims;

    for y in 0..height / 2 {
        for x in 0..width {
            let y2 = height - y - 1;
            let index_p2 = (width * y2 + x) as usize * COLOR_BUF_CHANNELS;
            let index_p = (width * y + x) as usize * COLOR_BUF_CHANNELS;
            let p2 = &buf.color[index_p2..index_p2 + COLOR_BUF_CHANNELS];
            let p = &buf.color[index_p..index_p + COLOR_BUF_CHANNELS];

            let index_p2 = (width * y2 + x) as usize * FRAME_BUF_CHANNELS;
            let index_p = (width * y + x) as usize * FRAME_BUF_CHANNELS;
            frame[index_p2] = p[0];
            frame[index_p2 + 1] = p[1];
            frame[index_p2 + 2] = p[2];
            frame[index_p2 + 3] = renderer::FULLY_OPAQUE;

            frame[index_p] = p2[0];
            frame[index_p + 1] = p2[1];
            frame[index_p + 2] = p2[2];
            frame[index_p + 3] = renderer::FULLY_OPAQUE;
        }
    }

    if height % 2 == 1 {
        let y = height / 2;
        for x in 0..width {
            let index_p = (width * y + x) as usize * COLOR_BUF_CHANNELS;
            let p = &buf.color[index_p..index_p + COLOR_BUF_CHANNELS];

            let index_p = (width * y + x) as usize * FRAME_BUF_CHANNELS;
            frame[index_p] = p[0];
            frame[index_p + 1] = p[1];
            frame[index_p + 2] = p[2];
            frame[index_p + 3] = renderer::FULLY_OPAQUE;
        }
    }
}

static LIGHT: [(Point3d, Vec3f); 36] = [
    (Point3d { x: -0.5, y: -0.5, z: -0.5 }, Vec3f { x:  0.0, y:  0.0, z: -1.0 }),
    (Point3d { x:  0.5, y: -0.5, z: -0.5 }, Vec3f { x:  0.0, y:  0.0, z: -1.0 }),
    (Point3d { x:  0.5, y:  0.5, z: -0.5 }, Vec3f { x:  0.0, y:  0.0, z: -1.0 }),
    (Point3d { x:  0.5, y:  0.5, z: -0.5 }, Vec3f { x:  0.0, y:  0.0, z: -1.0 }),
    (Point3d { x: -0.5, y:  0.5, z: -0.5 }, Vec3f { x:  0.0, y:  0.0, z: -1.0 }),
    (Point3d { x: -0.5, y: -0.5, z: -0.5 }, Vec3f { x:  0.0, y:  0.0, z: -1.0 }),
    (Point3d { x: -0.5, y: -0.5, z:  0.5 }, Vec3f { x:  0.0, y:  0.0, z:  1.0 }),
    (Point3d { x:  0.5, y: -0.5, z:  0.5 }, Vec3f { x:  0.0, y:  0.0, z:  1.0 }),
    (Point3d { x:  0.5, y:  0.5, z:  0.5 }, Vec3f { x:  0.0, y:  0.0, z:  1.0 }),
    (Point3d { x:  0.5, y:  0.5, z:  0.5 }, Vec3f { x:  0.0, y:  0.0, z:  1.0 }),
    (Point3d { x: -0.5, y:  0.5, z:  0.5 }, Vec3f { x:  0.0, y:  0.0, z:  1.0 }),
    (Point3d { x: -0.5, y: -0.5, z:  0.5 }, Vec3f { x:  0.0, y:  0.0, z:  1.0 }),
    (Point3d { x: -0.5, y:  0.5, z:  0.5 }, Vec3f { x: -1.0, y:  0.0, z:  0.0 }),
    (Point3d { x: -0.5, y:  0.5, z: -0.5 }, Vec3f { x: -1.0, y:  0.0, z:  0.0 }),
    (Point3d { x: -0.5, y: -0.5, z: -0.5 }, Vec3f { x: -1.0, y:  0.0, z:  0.0 }),
    (Point3d { x: -0.5, y: -0.5, z: -0.5 }, Vec3f { x: -1.0, y:  0.0, z:  0.0 }),
    (Point3d { x: -0.5, y: -0.5, z:  0.5 }, Vec3f { x: -1.0, y:  0.0, z:  0.0 }),
    (Point3d { x: -0.5, y:  0.5, z:  0.5 }, Vec3f { x: -1.0, y:  0.0, z:  0.0 }),
    (Point3d { x:  0.5, y:  0.5, z:  0.5 }, Vec3f { x:  1.0, y:  0.0, z:  0.0 }),
    (Point3d { x:  0.5, y:  0.5, z: -0.5 }, Vec3f { x:  1.0, y:  0.0, z:  0.0 }),
    (Point3d { x:  0.5, y: -0.5, z: -0.5 }, Vec3f { x:  1.0, y:  0.0, z:  0.0 }),
    (Point3d { x:  0.5, y: -0.5, z: -0.5 }, Vec3f { x:  1.0, y:  0.0, z:  0.0 }),
    (Point3d { x:  0.5, y: -0.5, z:  0.5 }, Vec3f { x:  1.0, y:  0.0, z:  0.0 }),
    (Point3d { x:  0.5, y:  0.5, z:  0.5 }, Vec3f { x:  1.0, y:  0.0, z:  0.0 }),
    (Point3d { x: -0.5, y: -0.5, z: -0.5 }, Vec3f { x:  0.0, y: -1.0, z:  0.0 }),
    (Point3d { x:  0.5, y: -0.5, z: -0.5 }, Vec3f { x:  0.0, y: -1.0, z:  0.0 }),
    (Point3d { x:  0.5, y: -0.5, z:  0.5 }, Vec3f { x:  0.0, y: -1.0, z:  0.0 }),
    (Point3d { x:  0.5, y: -0.5, z:  0.5 }, Vec3f { x:  0.0, y: -1.0, z:  0.0 }),
    (Point3d { x: -0.5, y: -0.5, z:  0.5 }, Vec3f { x:  0.0, y: -1.0, z:  0.0 }),
    (Point3d { x: -0.5, y: -0.5, z: -0.5 }, Vec3f { x:  0.0, y: -1.0, z:  0.0 }),
    (Point3d { x: -0.5, y:  0.5, z: -0.5 }, Vec3f { x:  0.0, y:  1.0, z:  0.0 }),
    (Point3d { x:  0.5, y:  0.5, z: -0.5 }, Vec3f { x:  0.0, y:  1.0, z:  0.0 }),
    (Point3d { x:  0.5, y:  0.5, z:  0.5 }, Vec3f { x:  0.0, y:  1.0, z:  0.0 }),
    (Point3d { x:  0.5, y:  0.5, z:  0.5 }, Vec3f { x:  0.0, y:  1.0, z:  0.0 }),
    (Point3d { x: -0.5, y:  0.5, z:  0.5 }, Vec3f { x:  0.0, y:  1.0, z:  0.0 }),
    (Point3d { x: -0.5, y:  0.5, z: -0.5 }, Vec3f { x:  0.0, y:  1.0, z:  0.0 }),
];

fn get_light() -> Vec<renderer::Triangle<'static>> {
    let mut out = Vec::new();
    const WHITE: [u8; 3] = [255, 255, 255];

    for vert in LIGHT.chunks(3) {
        let ((a, n_a), (b, n_b), (c, n_c)) = (vert[0], vert[1], vert[2]);

        out.push(renderer::Triangle {
            a: renderer::Vertex {
                pos: a,
                pos_world: Point3d::origin(),
                pos_clip: Point3d::origin(),
                color: WHITE,
                n: n_a.inv(),
                tex: [0.0, 0.0]
            },
            b: renderer::Vertex {
                pos: b,
                pos_world: Point3d::origin(),
                pos_clip: Point3d::origin(),
                color: WHITE,
                n: n_b.inv(),
                tex: [0.0, 0.0]
            },
            c: renderer::Vertex {
                pos: c,
                pos_world: Point3d::origin(),
                pos_clip: Point3d::origin(),
                color: WHITE,
                n: n_c.inv(),
                tex: [0.0, 0.0],
            },
            tex: None,
            // these values are intitalized in the vertex shader
            clip: Clip::Zero,
        });
    }

    out
}

fn update_light<'a>(
    light: &[renderer::Triangle<'a>],
    processed_light: &mut [renderer::Triangle<'a>],
    dummy1: &mut [[Point3d; 3]],
    dummy2: &mut [renderer::Triangle<'a>],
    light_pos: Point3d,
    light_scale: f64,
    view: &Mat4f,
    proj: &Mat4f,
    n: f64, dims: (u32, u32)
) {
    let uniforms = renderer::Uniforms {
        model: Mat4f::from_trans_scale(light_pos.x, light_pos.y, light_pos.z, light_scale),
        view: *view,
        proj: *proj,
        inv_view: view.inverse(),
        near_clipping_plane: n,
        // the above fields are the only ones used by the vertex shader
        ..Default::default()
    };

    vertex_shader_pass(light, processed_light, dummy2, dummy1, &uniforms, dims, None, 0)
}

fn render_to_image(config: &Config, save_name: &str) {
    let mut buf = Buffer {
        color: vec![0; (config.width * config.height) as usize * COLOR_BUF_CHANNELS].into_boxed_slice(),
        depth: vec![DEPTH_INIT; (config.width * config.height) as usize].into_boxed_slice(),
        shadow_depth: vec![DEPTH_INIT; (config.shadow_buf_width * config.shadow_buf_height) as usize].into_boxed_slice(),
    };
    clear(&mut buf, config.clear_color);

    let dims = (config.width, config.height);

    let (model, view, proj, shadow_view, shadow_proj) = get_matrices(&config, None);
    let uniforms = renderer::Uniforms {
        model,
        view,
        proj,
        shadow_view,
        shadow_proj,
        inv_view: view.inverse(),
        inv_proj: proj.inverse(),
        shadow_inv_view: shadow_view.inverse(),
        shadow_inv_proj: shadow_proj.inverse(),
        near_clipping_plane: config.n,
        light_pos: config.light_pos,
        cam_pos: Point3d::origin(),
        ambient: config.ambient,
        diffuse: config.diffuse,
        specular: config.specular,
        light_intensity: config.light_intensity,
        light_dissipation: config.light_dissipation,
        shininess: config.shininess,
        legacy: config.legacy,
        shadow_mode: config.shadow_mode,
        shadow_buf_width: config.shadow_buf_width,
        shadow_buf_height: config.shadow_buf_height,
        tan_theta: (config.fov.to_radians() / config.height as f64).tan(), // tan of the angle between two pixels
        tex_sample_lerp: config.tex_sample_lerp,
        wireframe: config.wireframe,
    };

    println!("{}", uniforms.tan_theta.atan().to_degrees());

    let tri_count = config.triangles.len();

    let show_progress = config.show_progress;
    if show_progress {
        println!("Performing vertex shader pass...");
    }

    let mut processed_tris = config.triangles.clone().into_boxed_slice();
    let mut processed_tris_shadow = config.triangles.clone().into_boxed_slice();
    let mut occlusion_testing = vec![[Point3d::default(); 3]; processed_tris.len()];
    vertex_shader_pass(&config.triangles, &mut processed_tris, &mut processed_tris_shadow, &mut occlusion_testing, &uniforms, dims, None, 0);

    if show_progress {
        println!("Done!");
    }

    // rows per chunk (except maybe last chunk)
    let chunk_size_y = config.height / config.render_threads;
    let color_chunks = buf.color.chunks_mut((chunk_size_y * dims.0) as usize * COLOR_BUF_CHANNELS);
    let depth_chunks = buf.depth.chunks_mut((chunk_size_y * dims.0) as usize);
    let threads_left: Mutex<Vec<_>> = Mutex::new((0..config.render_threads + if config.height % config.render_threads == 0 { 0 } else  { 1 }).collect());

    let start = Instant::now();
    std::thread::scope(|spawner| {
        for (i, (color, depth)) in color_chunks.zip(depth_chunks).enumerate() {
            let chunk_height = depth.len() as u32 / dims.0;
            let mut sub_buf = SubBuffer {
                color: Some(color),
                depth,
                dims: (dims.0, chunk_height), // all chunks are the same width, but not neccassarily the same height
                start_y: i as u32 * chunk_size_y,
            };

            let threads_left = &threads_left;
            let processed_tris = processed_tris.as_ref();
            let occlusion_testing = occlusion_testing.as_ref();
            let shadow_depth = &buf.shadow_depth;
            let uniforms = &uniforms;
            spawner.spawn(move || {
                let mut pixels_shaded = 0;
                for (j, tri) in processed_tris.iter().enumerate() {
                    // + 1 to prevent mod 0
                    if j % ((tri_count / 100) + 1) == 0 && show_progress {
                        println!(
                            "{:.2}% complete ({}/{} triangles rendered, {} pixels shaded) on thread {}",
                            (j as f64 / tri_count as f64) * 100.0,
                            j,
                            tri_count,
                            pixels_shaded,
                            i
                        );
                    }

                    match &tri.clip {
                        Clip::Zero => {},
                        Clip::One(second_triangle) => {
                            pixels_shaded += renderer::rasterize(&mut sub_buf, &second_triangle, &occlusion_testing, Some(shadow_depth), &uniforms.inv_view, &uniforms.inv_proj, uniforms);
                        }
                        Clip::Two => {},
                        Clip::Three => continue,
                    }

                    pixels_shaded += renderer::rasterize(&mut sub_buf, tri, &occlusion_testing, Some(shadow_depth), &uniforms.inv_view, &uniforms.inv_proj, uniforms);
                }

                let mut threads_left = threads_left.lock().expect("Failed to obtain threads_left lock");
                let index = threads_left.binary_search(&(i as u32)).unwrap();
                threads_left.remove(index);
                println!("Thread {} is complete! Threads {:?} are still running.", i, threads_left);
            });
        }
    });
    println!("Finished rendering {} triangles in {} secs.", tri_count, Instant::now().duration_since(start).as_secs_f64());

    let mut img = RgbImage::from_raw(config.width, config.height, buf.color.into_vec()).unwrap();
    image::imageops::flip_vertical_in_place(&mut img);
    img.save(save_name.to_string() + ".png").expect("Failed to save image");
}

fn get_config(name: &str) -> String {
    let config_txt = std::fs::read_to_string(name.to_owned() + ".txt").expect("Failed to locate config file");
    // remove whitespace and caridge returns
    config_txt.chars().filter(|&c| c != ' ' && c != '\r').collect()
}

pub struct MtlData {
    color: [u8; 3],
    tex: Option<RgbaImage>,
}

fn load_mtl_data(mtl: &Option<String>) -> HashMap<String, MtlData> {
    let mut mtls = HashMap::new();

    if let Some(mtl) = mtl {
        for mat in mtl.split("newmtl ").skip(1) {
            if let Some(name) = get_name(mat) {
                let color = get_color(mat).expect("mtl is missing diffuse color");
                let tex = get_tex(mat);
                mtls.insert(name, MtlData { color, tex });
            }
        }
    }

    mtls
}

fn get_name(mat: &str) -> Option<String> {
    Some(mat.split_at(mat.find('\n')?).0.trim().to_string())
}

fn get_color(mat: &str) -> Option<[u8; 3]> {
    for line in mat.lines() {
        if line.starts_with("Kd ") {
            let color_str = line.strip_prefix("Kd ").unwrap();
            let mut color_iter = color_str.split(' ');

            let mut color = [0; 3];
            color[0] = (color_iter.next().expect("Not enough channels in mtl color").parse::<f64>().expect("Failed to parse mtl color red") * 255.0) as u8;
            color[1] = (color_iter.next().expect("Not enough channels in mtl color").parse::<f64>().expect("Failed to parse mtl color green") * 255.0) as u8;
            color[2] = (color_iter.next().expect("Not enough channels in mtl color").parse::<f64>().expect("Failed to parse mtl color blue") * 255.0) as u8;

            return Some(color);
        }
    }

    None
}

fn get_tex(mat: &str) -> Option<RgbaImage> {
    for line in mat.lines() {
        if line.starts_with("map_Kd ") {
            let path = line.strip_prefix("map_Kd ").unwrap();
            let mut tex = image::load(
                BufReader::new(File::open(path).expect(&format!("Failed to open texture at {}", path))),
                image::ImageFormat::from_path(path).expect(&format!("No such image type at {}", path))
            ).expect(&format!("Failed to load texture at {}", path)).into_rgba8();
            image::imageops::flip_vertical_in_place(&mut tex);
            return Some(tex);
        }
    }

    None
}

fn get_matrices(config: &Config, camera: Option<&Camera>) -> (Mat4f, Mat4f, Mat4f, Mat4f, Mat4f) {
    if config.legacy {
        return (Mat4f::new(), Mat4f::new(), Mat4f::new(), Mat4f::new(), Mat4f::new());
    }

    let trans = Mat4f::from_trans_scale(config.trans_x, config.trans_y, config.trans_z, config.scale);
    let rot_x = Mat4f::from_rot_x(config.rot_x.to_radians());
    let rot_y = Mat4f::from_rot_y(config.rot_y.to_radians());
    let rot_z = Mat4f::from_rot_z(config.rot_z.to_radians());

    let mut model = math::mul_matrix_matrix(&trans, &rot_x);
    model = math::mul_matrix_matrix(&model, &rot_y);
    model = math::mul_matrix_matrix(&model, &rot_z);

    let view = if let Some(camera) = camera {
        math::view(&camera.loc, camera.yaw, camera.pitch)
    } else {
        Mat4f::new()
    };

    let shadow_view = if config.shadow_mode == 2 {
        math::view(&config.light_pos, config.light_yaw, config.light_pitch)
    } else {
        Mat4f::new()
    };

    let perspective = math::get_perspective(config.fov, config.width as f64 / config.height as f64, config.n, config.f);
    let proj = math::frustum(&perspective);

    let shadow_proj = if config.shadow_mode == 2 {
        let perspective = math::get_perspective(config.shadow_render_fov, config.shadow_buf_width as f64 / config.shadow_buf_height as f64, config.n, config.f); // NOTE: careful with n and f
        math::frustum(&perspective)
    } else {
        Mat4f::new()
    };

    (model, view, proj, shadow_view, shadow_proj)
}

// processed_tris will be filled with the output of the vertex shader
fn vertex_shader_pass<'a>(
    tris: &[renderer::Triangle<'a>],
    processed_tris: &mut [renderer::Triangle<'a>],
    processed_tris_shadow: &mut [renderer::Triangle<'a>],
    occlusion_testing: &mut [[Point3d; 3]],
    u: &renderer::Uniforms,
    dims: (u32, u32),
    pool: Option<&mut Pool>,
    threads: u32,
) {
    assert_eq!(tris.len(), processed_tris.len());
    assert_eq!(tris.len(), processed_tris_shadow.len());
    assert_eq!(tris.len(), occlusion_testing.len());

    if let Some(pool) = pool {
        let chunk_size = (tris.len() / threads as usize).max(1);
        pool.scoped(|spawner| {
            let chunks = tris.chunks(chunk_size);
            let p_chunks = processed_tris.chunks_mut(chunk_size);
            let p_shadow_chunks = processed_tris_shadow.chunks_mut(chunk_size);
            let occlusion_chunks = occlusion_testing.chunks_mut(chunk_size);

            for (
                ((chunk, p_chunk),
                p_shadow_chunk),
                occlusion_chunk
            ) in chunks.zip(p_chunks).zip(p_shadow_chunks).zip(occlusion_chunks) {
                spawner.execute(move || {
                    for (((tri, processed_tri), processed_tri_shadow), occlusion_test) in chunk.iter().zip(p_chunk.iter_mut()).zip(p_shadow_chunk.iter_mut()).zip(occlusion_chunk.iter_mut()) {
                        process_tri(tri, processed_tri, processed_tri_shadow, occlusion_test, u, dims);
                    }
                });
            }
        });
    } else {
        for (((tri, processed_tri), processed_tri_shadow), occlusion_test) in tris.iter().zip(processed_tris.iter_mut()).zip(processed_tris_shadow.iter_mut()).zip(occlusion_testing.iter_mut()) {
            process_tri(tri, processed_tri, processed_tri_shadow, occlusion_test, u, dims);
        }
    }

    fn process_tri<'a>(
        tri: &renderer::Triangle<'a>,
        processed_tri: &mut renderer::Triangle<'a>,
        processed_tri_shadow: &mut renderer::Triangle<'a>,
        occlusion_test: &mut [Point3d; 3],
        u: &renderer::Uniforms,
        dims: (u32, u32)
    ) {
        let (mut processed, occlusion) = renderer::vertex_shader(tri, &u.model, &u.view, &u.proj, &u.inv_view, u, dims);

        renderer::sort_tri_points_y(&mut processed);
        if let Clip::One(tri) = &mut processed.clip {
            renderer::sort_tri_points_y(tri);
        }

        *occlusion_test = occlusion;
        *processed_tri = processed;

        if u.shadow_mode == 2 {
            let (mut processed_shadow, _) = renderer::vertex_shader(tri, &u.model, &u.shadow_view, &u.shadow_proj, &u.shadow_inv_view, u, (u.shadow_buf_width, u.shadow_buf_height));

            renderer::sort_tri_points_y(&mut processed_shadow);
            if let Clip::One(tri) = &mut processed_shadow.clip {
                renderer::sort_tri_points_y(tri);
            }

            *processed_tri_shadow = processed_shadow;
        }
    }
}