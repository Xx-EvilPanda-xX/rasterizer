use std::{io::BufReader, fs::File, collections::HashMap, time::Instant};
use camera::Camera;
use image::{RgbImage, RgbaImage};
use math::*;
use config::Config;
use winit::{event_loop::{ControlFlow, EventLoop}, window::WindowBuilder, dpi::{LogicalSize, PhysicalPosition}, event::{Event, VirtualKeyCode, DeviceEvent}};
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
}

pub struct SubBuffer<'a> {
    color: &'a mut [u8],
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

    if config.render_shadows {
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

fn start_interactive(config: Config<'static>) {
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let size = LogicalSize::new(config.width as f64, config.height as f64);
    let window = WindowBuilder::new()
        .with_title("Renderer")
        .with_inner_size(size)
        .with_min_inner_size(size)
        .build(&event_loop)
        .unwrap();

    window.set_cursor_visible(false);

    let mut frame_buffer = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);

        Pixels::new(config.width, config.height, surface_texture).expect("Failed to create pixel buffer")
    };

    let dims = (config.width, config.height);

    let mut processed_tris = config.triangles.clone().into_boxed_slice();
    let mut buf = Buffer {
        color: vec![0; (config.width * config.height) as usize * COLOR_BUF_CHANNELS].into_boxed_slice(),
        depth: vec![DEPTH_INIT; (config.width * config.height) as usize].into_boxed_slice(),
    };

    let mut pool = Pool::new(config.render_threads);

    let mut last_instant = Instant::now();
    let mut last_frame = Instant::now();

    let mut camera = Camera::new(Point3d::origin(), 0.0, 0.0);

    event_loop.run(move |event, _, control_flow| {
        match &event {
            Event::RedrawRequested(_) => {
                let now = Instant::now();
                let frame_time = now.duration_since(last_frame).as_secs_f64();
                last_frame = now;
                window.set_title(&format!("Renderer | {} FPS", (1.0 / frame_time).trunc()));

                // vertex shader + misc
                let (model, view, proj) = get_matrices(&config, Some(&camera));
                let uniforms = renderer::Uniforms {
                    model,
                    view,
                    proj,
                    inv_view: view.inverse(),
                    inv_proj: proj.inverse(),
                    light_pos: config.light_pos,
                    cam_pos: camera.loc,
                    ambient: config.ambient,
                    diffuse: config.diffuse,
                    specular: config.specular,
                    shininess: config.shininess,
                    legacy: config.legacy,
                    render_shadows: config.render_shadows,
                    tex_sample_lerp: config.tex_sample_lerp,
                };

                vertex_shader_pass(&config.triangles, &mut processed_tris, &uniforms, dims, Some(&mut pool), config.render_threads);
                clear(&mut buf, config.clear_color);
                // end vertex shader + misc

                // rasterize
                let chunk_size_y = config.height / config.render_threads;
                let color_chunks = buf.color.chunks_mut((chunk_size_y * dims.0) as usize * COLOR_BUF_CHANNELS);
                let depth_chunks = buf.depth.chunks_mut((chunk_size_y * dims.0) as usize);

                pool.scoped(|spawner| {
                    for (i, (color, depth)) in color_chunks.zip(depth_chunks).enumerate() {
                        let chunk_height = depth.len() as u32 / dims.0;
                        let mut sub_buf = SubBuffer {
                            color,
                            depth,
                            dims: (dims.0, chunk_height), // all chunks are the same width, but not neccassarily the same height
                            start_y: i as u32 * chunk_size_y,
                        };

                        let processed_tris = processed_tris.as_ref();
                        let uniforms = &uniforms;
                        spawner.execute(move || {
                            for tri in processed_tris {
                                if tri.clipped {
                                    continue;
                                }

                                renderer::rasterize(&mut sub_buf, tri, processed_tris, uniforms);
                            }
                        });
                    }
                });
                // end rasterize

                // present
                let pixels = frame_buffer.get_frame_mut();
                flip_and_copy(&mut buf, pixels, dims);

                if let Err(e) = frame_buffer.render() {
                    println!("{}", e);
                    *control_flow = ControlFlow::Exit;
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
            if input.key_pressed(VirtualKeyCode::Escape) || input.quit() {
                println!("Bye!");
                *control_flow = ControlFlow::Exit;
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                if let Err(e) = frame_buffer.resize_surface(size.width, size.height) {
                    println!("{}", e);
                    *control_flow = ControlFlow::Exit;
                }
            }

            camera.update_pos(dt, &input);
            window.request_redraw();
        }
    });
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
}

fn render_to_image(config: &Config, save_name: &str) {
    let mut buf = Buffer {
        color: vec![0; (config.width * config.height) as usize * COLOR_BUF_CHANNELS].into_boxed_slice(),
        depth: vec![DEPTH_INIT; (config.width * config.height) as usize].into_boxed_slice(),
    };
    clear(&mut buf, config.clear_color);

    let dims = (config.width, config.height);

    let (model, view, proj) = get_matrices(&config, None);
    let uniforms = renderer::Uniforms {
        model,
        view,
        proj,
        inv_view: view.inverse(),
        inv_proj: proj.inverse(),
        light_pos: config.light_pos,
        cam_pos: Point3d::origin(),
        ambient: config.ambient,
        diffuse: config.diffuse,
        specular: config.specular,
        shininess: config.shininess,
        legacy: config.legacy,
        render_shadows: config.render_shadows,
        tex_sample_lerp: config.tex_sample_lerp,
    };

    let tri_count = config.triangles.len();

    let show_progress = config.show_progress;
    if show_progress {
        println!("Performing vertex shader pass...");
    }

    let mut processed_tris = config.triangles.clone().into_boxed_slice();
    vertex_shader_pass(&config.triangles, &mut processed_tris, &uniforms, dims, None, 0);

    if show_progress {
        println!("Done!");
    }

    // rows per chunk (except maybe last chunk)
    let chunk_size_y = config.height / config.render_threads;
    let mut color_chunks = buf.color.chunks_mut((chunk_size_y * dims.0) as usize * COLOR_BUF_CHANNELS);
    let mut depth_chunks = buf.depth.chunks_mut((chunk_size_y * dims.0) as usize);

    let start = Instant::now();
    std::thread::scope(|spawner| {
        for i in 0..config.render_threads {
            // obtain current chunks
            let color = color_chunks.next().unwrap();
            let depth = depth_chunks.next().unwrap();

            let chunk_height = depth.len() as u32 / dims.0;
            let mut sub_buf = SubBuffer {
                color,
                depth,
                dims: (dims.0, chunk_height), // all chunks are the same width, but not neccassarily the same height
                start_y: i * chunk_size_y,
            };

            let processed_tris = processed_tris.as_ref();
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

                    if tri.clipped {
                        continue;
                    }

                    pixels_shaded += renderer::rasterize(&mut sub_buf, tri, processed_tris, uniforms);
                }
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
    let mut config = String::new();

    // omit whitespace
    for c in config_txt.chars() {
        if c != ' ' && c != '\r' {
            config.push(c);
        }
    }

    config
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
        if line.starts_with("Kd") {
            let color_str = line.split_at(line.find(' ').unwrap() + 1).1;
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
        if line.starts_with("map_Kd") {
            let path = line.split_at(line.find(' ').unwrap() + 1).1;
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

fn get_matrices(config: &Config, camera: Option<&Camera>) -> (Mat4f, Mat4f, Mat4f) {
    if config.legacy {
        return (Mat4f::new(), Mat4f::new(), Mat4f::new());
    }

    let mut trans = Mat4f::new();
    let mut rot_x = Mat4f::new();
    let mut rot_y = Mat4f::new();
    let mut rot_z = Mat4f::new();

    let theta_x = config.rot_x.to_radians();
    let theta_y = config.rot_y.to_radians();
    let theta_z = config.rot_z.to_radians();

    trans[0][0] = config.scale;
    trans[1][1] = config.scale;
    trans[2][2] = config.scale;
    trans[0][3] = config.trans_x;
    trans[1][3] = config.trans_y;
    trans[2][3] = config.trans_z;

    rot_x[1][1] = theta_x.cos();
    rot_x[1][2] = -theta_x.sin();
    rot_x[2][1] = theta_x.sin();
    rot_x[2][2] = theta_x.cos();

    rot_y[2][2] = theta_y.cos();
    rot_y[0][0] = theta_y.cos();
    rot_y[0][2] = theta_y.sin();
    rot_y[2][0] = -theta_y.sin();

    rot_z[1][1] = theta_z.cos();
    rot_z[0][0] = theta_z.cos();
    rot_z[0][1] = -theta_z.sin();
    rot_z[1][0] = theta_z.sin();

    let mut model = math::mul_matrix_matrix(&trans, &rot_x);
    model = math::mul_matrix_matrix(&model, &rot_y);
    model = math::mul_matrix_matrix(&model, &rot_z);

    let view = if let Some(camera) = camera {
        math::view(&camera.loc, camera.yaw, camera.pitch)
    } else {
        Mat4f::new()
    };

    let perspective = math::get_perspective(config.fov, config.width as f64 / config.height as f64, config.n, config.f);
    let proj = math::frustum(&perspective);

    (model, view, proj)
}

// processed_tris will be filled with the output of the vertex shader
fn vertex_shader_pass<'a>(
    tris: &[renderer::Triangle<'a>],
    processed_tris: &mut [renderer::Triangle<'a>],
    u: &renderer::Uniforms,
    dims: (u32, u32),
    pool: Option<&mut Pool>,
    threads: u32,
) {
    assert_eq!(tris.len(), processed_tris.len());

    if let Some(pool) = pool {
        let chunk_size = tris.len() / threads as usize;
        pool.scoped(|spawner| {
            let chunks = tris.chunks(chunk_size);
            let p_chunks = processed_tris.chunks_mut(chunk_size);
            for (chunk, p_chunk) in chunks.zip(p_chunks) {
                spawner.execute(move || {
                    for (tri, processed_tri) in chunk.iter().zip(p_chunk.iter_mut()) {
                        process_tri(tri, processed_tri, u, dims);
                    }
                });
            }
        });
    } else {
        for (tri, processed_tri) in tris.iter().zip(processed_tris.iter_mut()) {
            process_tri(tri, processed_tri, u, dims);
        }
    }

    fn process_tri<'a>(tri: &renderer::Triangle<'a>, processed_tri: &mut renderer::Triangle<'a>, u: &renderer::Uniforms, dims: (u32, u32)) {
        let mut processed = renderer::vertex_shader(tri, u, dims);
        renderer::sort_tri_points_y(&mut processed);
        processed.ab = (processed.b.pos_world.into_vec() - processed.a.pos_world.into_vec()).normalize();
        processed.ba = (processed.a.pos_world.into_vec() - processed.b.pos_world.into_vec()).normalize();
        processed.ac = (processed.c.pos_world.into_vec() - processed.a.pos_world.into_vec()).normalize();
        processed.bc = (processed.c.pos_world.into_vec() - processed.b.pos_world.into_vec()).normalize();

        *processed_tri = processed;
    }
}