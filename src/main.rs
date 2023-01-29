use std::{io::BufReader, fs::File, collections::HashMap};
use std::time::Instant;
use image::{RgbImage, RgbaImage, Rgb};
use math::*;
use config::Config;

mod config;
mod math;
mod renderer;

// z coord of the "screen" (anything with a smaller z will not be shown)
const SCREEN_Z: f64 = -1.0;
const DEPTH_INIT: f64 = 1.0;

struct Buffer {
    color: RgbImage,
    depth: Box<[f64]>,
}

pub struct SubBuffer<'a> {
    color: &'a mut [u8],
    depth: &'a mut [f64],
    dims: (u32, u32),
    start_y: u32,
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 && args.len() != 4 {
        println!("USAGE: {} [config_name] OPTIONAL: [obj_name] [out_name]", args[0]);
        std::process::exit(-1);
    }

    let config_str = get_config(&args[1]);
    let obj = if args.len() == 4 {
        Some(
            (std::fs::read_to_string(args[2].clone() + ".obj").expect("Failed to locate obj file"),
            std::fs::read_to_string(args[2].clone() + ".mtl").expect("Failed to locate mtl file"))
        )
    } else {
        None
    };

    let mtls = load_mtl_data(&obj);
    let config = Config::new(&config_str, &obj, &mtls);

    if config.render_shadows {
        println!("WARNING: Shadow rendering is incredibly slow and time to render will increase with the sqaure of the triangle count. (recommendation: do not exceed 10000 tris at 4k res)");
    }

    let mut buf = Buffer {
        color: RgbImage::from_pixel(config.width, config.height, Rgb::from(config.clear_color)),
        depth: vec![DEPTH_INIT; config.width as usize * config.height as usize].into_boxed_slice(),
    };
    let dims = (buf.color.width() as u32, buf.color.height() as u32);

    let matrices = get_matrices(&config);
    let inv_proj = matrices.1.inverse();
    let uniforms = renderer::Uniforms {
        model: matrices.0,
        proj: matrices.1,
        inv_proj,
        light_pos: config.light_pos,
        ambient: config.ambient,
        diffuse: config.diffuse,
        specular: config.specular,
        shininess: config.shininess,
        legacy: config.legacy,
        render_shadows: config.render_shadows,
    };

    let mut tris = config.triangles;
    let tri_count = tris.len();

    let show_progress = config.show_progress;

    if show_progress {
        println!("Performing vertex shader pass...");
    }

    vertex_shader_pass(&mut tris, &uniforms, dims);

    if show_progress {
        println!("Done!");
    }

    const COLOR_BUF_CHANNELS: usize = 3;
    // rows per chunk (except maybe last chunk)
    let chunk_size_y = buf.color.height() / config.render_threads;
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
                dims: (dims.0, chunk_height), // all chunks are the same width
                start_y: i * chunk_size_y,
            };

            let tris = &tris;
            let uniforms = &uniforms;
            spawner.spawn(move || {
                let mut pixels_shaded = 0;
                for (j, tri) in tris.iter().enumerate() {
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

                    pixels_shaded += renderer::rasterize(&mut sub_buf, tri, tris, uniforms);
                }
            });
        }
    });
    println!("Finished rendering {} triangles in {} secs.", tri_count, Instant::now().duration_since(start).as_secs_f64());

    image::imageops::flip_vertical_in_place(&mut buf.color);
    buf.color.save(args[args.len() - 1].clone() + ".png").expect("Failed to save image");
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

fn load_mtl_data(obj: &Option<(String, String)>) -> HashMap<String, MtlData> {
    let mut mtls = HashMap::new();

    if let Some(obj) = obj {
        for mat in obj.1.split("newmtl ").skip(1) {
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

fn get_matrices(config: &Config) -> (Mat4f, Mat4f) {
    if config.legacy {
        return (Mat4f::new(), Mat4f::new());
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

    let perspective = math::get_perspective(config.fov, config.width as f64 / config.height as f64, config.n, config.f);
    let proj = math::frustum(&perspective);

    (model, proj)
}

fn vertex_shader_pass(tris: &mut [renderer::Triangle], u: &renderer::Uniforms, dims: (u32, u32)) {
    for tri in tris.iter_mut() {
        renderer::vertex_shader(tri, u, dims);
        renderer::sort_tri_points_y(tri);

        let (a, b, c) = (tri.a.pos_world.into_vec(), tri.b.pos_world.into_vec(), tri.c.pos_world.into_vec());
        tri.ab = (b - a).normalize();
        tri.ba = (a - b).normalize();
        tri.ac = (c - a).normalize();
        tri.bc = (c - b).normalize();
    }
}