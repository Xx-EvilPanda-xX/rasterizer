use std::{mem::swap, io::BufReader, fs::File, collections::HashMap};
use config::Config;
use image::{RgbImage, RgbaImage, Rgb};
use math::{Mat4f, Vec3f};
use std::time::Instant;

mod config;
mod math;

// z coord of the "screen" (anything with a smaller z will not be shown)
const SCREEN_Z: f64 = -1.0;
const DEPTH_INIT: f64 = 1.0;

struct Buffer {
    color: RgbImage,
    depth: Box<[f64]>,
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

    let textures = load_textures(&obj);
    let config = Config::new(&config_str, &obj, &textures);

    let mut buf = Buffer {
        color: RgbImage::from_pixel(config.width, config.height, Rgb::from(config.clear_color)),
        depth: vec![DEPTH_INIT; config.width as usize * config.height as usize].into_boxed_slice(),
    };

    let uniforms = Uniforms {
        light_pos: config.light_pos,
        ambient: config.ambient,
        diffuse: config.diffuse,
        specular: config.specular,
        shininess: config.shininess,
        legacy: config.legacy
    };

    let matrices = get_matrices(&config);
    let tri_count = config.triangles.len();

    let start = Instant::now();
    for (i, triangle) in config.triangles.iter().enumerate() {
        if i % 1000 == 0 {
            println!("{:.2}% complete ({}/{} triangles rendered)", (i as f64 / tri_count as f64) * 100.0, i, tri_count);
        }

        rasterize(&mut buf, triangle, &uniforms, &matrices.0, &matrices.1);
    }
    println!("Finished rendering {} triangles in {} secs.", tri_count, Instant::now().duration_since(start).as_secs_f64());

    image::imageops::flip_vertical_in_place(&mut buf.color);
    buf.color.save(args[args.len() - 1].clone() + ".png").expect("Failed to save image");
}

fn get_config(name: &str) -> String {
    let config_txt = std::fs::read_to_string(name.to_owned() + ".txt").expect("Failed to locate config file");
    let mut config = String::new();

    for c in config_txt.chars() {
        if c != ' ' && c != '\r' {
            config.push(c);
        }
    }

    config
}

fn load_textures(obj: &Option<(String, String)>) -> HashMap<String, RgbaImage> {
    let mut textures = HashMap::new();

    if let Some(obj) = obj {
        for mat in obj.1.split("newmtl ") {
            if let Ok((key, tex)) = get_tex(mat) {
                textures.insert(key, tex);
            }
        }
    }

    textures
}

static MAP: &'static str = "map_Kd";

fn get_tex(mat: &str) -> Result<(String, RgbaImage), ()> {
    let key = mat.split_at(mat.find('\n').ok_or(())?).0.trim();
    let path = mat.split_at(mat.find(MAP).ok_or(())? + MAP.len() + 1).1;
    let path = path.split_at(path.find("\n").unwrap_or(path.len())).0.trim();
    let mut tex = image::load(
        BufReader::new(File::open(path).expect(&format!("Failed to open texture at {}", path))),
        image::ImageFormat::from_path(path).expect(&format!("No such image type at {}", path))
    ).expect(&format!("Failed to load texture at {}", path)).into_rgba8();
    image::imageops::flip_vertical_in_place(&mut tex);
    Ok((key.to_owned(), tex))
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

#[derive(Clone, Copy, Debug)]
pub struct Point {
    x: f64,
    y: f64,
    z: f64,
}

// if undefined slope, m = x intercept and b = infinity
#[derive(Clone, Copy, Debug)]
struct Line {
    m: f64,
    b: f64,
}

#[derive(Clone, Copy, Debug)]
struct Plane {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

#[derive(Clone, Debug)]
pub struct Triangle<'a> {
    a: Vertex,
    b: Vertex,
    c: Vertex,
    tex: Option<&'a RgbaImage>,
}

#[derive(Clone, Debug)]
pub struct Vertex {
    pos: Point,
    pos_world: Point,
    color: [u8; 3],
    n: Vec3f,
    tex: [f64; 2],
}

pub struct AttributePlanes {
    color_r: Plane,
    color_g: Plane,
    color_b: Plane,
    n_x: Plane,
    n_y: Plane,
    n_z: Plane,
    tex_x: Plane,
    tex_y: Plane,
    world_x: Plane,
    world_y: Plane,
    world_z: Plane,
    depth: Plane,
}

struct Uniforms {
    light_pos: Vec3f,
    ambient: f64,
    diffuse: f64,
    specular: f64,
    shininess: u32,
    legacy: bool,
}

impl Point {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn new_xy(x: f64, y: f64) -> Self {
        Self { x, y, z: 0.0 }
    }

    fn origin() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    fn from_arr(a: [f64; 3]) -> Self {
        Self { x: a[0], y: a[1], z: a[2] }
    }

    fn into_vec(&self) -> Vec3f {
        Vec3f { x: self.x, y: self.y, z: self.z }
    }
}

fn rasterize(buf: &mut Buffer, triangle: &Triangle, u: &Uniforms, model: &Mat4f, proj: &Mat4f) {
    let dims = (buf.color.width(), buf.color.height());
    let mut tri = triangle.clone();
    if vertex_shader(&mut tri, u, model, proj, dims) == VertexShaderResult::Clipped {
        // trianlge (at least one vertex) was clipped, we cannot continue
        return;
    };
    sort_tri_points_y(&mut tri);

    let (a, b, c) = (&tri.a.pos, &tri.b.pos, &tri.c.pos);
    let planes = AttributePlanes {
        color_r: plane_from_points_z(a, b, c, tri.a.color[0] as f64, tri.b.color[0] as f64, tri.c.color[0] as f64),
        color_g: plane_from_points_z(a, b, c, tri.a.color[1] as f64, tri.b.color[1] as f64, tri.c.color[1] as f64),
        color_b: plane_from_points_z(a, b, c, tri.a.color[2] as f64, tri.b.color[2] as f64, tri.c.color[2] as f64),
        n_x: plane_from_points_z(a, b, c, tri.a.n.x, tri.b.n.x, tri.c.n.x),
        n_y: plane_from_points_z(a, b, c, tri.a.n.y, tri.b.n.y, tri.c.n.y),
        n_z: plane_from_points_z(a, b, c, tri.a.n.z, tri.b.n.z, tri.c.n.z),
        tex_x: plane_from_points_z(a, b, c, tri.a.tex[0], tri.b.tex[0], tri.c.tex[0]),
        tex_y: plane_from_points_z(a, b, c, tri.a.tex[1], tri.b.tex[1], tri.c.tex[1]),
        world_x: plane_from_points_z(a, b, c, tri.a.pos_world.x, tri.b.pos_world.x, tri.c.pos_world.x),
        world_y: plane_from_points_z(a, b, c, tri.a.pos_world.y, tri.b.pos_world.y, tri.c.pos_world.y),
        world_z: plane_from_points_z(a, b, c, tri.a.pos_world.z, tri.b.pos_world.z, tri.c.pos_world.z),
        depth: plane_from_points_z(a, b, c, tri.a.pos.z, tri.b.pos.z, tri.c.pos.z),
    };

    for i in 0..2 {
        let (mut start_y, mut end_y) = match i {
            0 => (tri.a.pos.y.ceil() as u32, tri.b.pos.y.ceil() as u32),
            1 => (tri.b.pos.y.ceil() as u32, tri.c.pos.y.ceil() as u32),
            _ => unreachable!(),
        };

        start_y = start_y.clamp(0, dims.1 - 1);
        end_y = end_y.clamp(0, dims.1 - 1);
        for y in start_y..end_y {
            let (start_x, end_x) = match i {
                0 => top_scanline(&tri, y),
                1 => bottom_scanline(&tri, y),
                _ => unreachable!(),
            };

            // YOU CANNOT ROUND HERE. It creates situtations where the start and end x are outside our tri
            let mut start_x = start_x.ceil() as u32;
            let mut end_x = end_x.ceil() as u32;
            if start_x > end_x {
                swap(&mut start_x, &mut end_x);
            }

            start_x = start_x.clamp(0, dims.0 - 1);
            end_x = end_x.clamp(0, dims.0 - 1);
            for x in start_x..end_x {
                let (color, depth) = pixel_shader(&tri, u, &planes, x, y);

                // discard transparency
                if color[3] == 0 {
                    continue;
                }

                let i = y * dims.0 + x;
                if depth < buf.depth[i as usize] && depth >= SCREEN_Z {
                    buf.color.put_pixel(x, y, Rgb::from([color[0], color[1], color[2]]));
                    buf.depth[i as usize] = depth;
                }
            };
        }
    }

    fn top_scanline(tri: &Triangle, y: u32) -> (f64, f64) {
        let (a, b, c) = (&tri.a.pos, &tri.b.pos, &tri.c.pos);
        let ab = a.x == b.x;
        let ac = a.x == c.x;
        if !ab && !ac {
            let ab = line_from_points(a, b);
            let ac = line_from_points(a, c);
            let start = solve_x(&ab, y as f64);
            let end = solve_x(&ac, y as f64);
            (start, end)
        } else if !ab && ac {
            let ab = line_from_points(a, b);
            let start = solve_x(&ab, y as f64);
            let end = c.x;
            (start, end)
        } else if ab && !ac {
            let ac = line_from_points(a, c);
            let start = b.x;
            let end = solve_x(&ac, y as f64);
            (start, end)
        } else {
            // this doesn't matter since this part of the tri will be invisible
            (0.0, 0.0)
        }
    }

    fn bottom_scanline(tri: &Triangle, y: u32) -> (f64, f64) {
        let (a, b, c) = (&tri.a.pos, &tri.b.pos, &tri.c.pos);
        let cb = c.x == b.x;
        let ca = c.x == a.x;
        if !cb && !ca {
            let cb = line_from_points(c, b);
            let ca = line_from_points(c, a);
            let start = solve_x(&cb, y as f64);
            let end = solve_x(&ca, y as f64);
            (start, end)
        } else if !cb && ca {
            let cb = line_from_points(c, b);
            let start = solve_x(&cb, y as f64);
            let end = a.x;
            (start, end)
        } else if cb && !ca {
            let ca = line_from_points(c, a);
            let start = b.x;
            let end = solve_x(&ca, y as f64);
            (start, end)
        } else {
            // this doesn't matter since this part of the tri will be invisible
            (0.0, 0.0)
        }
    }
}

#[derive(PartialEq, Eq)]
enum VertexShaderResult {
    Ok,
    Clipped,
}

// returns: triangle in world space (or nothing if tri is clipped)
fn vertex_shader(tri: &mut Triangle, u: &Uniforms, model: &Mat4f, proj: &Mat4f, dims: (u32, u32)) -> VertexShaderResult {
    let (a, b, c) = (&mut tri.a, &mut tri.b, &mut tri.c);

    a.pos = math::mul_point_matrix(&a.pos, model);
    b.pos = math::mul_point_matrix(&b.pos, model);
    c.pos = math::mul_point_matrix(&c.pos, model);

    a.pos_world = a.pos;
    b.pos_world = b.pos;
    c.pos_world = c.pos;

    // no non-uniform scaling is actually done to our points, so normals will be fine too.
    a.n = math::mul_point_matrix(&a.n.into_point(), &model.no_trans()).into_vec().normalize();
    b.n = math::mul_point_matrix(&b.n.into_point(), &model.no_trans()).into_vec().normalize();
    c.n = math::mul_point_matrix(&c.n.into_point(), &model.no_trans()).into_vec().normalize();

    // primitive implementation of clipping (so z !>= 0 for perspective division, otherwise weird stuff unfolds)
    if (a.pos.z >= 0.0 || b.pos.z >= 0.0 || c.pos.z >= 0.0) && !u.legacy {
        return VertexShaderResult::Clipped;
    }

    a.pos = math::mul_point_matrix(&a.pos, proj);
    b.pos = math::mul_point_matrix(&b.pos, proj);
    c.pos = math::mul_point_matrix(&c.pos, proj);

    // normalize to 0 to 1 and scale to raster space
    a.pos.x = (a.pos.x + 1.0) / 2.0 * dims.0 as f64;
    b.pos.x = (b.pos.x + 1.0) / 2.0 * dims.0 as f64;
    c.pos.x = (c.pos.x + 1.0) / 2.0 * dims.0 as f64;
    a.pos.y = (a.pos.y + 1.0) / 2.0 * dims.1 as f64;
    b.pos.y = (b.pos.y + 1.0) / 2.0 * dims.1 as f64;
    c.pos.y = (c.pos.y + 1.0) / 2.0 * dims.1 as f64;
    VertexShaderResult::Ok
}

const FULLY_OPAQUE: u8 = 255;
const INTERPOLATE_FAST: bool = true;

/*
interpolate_fast treats our attributes as the z values of our
triangles, then solves for z when x and y are known at any
arbitrary point in that plane.

interpolate_slow finds the maximum distance away from a vertex
that a point on our triangle can be, as well as the actual distance.
From there, it can produce a "weight" for each vertex, to be
multiplied with our attributes.
*/

fn pixel_shader(tri: &Triangle, u: &Uniforms, planes: &AttributePlanes, x: u32, y: u32) -> ([u8; 4], f64) {
    if INTERPOLATE_FAST {
        let base_color = if let Some(tex) = tri.tex {
            let vt_x = lerp_fast(&planes.tex_x, x, y);
            let vt_y = lerp_fast(&planes.tex_y, x, y);
            tex_sample(tex, vt_x, vt_y)
        } else {
            [lerp_fast(&planes.color_r, x, y).round() as u8,
            lerp_fast(&planes.color_g, x, y).round() as u8,
            lerp_fast(&planes.color_b, x, y).round() as u8,
            FULLY_OPAQUE]
        };

        let norm = Vec3f::new(
            lerp_fast(&planes.n_x, x, y),
            lerp_fast(&planes.n_y, x, y),
            lerp_fast(&planes.n_z, x, y)
        );

        let pix_pos = Point::new(
            lerp_fast(&planes.world_x, x, y),
            lerp_fast(&planes.world_y, x, y),
            lerp_fast(&planes.world_z, x, y)
        );

        let color = mul_color(&base_color, calc_lighting(&norm, &pix_pos, u));
        (color, lerp_fast(&planes.depth, x, y))
    } else {
        let weights = lerp_slow(tri, x, y);

        let base_color = if let Some(tex) = tri.tex {
            let vt_x = tri.a.tex[0] * weights.a + tri.b.tex[0] * weights.b + tri.c.tex[0] * weights.c;
            let vt_y = tri.a.tex[1] * weights.a + tri.b.tex[1] * weights.b + tri.c.tex[1] * weights.c;
            tex_sample(tex, vt_x, vt_y)
        } else {
            let a = [tri.a.color[0] as f64, tri.a.color[1] as f64, tri.a.color[2] as f64];
            let b = [tri.b.color[0] as f64, tri.b.color[1] as f64, tri.b.color[2] as f64];
            let c = [tri.c.color[0] as f64, tri.c.color[1] as f64, tri.c.color[2] as f64];
            [(a[0] * weights.a + b[0] * weights.b + c[0] * weights.c).round() as u8,
            (a[1] * weights.a + b[1] * weights.b + c[1] * weights.c).round() as u8,
            (a[2] * weights.a + b[2] * weights.b + c[2] * weights.c).round() as u8,
            FULLY_OPAQUE]
        };

        let norm = Vec3f::new(
            tri.a.n.x * weights.a + tri.b.n.x * weights.b + tri.c.n.x * weights.c,
            tri.a.n.y * weights.a + tri.b.n.y * weights.b + tri.c.n.y * weights.c,
            tri.a.n.z * weights.a + tri.b.n.z * weights.b + tri.c.n.z * weights.c,
        );

        let pix_pos = Point::new(
            tri.a.pos_world.x * weights.a + tri.b.pos_world.x * weights.b + tri.c.pos_world.x * weights.c,
            tri.a.pos_world.y * weights.a + tri.b.pos_world.y * weights.b + tri.c.pos_world.y * weights.c,
            tri.a.pos_world.z * weights.a + tri.b.pos_world.z * weights.b + tri.c.pos_world.z * weights.c,
        );

        let color = mul_color(&base_color, calc_lighting(&norm, &pix_pos, u));
        (color, tri.a.pos.z * weights.a + tri.b.pos.z * weights.b + tri.c.pos.z * weights.c)
    }
}

fn calc_lighting(norm: &Vec3f, pix_pos: &Point, u: &Uniforms) -> f64 {
    // pixel to light
    let light_dir = (u.light_pos - pix_pos.into_vec()).normalize();

    let ambient = u.ambient;
    let diffuse = Vec3f::dot(&norm, &light_dir).max(0.0) * u.diffuse;

    // our cam is always at the origin, so view dir is just the pixel pos (cam to pixel)
    let view_dir = pix_pos.into_vec().normalize();
    let reflected = reflect(&light_dir.inv(), &norm);
    let specular = Vec3f::dot(&view_dir.inv(), &reflected).max(0.0).powi(u.shininess as i32) * u.specular;

    ambient + diffuse + specular
}

// all parameters must be normalized
fn reflect(incoming: &Vec3f, norm: &Vec3f) -> Vec3f {
    let inc = incoming.inv();
    // the cos of the angle between our incoming vec and our norm
    let cos_theta = Vec3f::dot(&inc, &norm);

    // the point along our norm with distance `cos_theta` from the origin
    // angle > 90 handles itself because our norm will be inverted due to a negative cos
    let norm_int = Point::new(
        lerp(0.0, norm.x, cos_theta),
        lerp(0.0, norm.y, cos_theta),
        lerp(0.0, norm.z, cos_theta)
    ).into_vec();

    // vector from our incoming vec to the above point
    let inc_norm_int = norm_int - inc;
    let reflected = norm_int + inc_norm_int;
    reflected
}

fn lerp(a: f64, b: f64, c: f64) -> f64 {
    a + c * (b - a)
}

fn mul_color(color: &[u8; 4], x: f64) -> [u8; 4] {
    [(color[0] as f64 * x) as u8,
    (color[1] as f64 * x) as u8,
    (color[2] as f64 * x) as u8,
    color[3]]
}

// slowest function by far
// optimizations needed
fn tex_sample(tex: &RgbaImage, x: f64, y: f64) -> [u8; 4] {
    // confine coords to be between 0 and 1
    let x = x.fract();
    let y = y.fract();
    let px = (x * (tex.width() - 1) as f64) as u32;
    let py = (y * (tex.height() - 1) as f64) as u32;
    tex.get_pixel(px, py).0
}

fn lerp_fast(p: &Plane, x: u32, y: u32) -> f64 {
    let point = Point::new_xy(x as f64, y as f64);
    let z = (p.d - p.a * point.x - p.b * point.y) / p.c;
    z
}

struct VertexWeights {
    a: f64,
    b: f64,
    c: f64,
}

fn lerp_slow(tri: &Triangle, x: u32, y: u32) -> VertexWeights {
    let (a, b, c) = (&tri.a.pos, &tri.b.pos, &tri.c.pos);
    let p = Point::new_xy(x as f64, y as f64);

    // sides of our triangle
    let ab = line_from_points(a, b);
    let bc = line_from_points(b, c);
    let ac = line_from_points(a, c);

    // distance from each vertex
    let dist_a = dist_2(&p, a);
    let dist_b = dist_2(&p, b);
    let dist_c = dist_2(&p, c);

    // line passing through each vertex and our point, then find the point of intersection with opposite side
    let ap = line_from_points(a, &p);
    let ap_bc = solve_lines(&ap, &bc);
    let max_dist_a = dist_2(a, &ap_bc);

    let bp = line_from_points(b, &p);
    let bp_ac = solve_lines(&bp, &ac);
    let max_dist_b = dist_2(b, &bp_ac);

    let cp = line_from_points(c, &p);
    let cp_ab = solve_lines(&cp, &ab);
    let max_dist_c = dist_2(c, &cp_ab);

    // weight vertices based off distance from the point
    let a_weight = 1.0 - (dist_a / max_dist_a).sqrt();
    let b_weight = 1.0 - (dist_b / max_dist_b).sqrt();
    let c_weight = 1.0 - (dist_c / max_dist_c).sqrt();

    VertexWeights { a: a_weight, b: b_weight, c: c_weight }
}

fn dist_2(a: &Point, b: &Point) -> f64 {
    let diff_x = a.x - b.x;
    let diff_y = a.y - b.y;
    diff_x * diff_x + diff_y * diff_y
}

// find m and b from two points
fn line_from_points(p1: &Point, p2: &Point) -> Line {
    let dy = p1.y - p2.y;
    let dx = p1.x - p2.x;

    if dx == 0.0 {
        return Line { m: p1.x, b: f64::INFINITY };
    }

    let m = dy / dx;
    let b = p1.y - dy * p1.x / dx;

    Line { m, b }
}

// only the x and y components of the points being passed in here are used, the z coming from the last args
fn plane_from_points_z(p1: &Point, p2: &Point, p3: &Point, z1: f64, z2: f64, z3: f64) -> Plane {
    plane_from_points(&Point::new(p1.x, p1.y, z1), &Point::new(p2.x, p2.y, z2), &Point::new(p3.x, p3.y, z3))
}

fn plane_from_points(p1: &Point, p2: &Point, p3: &Point) -> Plane {
    let v1 = Vec3f::new(p2.x, p2.y, p2.z) - Vec3f::new(p1.x, p1.y, p1.z);
    let v2 = Vec3f::new(p3.x, p3.y, p3.z) - Vec3f::new(p1.x, p1.y, p1.z);

    let n = Vec3f::cross(&v1, &v2);
    let d = n.x * p1.x + n.y * p1.y + n.z * p1.z;

    Plane {
        a: n.x,
        b: n.y,
        c: n.z,
        d,
    }
}

// find the point of intersection of two lines
fn solve_lines(l1: &Line, l2: &Line) -> Point {
    // account for lines that are edge cases to the below formula
    if l1.b.is_infinite() {
        return Point::new_xy(l1.m, solve_y(l2, l1.m));
    }
    if l2.b.is_infinite() {
        return Point::new_xy(l2.m, solve_y(l1, l2.m));
    }
    if l1.m == 0.0 {
        return Point::new_xy(solve_x(l2, l1.b), l1.b);
    }
    if l2.m == 0.0 {
        return Point::new_xy(solve_x(l1, l2.b), l2.b);
    }

    let y = (l2.b * l1.m - l1.b * l2.m) / (l1.m - l2.m);
    let x = solve_x(&l1, y);
    Point::new_xy(x, y)
}

fn solve_x(l: &Line, y: f64) -> f64 {
    (y - l.b) / l.m
}

fn solve_y(l: &Line, x: f64) -> f64 {
    l.m * x + l.b
}

fn sort_tri_points_y(tri: &mut Triangle) {
    if tri.a.pos.y > tri.b.pos.y {
        swap(&mut tri.a, &mut tri.b);
    }

    if tri.b.pos.y > tri.c.pos.y {
        swap(&mut tri.b, &mut tri.c);
    }

    if tri.a.pos.y > tri.b.pos.y {
        swap(&mut tri.a, &mut tri.b);
    }
}