use std::mem::swap;
use config::Config;
use image::{RgbImage, Rgb};
use dyn_array::DynArray;
use math::Mat4f;
use std::time::Instant;

mod config;
mod math;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 && args.len() != 3 {
        println!("USAGE: {} [out_name] OPTIONAL: [obj_name]", args[0]);
        std::process::exit(-1);
    }

    let config_txt = std::fs::read_to_string("config.txt").expect("Failed to located config.txt");
    let obj_txt = if args.len() == 3 {
        Some(std::fs::read_to_string(&args[2]).expect("Failed to located obj file"))
    } else {
        None
    };

    let config = Config::new(&config_txt, obj_txt);
    let mut buf = Buffer {
        color: RgbImage::from_pixel(config.width, config.height, Rgb::from(config.clear_color)),
        depth: DynArray::new([config.width as usize, config.height as usize], 1.0),
    };
    let dims = (config.width, config.height);
    let tri_count = config.triangles.len();
    let matrices = get_matrices(&config);

    let start = Instant::now();
    for (i, triangle) in config.triangles.into_iter().enumerate() {
        if i % 1000 == 0 {
            println!("{:.2}% complete ({}/{} triangles rendered)", (i as f64 / tri_count as f64) * 100.0, i, tri_count);
        }
        rasterize(&mut buf, triangle, &matrices.0, &matrices.1, dims);
    }
    println!("Finished rendering {} triangles in {} secs.", tri_count, Instant::now().duration_since(start).as_secs_f64());

    image::imageops::flip_vertical_in_place(&mut buf.color);
    buf.color.save(args[1].clone() + ".png").expect("Failed to save image");
}

fn get_matrices(config: &Config) -> (Mat4f, Mat4f) {
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

    let perspective = math::get_perspective(config.fov, config.width as f64 / config.height  as f64, config.n, config.f);
    let proj = math::frustum(&perspective);

    (model, proj)
}

pub struct Triangle {
    a: Point,
    b: Point,
    c: Point,
    color_a: [u8; 3],
    color_b: [u8; 3],
    color_c: [u8; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct Point {
    x: f64,
    y: f64,
    z: f64,
}

impl Point {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn new_xy(x: f64, y: f64) -> Self {
        Self { x, y, z: 0.0 }
    }

    fn from_arr(a: [f64; 3]) -> Self {
        Self { x: a[0], y: a[1], z: a[2] }
    }
}

// if undefined slope, m = x intercept and b = infinity
#[derive(Clone, Copy, Debug)]
struct Line {
    m: f64,
    b: f64,
}

struct Buffer {
    color: RgbImage,
    depth: DynArray<f64, 2>,
}

fn rasterize(buf: &mut Buffer, mut tri: Triangle, model: &Mat4f, proj: &Mat4f, dims: (u32, u32)) {
    if !vertex_shader(&mut tri, model, proj, dims) {
        return;
    };
    sort_tri_points_y(&mut tri);

    fn top_scanline(tri: &Triangle, y: u32) -> (f64, f64) {
        let ab = tri.a.x == tri.b.x;
        let ac = tri.a.x == tri.c.x;
        if !ab && !ac {
            let Line { m: m1, b: b1 } = line_from_points(tri.a, tri.b);
            let Line { m: m2, b: b2 } = line_from_points(tri.a, tri.c);
            let start = (y as f64 - b1) / m1;
            let end = (y as f64 - b2) / m2;
            (start, end)
        } else if !ab && ac {
            let Line { m: m1, b: b1 } = line_from_points(tri.a, tri.b);
            let start = (y as f64 - b1) / m1;
            let end = tri.c.x;
            (start, end)
        } else if ab && !ac {
            let Line { m: m2, b: b2 } = line_from_points(tri.a, tri.c);
            let start = tri.b.x;
            let end = (y as f64 - b2) / m2;
            (start, end)
        } else {
            // this doesn't matter since this part of the tri will be invisible
            (0.0, 0.0)
        }
    }

    fn bottom_scanline(tri: &Triangle, y: u32) -> (f64, f64) {
        let cb = tri.c.x == tri.b.x;
        let ca = tri.c.x == tri.a.x;
        if !cb && !ca {
            let Line { m: m1, b: b1 } = line_from_points(tri.c, tri.b);
            let Line { m: m2, b: b2 } = line_from_points(tri.c, tri.a);
            let start = (y as f64 - b1) / m1;
            let end = (y as f64 - b2) / m2;
            (start, end)
        } else if !cb && ca {
            let Line { m: m1, b: b1 } = line_from_points(tri.c, tri.b);
            let start = (y as f64 - b1) / m1;
            let end = tri.a.x;
            (start, end)
        } else if cb && !ca {
            let Line { m: m2, b: b2 } = line_from_points(tri.c, tri.a);
            let start = tri.b.x;
            let end = (y as f64 - b2) / m2;
            (start, end)
        } else {
            // this doesn't matter since this part of the tri will be invisible
            (0.0, 0.0)
        }
    }

    for i in 0..2 {
        let (mut start_y, mut end_y) = match i {
            0 => (tri.a.y.ceil() as u32, tri.b.y.ceil() as u32),
            1 => (tri.b.y.ceil() as u32, tri.c.y.ceil() as u32),
            _ => unreachable!(),
        };

        start_y = start_y.clamp(0, buf.color.height() - 1);
        end_y = end_y.clamp(0, buf.color.height() - 1);
        for y in start_y..end_y {
            let (start_x, end_x) = match i {
                0 => top_scanline(&tri, y),
                1 => bottom_scanline(&tri, y),
                _ => unreachable!(),
            };

            let mut start_x = start_x.round() as u32;
            let mut end_x = end_x.round() as u32;
            if start_x > end_x {
                swap(&mut start_x, &mut end_x);
            }

            start_x = start_x.clamp(0, buf.color.width() - 1);
            end_x = end_x.clamp(0, buf.color.width() - 1);
            for x in start_x..end_x {
                let (color, depth) = get_color_and_depth(&tri, x, y);
                if depth < buf.depth[[x as usize, y as usize]] && depth >= -1.0 {
                    buf.color.put_pixel(x, y, Rgb::from(color));
                    buf.depth[[x as usize, y as usize]] = depth;
                }
            }
        }
    }
}

fn get_color_and_depth(tri: &Triangle, x: u32, y: u32) -> ([u8; 3], f64) {
    let p = Point::new_xy(x as f64 + 0.5, y as f64 + 0.5);

    // distance from each vertex
    let dist_a = dist(p, tri.a);
    let dist_b = dist(p, tri.b);
    let dist_c = dist(p, tri.c);

    // sides of the tri
    let ab = line_from_points(tri.a, tri.b);
    let bc = line_from_points(tri.b, tri.c);
    let ac = line_from_points(tri.a, tri.c);

    // line passing through each vertex and our point, then find the point of intersection with opposite side
    let ap = line_from_points(tri.a, p);
    let ap_bc = solve_lines(ap, bc);
    let max_dist_a = dist(tri.a, ap_bc);

    let bp = line_from_points(tri.b, p);
    let bp_ac = solve_lines(bp, ac);
    let max_dist_b = dist(tri.b, bp_ac);

    let cp = line_from_points(tri.c, p);
    let cp_ab = solve_lines(cp, ab);
    let max_dist_c = dist(tri.c, cp_ab);

    // weight vertices based off distance from the point
    let a_weight = 1.0 - dist_a / max_dist_a;
    let b_weight = 1.0 - dist_b / max_dist_b;
    let c_weight = 1.0 - dist_c / max_dist_c;

    let a = [tri.color_a[0] as f64, tri.color_a[1] as f64,tri.color_a[2] as f64];
    let b = [tri.color_b[0] as f64, tri.color_b[1] as f64,tri.color_b[2] as f64];
    let c = [tri.color_c[0] as f64, tri.color_c[1] as f64,tri.color_c[2] as f64];
    ([(a[0] * a_weight + b[0] * b_weight + c[0] * c_weight).round() as u8,
    (a[1] * a_weight + b[1] * b_weight + c[1] * c_weight).round() as u8,
    (a[2] * a_weight + b[2] * b_weight + c[2] * c_weight).round() as u8],
    tri.a.z * a_weight + tri.b.z * b_weight + tri.c.z * c_weight)
}

fn dist(a: Point, b: Point) -> f64 {
    let diff_x = a.x - b.x;
    let diff_y = a.y - b.y;
    (diff_x * diff_x + diff_y * diff_y).sqrt()
}

// find m and b from two points
fn line_from_points(p1: Point, p2: Point) -> Line {
    let dy = p1.y - p2.y;
    let dx = p1.x - p2.x;

    if dx == 0.0 {
        return Line { m: p1.x, b: f64::INFINITY };
    }

    let m = dy / dx;
    let b = p1.y - dy * p1.x / dx;

    Line { m, b }
}

// find the point of intersection of two lines
fn solve_lines(l1: Line, l2: Line) -> Point {
    // account for lines that are edge cases to the below formula
    if l1.b.is_infinite() {
        return Point::new_xy(l1.m, eval(l2, l1.m, false))
    }
    if l2.b.is_infinite() {
        return Point::new_xy(l2.m, eval(l1, l2.m, false));
    }
    if l1.m == 0.0 {
        return Point::new_xy(eval(l2, l1.b, true), l1.b)
    }
    if l2.m == 0.0 {
        return Point::new_xy(eval(l1, l2.b, true), l2.b)
    }

    let y = (l2.b * l1.m - l1.b * l2.m) / (l1.m - l2.m);
    let x = (y - l1.b) / l1.m;
    Point::new_xy(x, y)
}

fn eval(l: Line, val: f64, solve_x: bool) -> f64 {
    if solve_x {
        (val - l.b) / l.m
    } else {
        l.m * val + l.b
    }
}

fn sort_tri_points_y(tri: &mut Triangle) {
    if tri.a.y > tri.b.y {
        swap(&mut tri.a, &mut tri.b);
    }

    if tri.b.y > tri.c.y {
        swap(&mut tri.b, &mut tri.c);
    }

    if tri.a.y > tri.b.y {
        swap(&mut tri.a, &mut tri.b);
    }
}

fn vertex_shader(tri: &mut Triangle, model: &Mat4f, proj: &Mat4f, dims: (u32, u32)) -> bool {
    tri.a = math::mul_point_matrix(&tri.a, model);
    tri.b = math::mul_point_matrix(&tri.b, model);
    tri.c = math::mul_point_matrix(&tri.c, model);

    // primitive implementation of clipping (so z !>= 0 for perspective division, otherwise weird stuff unfolds)
    if tri.a.z >= 0.0 || tri.b.z >= 0.0 || tri.c.z >= 0.0 {
        return false;
    }

    tri.a = math::mul_point_matrix(&tri.a, proj);
    tri.b = math::mul_point_matrix(&tri.b, proj);
    tri.c = math::mul_point_matrix(&tri.c, proj);

    tri.a.x = (tri.a.x + 1.0) / 2.0;
    tri.b.x = (tri.b.x + 1.0) / 2.0;
    tri.c.x = (tri.c.x + 1.0) / 2.0;
    tri.a.y = (tri.a.y + 1.0) / 2.0;
    tri.b.y = (tri.b.y + 1.0) / 2.0;
    tri.c.y = (tri.c.y + 1.0) / 2.0;
    tri.a.z = (tri.a.z + 1.0) / 2.0;
    tri.b.z = (tri.b.z + 1.0) / 2.0;
    tri.c.z = (tri.c.z + 1.0) / 2.0;

    tri.a.x *= dims.0 as f64;
    tri.b.x *= dims.0 as f64;
    tri.c.x *= dims.0 as f64;
    tri.a.y *= dims.1 as f64;
    tri.b.y *= dims.1 as f64;
    tri.c.y *= dims.1 as f64;
    true
}