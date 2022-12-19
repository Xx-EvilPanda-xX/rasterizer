use std::mem::swap;
use config::Config;
use image::{RgbImage, Rgb};

mod config;

fn main() {
    let config_txt = std::fs::read_to_string("config.txt").expect("Failed to located config.txt");
    let config = Config::new(&config_txt);
    let mut img = RgbImage::new(config.width, config.height);
    let dims = (config.width, config.height);

    for triangle in config.triangles {
        rasterize(&mut img, triangle, dims);
    }

    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        println!("USAGE: {} [out_name]", args[0]);
        std::process::exit(-1);
    }
    img.save(args[1].clone() + ".png").expect("Failed to save image");
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
struct Point {
    x: f32,
    y: f32,
}

impl Point {
    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    fn from_u32(x: u32, y: u32) -> Self {
        Self {
            x: x as f32,
            y: y as f32,
        }
    }

    fn from_arr(a: [f32; 2]) -> Self {
        Self { x: a[0], y: a[1] }
    }
}

// if undefined slope, m = x intercept and b = infinity
#[derive(Clone, Copy, Debug)]
struct Line {
    m: f32,
    b: f32,
}

fn rasterize(buf: &mut RgbImage, mut tri: Triangle, dims: (u32, u32)) {
    scale_tri_points(&mut tri, dims);
    sort_tri_points_y(&mut tri);

    fn top_scanline(tri: &Triangle, y: u32) -> (f32, f32) {
        let ab = tri.a.x == tri.b.x;
        let ac = tri.a.x == tri.c.x;
        if !ab && !ac {
            let Line { m: m1, b: b1 } = line_from_points(tri.a, tri.b);
            let Line { m: m2, b: b2 } = line_from_points(tri.a, tri.c);
            let start = (y as f32 - b1) / m1;
            let end = (y as f32 - b2) / m2;
            (start, end)
        } else if !ab && ac {
            let Line { m: m1, b: b1 } = line_from_points(tri.a, tri.b);
            let start = (y as f32 - b1) / m1;
            let end = tri.c.x;
            (start, end)
        } else if ab && !ac {
            let Line { m: m2, b: b2 } = line_from_points(tri.a, tri.c);
            let start = tri.b.x;
            let end = (y as f32 - b2) / m2;
            (start, end)
        } else {
            // this doesn't matter since this part of the tri will be invisible
            (0.0, 0.0)
        }
    }

    fn bottom_scanline(tri: &Triangle, y: u32) -> (f32, f32) {
        let cb = tri.c.x == tri.b.x;
        let ca = tri.c.x == tri.a.x;
        if !cb && !ca {
            let Line { m: m1, b: b1 } = line_from_points(tri.c, tri.b);
            let Line { m: m2, b: b2 } = line_from_points(tri.c, tri.a);
            let start = (y as f32 - b1) / m1;
            let end = (y as f32 - b2) / m2;
            (start, end)
        } else if !cb && ca {
            let Line { m: m1, b: b1 } = line_from_points(tri.c, tri.b);
            let start = (y as f32 - b1) / m1;
            let end = tri.a.x;
            (start, end)
        } else if cb && !ca {
            let Line { m: m2, b: b2 } = line_from_points(tri.c, tri.a);
            let start = tri.b.x;
            let end = (y as f32 - b2) / m2;
            (start, end)
        } else {
            // this doesn't matter since this part of the tri will be invisible
            (0.0, 0.0)
        }
    }

    for i in 0..2 {
        let (start_y, end_y) = match i {
            // add one to start_y becuase the first scanline is a little weird
            0 => (tri.a.y.ceil() as u32, tri.b.y.ceil() as u32),
            1 => (tri.b.y.ceil() as u32, tri.c.y.ceil() as u32),
            _ => unreachable!(),
        };

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

            for x in start_x..end_x {
                if x < buf.width() && y < buf.height() {
                    let color = get_color(&tri, x, y);
                    buf.put_pixel(x, y, Rgb::from(color));
                }
            }
        }
    }
}

fn get_color(tri: &Triangle, x: u32, y: u32) -> [u8; 3] {
    let p = Point::from_u32(x, y);

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

    let a = [tri.color_a[0] as f32, tri.color_a[1] as f32,tri.color_a[2] as f32];
    let b = [tri.color_b[0] as f32, tri.color_b[1] as f32,tri.color_b[2] as f32];
    let c = [tri.color_c[0] as f32, tri.color_c[1] as f32,tri.color_c[2] as f32];
    [(a[0] * a_weight + b[0] * b_weight + c[0] * c_weight).round() as u8,
    (a[1] * a_weight + b[1] * b_weight + c[1] * c_weight).round() as u8,
    (a[2] * a_weight + b[2] * b_weight + c[2] * c_weight).round() as u8]
}

fn dist(a: Point, b: Point) -> f32 {
    let diff_x = a.x - b.x;
    let diff_y = a.y - b.y;
    (diff_x * diff_x + diff_y * diff_y).sqrt()
}

// find m and b from two points
fn line_from_points(p1: Point, p2: Point) -> Line {
    let dy = p1.y - p2.y;
    let dx = p1.x - p2.x;

    if dx == 0.0 {
        return Line { m: p1.x, b: f32::INFINITY };
    }

    let m = dy / dx;
    let b = p1.y - dy * p1.x / dx;

    Line { m, b }
}

// find the point of intersection of two lines
fn solve_lines(l1: Line, l2: Line) -> Point {
    // account for lines that are edge cases to the below formula
    if l1.b.is_infinite() {
        return Point::new(l1.m, eval(l2, l1.m, false))
    }
    if l2.b.is_infinite() {
        return Point::new(l2.m, eval(l1, l2.m, false));
    }
    if l1.m == 0.0 {
        return Point::new(eval(l2, l1.b, true), l1.b)
    }
    if l2.m == 0.0 {
        return Point::new(eval(l1, l2.b, true), l2.b)
    }

    let y = (l2.b * l1.m - l1.b * l2.m) / (l1.m - l2.m);
    let x = (y - l1.b) / l1.m;
    Point::new(x, y)
}

fn eval(l: Line, val: f32, solve_x: bool) -> f32 {
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

fn scale_tri_points(tri: &mut Triangle, dims: (u32, u32)) {
    tri.a.x *= dims.0 as f32;
    tri.b.x *= dims.0 as f32;
    tri.c.x *= dims.0 as f32;
    tri.a.y *= dims.1 as f32;
    tri.b.y *= dims.1 as f32;
    tri.c.y *= dims.1 as f32;
}