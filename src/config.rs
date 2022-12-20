use std::str::FromStr;
use core::fmt::Debug;
use super::{Point, Triangle};

pub struct Config {
    pub width: u32,
    pub height: u32,
    pub fov: f64,
    pub n: f64,
    pub f: f64,
    pub scale: f64,
    pub rot_x: f64,
    pub rot_y: f64,
    pub rot_z: f64,
    pub triangles: Vec<Triangle>,
    pub trans_x: f64,
    pub trans_y: f64,
    pub trans_z: f64,
}

impl Config {
    pub fn new(text: &str, obj: Option<String>) -> Self {
        let mut config = String::new();
        for c in text.chars() {
            if c != ' ' && c != '\r' {
                config.push(c);
            }
        }

        let mut config_processed = String::new();
        for line in config.lines() {
            if line.starts_with("//") {
                continue;
            } else {
                config_processed.push_str(line);
                config_processed.push('\n');
            }
        }

        let mut sections = config_processed.split_inclusive("\n\n");
        let mut img_config = sections.next().expect("missing image config");

        let width = field("width", &mut img_config);
        let height = field("height", &mut img_config);
        let fov = field("fov", &mut img_config);
        let n = field("n", &mut img_config);
        let f = field("f", &mut img_config);
        let scale = field("scale", &mut img_config);
        let rot_x = field("rot_x", &mut img_config);
        let rot_y = field("rot_y", &mut img_config);
        let rot_z = field("rot_z", &mut img_config);
        let trans_x = field("trans_x", &mut img_config);
        let trans_y = field("trans_y", &mut img_config);
        let trans_z = field("trans_z", &mut img_config);

        let mut triangles = Vec::new();

        if let Some(text) = obj {
            triangles = load_obj(&text);
        } else {
            for section in sections {
                triangles.push(Triangle::from_config(section));
            }
        }

        Self {
            width: width.parse().expect("Failed to parse width"),
            height: height.parse().expect("Failed to parse height"),
            fov: fov.parse().expect("Failed to parse height"),
            n: n.parse().expect("Failed to parse n"),
            f: f.parse().expect("Failed to parse f"),
            scale: scale.parse().expect("Failed to parse scale"),
            rot_x: rot_x.parse().expect("Failed to parse rot_x"),
            rot_y: rot_y.parse().expect("Failed to parse rot_y"),
            rot_z: rot_z.parse().expect("Failed to parse rot_z"),
            trans_x: trans_x.parse().expect("Failed to parse trans_x"),
            trans_y: trans_y.parse().expect("Failed to parse trans_y"),
            trans_z: trans_z.parse().expect("Failed to parse trans_z"),
            triangles,
        }
    }
}

impl Triangle {
    pub fn from_config(mut config: &str) -> Self {
        let a = field("a", &mut config);
        let b = field("b", &mut config);
        let c = field("c", &mut config);
        let color_a = field("color_a", &mut config);
        let color_b = field("color_b", &mut config);
        let color_c = field("color_c", &mut config);

        Self {
            a: Point::from_arr(parse_arr(a)),
            b: Point::from_arr(parse_arr(b)),
            c: Point::from_arr(parse_arr(c)),
            color_a: parse_arr(color_a),
            color_b: parse_arr(color_b),
            color_c: parse_arr(color_c),
        }
    }
}

fn parse_arr<T: Copy + Default + FromStr, const N: usize>(mut s: &str) -> [T; N]
        where <T as FromStr>::Err: Debug
{
    s = s.strip_prefix('[').expect("arrays start with `[`");
    s = s.strip_suffix(']').expect("arrays end with `[`");

    let mut elems = s.split(',');
    let mut out = [T::default(); N];
    for i in 0..N {
        let e = elems.next().expect("not enough array elements");
        out[i] = e.parse().expect("Failed to parse array element");
    }

    if elems.next().is_some() {
        panic!("too many array elements");
    }

    out
}

fn field<'a>(name: &str, args: &mut &'a str) -> &'a str {
    *args = args.strip_prefix(&format!("{}=", name)).expect(&format!("missing {}", name));
    let newline = args.find('\n').expect("missing newline");
    let f = &args[..newline];
    *args = &args[newline + 1..];
    f
}

fn load_obj(text: &str) -> Vec<Triangle> {
    let mut vertices = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for line in text.lines() {
        if line.starts_with("v ") {
            let v = line.strip_prefix("v ").unwrap();
            let mut elems = v.split(' ');
            let x = elems.next().unwrap().trim().parse().unwrap();
            let y = elems.next().unwrap().trim().parse().unwrap();
            let z = elems.next().unwrap().trim().parse().unwrap();
            vertices.push(Point::new(x, y, z));
        } else if line.starts_with("f ") {
            let f = line.strip_prefix("f ").unwrap();
            let mut elems = f.split(' ');
            let idx = [elems.next().unwrap(), elems.next().unwrap(), elems.next().unwrap()];
            for i in idx {
                // minus one because obj file indices start at 1
                indices.push(i.split_at(i.find('/').unwrap()).0.parse::<u32>().unwrap() - 1);
            }
        }
    }

    let mut triangles = Vec::new();

    for tri in indices.chunks(3) {
        let a = vertices[tri[0] as usize];
        let b = vertices[tri[1] as usize];
        let c = vertices[tri[2] as usize];

        triangles.push(Triangle {
            a, b, c,
            color_a: [((a.x * 255.0).abs() as u32 % 255) as u8, ((a.y * 255.0).abs() as u32 % 255) as u8, ((a.z * 255.0).abs() as u32 % 255) as u8],
            color_b: [((a.x * 255.0).abs() as u32 % 255) as u8, ((a.y * 255.0).abs() as u32 % 255) as u8, ((a.z * 255.0).abs() as u32 % 255) as u8],
            color_c: [((a.x * 255.0).abs() as u32 % 255) as u8, ((a.y * 255.0).abs() as u32 % 255) as u8, ((a.z * 255.0).abs() as u32 % 255) as u8],
            // smooth shading
            // color_a: [((a.x * 255.0).abs() as u32 % 255) as u8, ((a.y * 255.0).abs() as u32 % 255) as u8, ((a.z * 255.0).abs() as u32 % 255) as u8],
            // color_b: [((b.x * 255.0).abs() as u32 % 255) as u8, ((b.y * 255.0).abs() as u32 % 255) as u8, ((b.z * 255.0).abs() as u32 % 255) as u8],
            // color_c: [((c.x * 255.0).abs() as u32 % 255) as u8, ((c.y * 255.0).abs() as u32 % 255) as u8, ((c.z * 255.0).abs() as u32 % 255) as u8],
        });
    }

    triangles
}