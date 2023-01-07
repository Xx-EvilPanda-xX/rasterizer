use std::{str::FromStr, collections::HashMap};

use core::fmt::Debug;
use image::RgbaImage;

use super::{Point, Triangle, Vertex, math::Vec3f};

pub struct Config<'a> {
    pub width: u32,
    pub height: u32,
    pub clear_color: [u8; 3],
    pub fov: f64,
    pub n: f64,
    pub f: f64,
    pub scale: f64,
    pub rot_x: f64,
    pub rot_y: f64,
    pub rot_z: f64,
    pub trans_x: f64,
    pub trans_y: f64,
    pub trans_z: f64,
    pub light_dir: Vec3f,
    pub ambient: f64,
    pub diffuse: f64,
    pub specular: f64,
    pub shininess: u32,
    pub triangles: Vec<Triangle<'a>>,
}

impl<'a> Config<'a> {
    pub fn new(text: &str, obj: &Option<(String, String)>, textures: &'a HashMap<String, RgbaImage>) -> Self {
        let mut config = String::new();
        for line in text.lines() {
            if line.starts_with("//") {
                continue;
            } else {
                config.push_str(line);
                config.push('\n');
            }
        }

        let mut sections = config.split_inclusive("\n\n");
        let mut img_config = sections.next().expect("missing image config");

        let width = field("width", &mut img_config);
        let height = field("height", &mut img_config);
        let clear_color = field("clear_color", &mut img_config);
        let color_freq = field("color_freq", &mut img_config);
        let shade_mode = field("shade_mode", &mut img_config);
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
        let light_dir = field("light_dir", &mut img_config);
        let ambient = field("ambient", &mut img_config);
        let diffuse = field("diffuse", &mut img_config);
        let specular = field("specular", &mut img_config);
        let shininess = field("shininess", &mut img_config);

        let mut triangles = Vec::new();

        if let Some(text) = obj {
            triangles = load_obj(
                &text.0,
                color_freq.parse().expect("Failed to parse color_freq"),
                shade_mode.parse().expect("Failed to parse shade_mode"),
                textures
            );
        } else {
            for section in sections {
                triangles.push(Triangle::from_config(section));
            }
        }

        Self {
            width: width.parse().expect("Failed to parse width"),
            height: height.parse().expect("Failed to parse height"),
            clear_color: parse_arr(clear_color),
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
            light_dir: Vec3f::from_arr(parse_arr(light_dir)),
            ambient: ambient.parse().expect("Failed to parse ambient"),
            diffuse: diffuse.parse().expect("Failed to parse diffuse"),
            specular: specular.parse().expect("Failed to parse specular"),
            shininess: shininess.parse().expect("Failed to parse shininess"),
            triangles,
        }
    }
}

impl<'a> Triangle<'a> {
    pub fn from_config(mut config: &str, /* texture: Option<&'a image::RgbaImage> */) -> Self {
        let a = Point::from_arr(parse_arr(field("a", &mut config)));
        let b = Point::from_arr(parse_arr(field("b", &mut config)));
        let c = Point::from_arr(parse_arr(field("c", &mut config)));
        let color_a = parse_arr(field("color_a", &mut config));
        let color_b = parse_arr(field("color_b", &mut config));
        let color_c = parse_arr(field("color_c", &mut config));
        let tex_a = parse_arr(field("tex_a", &mut config));
        let tex_b = parse_arr(field("tex_b", &mut config));
        let tex_c = parse_arr(field("tex_c", &mut config));

        let ab = Vec3f::new(b.x, b.y, b.z) - Vec3f::new(a.x, a.y, a.z);
        let ac = Vec3f::new(c.x, c.y, c.z) - Vec3f::new(a.x, a.y, a.z);
        let n = Vec3f::cross(&ab, &ac).normalize();
        Self {
            a: Vertex {
                pos: a,
                pos_world: Point::origin(),
                n,
                color: color_a,
                tex: tex_a,
            },
            b: Vertex {
                pos: b,
                pos_world: Point::origin(),
                color: color_b,
                n,
                tex: tex_b,
            },
            c: Vertex {
                pos: c,
                pos_world: Point::origin(),
                color: color_c,
                n,
                tex: tex_c,
            },
            tex: None,
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

pub fn field<'a>(name: &str, args: &mut &'a str) -> &'a str {
    *args = args.strip_prefix(&format!("{}=", name)).expect(&format!("missing {}", name));
    let newline = args.find('\n').expect("missing newline");
    let f = &args[..newline];
    *args = &args[newline + 1..];
    f
}

fn load_obj<'a>(obj: &str, color_freq: f64, shade_mode: i32, textures: &'a HashMap<String, RgbaImage>) -> Vec<Triangle<'a>> {
    // vertex data
    let mut vertices = Vec::new();
    // indices into vertex data
    let mut v_indices: Vec<[u32; 3]> = Vec::new();
    // normal data
    let mut normals = Vec::new();
    // indices into normal data
    let mut vn_indices: Vec<[u32; 3]> = Vec::new();
    // texture coordinates
    let mut tex_coords = Vec::new();
    // indices into texture coordinates and corresponding texture
    let mut vt_indices: Vec<([u32; 3], Option<&'a RgbaImage>)> = Vec::new();
    let mut current_tex = None;

    for line in obj.lines() {
        if line.starts_with("v ") {
            let v = line.strip_prefix("v ").unwrap();
            let mut elems = v.split(' ');
            let x = elems.next().unwrap().trim().parse().unwrap();
            let y = elems.next().unwrap().trim().parse().unwrap();
            let z = elems.next().unwrap().trim().parse().unwrap();
            vertices.push(Point::new(x, y, z));
        } else if line.starts_with("vt ") {
            let v = line.strip_prefix("vt ").unwrap();
            let mut elems = v.split(' ');
            let x = elems.next().unwrap().trim().parse().unwrap();
            let y = elems.next().unwrap().trim().parse().unwrap();
            tex_coords.push([x, y]);
        } else if line.starts_with("vn ") {
            let v = line.strip_prefix("vn ").unwrap();
            let mut elems = v.split(' ');
            let x = elems.next().unwrap().trim().parse().unwrap();
            let y = elems.next().unwrap().trim().parse().unwrap();
            let z = elems.next().unwrap().trim().parse().unwrap();
            normals.push(Vec3f::new(x, y, z));
        } else if line.starts_with("f ") {
            let f = line.strip_prefix("f ").unwrap();
            let mut elems = f.split(' ');
            let idx = [elems.next().unwrap(), elems.next().unwrap(), elems.next().unwrap()];

            let mut v = [0; 3];
            let mut vt = ([0; 3], false);
            let mut vn = ([0; 3], false);

            for (i, &s) in idx.iter().enumerate() {
                let mut s = s.split('/');

                v[i] = s.next().unwrap().parse::<u32>().unwrap() - 1;

                if let Some(s) = s.next() {
                    vt.0[i] = s.parse::<u32>().unwrap() - 1;
                    vt.1 = true;
                }

                if let Some(s) = s.next() {
                    vn.0[i] = s.parse::<u32>().unwrap() - 1;
                    vn.1 = true;
                }
            }

            v_indices.push(v);

            if vt.1 {
                vt_indices.push((vt.0, current_tex));
            }

            if vn.1 {
                vn_indices.push(vn.0);
            }
        } else if line.starts_with("usemtl ") {
            let k = line.strip_prefix("usemtl ").unwrap().trim();
            current_tex = textures.get(k);
        }
    }

    if !vt_indices.is_empty() {
        assert_eq!(v_indices.len(), vt_indices.len());
    }

    if !vn_indices.is_empty() {
        assert_eq!(v_indices.len(), vn_indices.len());
    }

    let mut triangles = Vec::new();

    for tri in v_indices.iter() {
        let a = vertices[tri[0] as usize];
        let b = vertices[tri[1] as usize];
        let c = vertices[tri[2] as usize];

        let color_a;
        let color_b;
        let color_c;

        match shade_mode {
            0 => {
                let ra = (((a.x * color_freq).sin() + 1.0) * 128.0) as u8;
                let ga = (((a.y * color_freq).sin() + 1.0) * 128.0) as u8;
                let ba = (((a.z * color_freq).sin() + 1.0) * 128.0) as u8;
                color_a = [ra, ga, ba];
                color_b = [ra, ga, ba];
                color_c = [ra, ga, ba];
            }
            1 => {
                let ra = (((a.x * color_freq).sin() + 1.0) * 128.0) as u8;
                let ga = (((a.y * color_freq).sin() + 1.0) * 128.0) as u8;
                let ba = (((a.z * color_freq).sin() + 1.0) * 128.0) as u8;
                let rb = (((b.x * color_freq).sin() + 1.0) * 128.0) as u8;
                let gb = (((b.y * color_freq).sin() + 1.0) * 128.0) as u8;
                let bb = (((b.z * color_freq).sin() + 1.0) * 128.0) as u8;
                let rc = (((c.x * color_freq).sin() + 1.0) * 128.0) as u8;
                let gc = (((c.y * color_freq).sin() + 1.0) * 128.0) as u8;
                let bc = (((c.z * color_freq).sin() + 1.0) * 128.0) as u8;
                color_a = [ra, ga, ba];
                color_b = [rb, gb, bb];
                color_c = [rc, gc, bc];
            }
            2 => {
                let ra = (((a.x * color_freq).sin() + 1.0) * 128.0) as u8;
                let ga = (((a.y * color_freq).cos() + 1.0) * 128.0) as u8;
                let ba = (((a.z * color_freq).sin() + 1.0) * 128.0) as u8;
                let rb = (((b.x * color_freq).cos() + 1.0) * 128.0) as u8;
                let gb = (((b.y * color_freq).sin() + 1.0) * 128.0) as u8;
                let bb = (((b.z * color_freq).cos() + 1.0) * 128.0) as u8;
                let rc = (((c.x * color_freq).sin() + 1.0) * 128.0) as u8;
                let gc = (((c.y * color_freq).cos() + 1.0) * 128.0) as u8;
                let bc = (((c.z * color_freq).sin() + 1.0) * 128.0) as u8;
                color_a = [ra, ga, ba];
                color_b = [rb, gb, bb];
                color_c = [rc, gc, bc];
            }
            _ => panic!("invalid shading mode")
        }

        let ab = Vec3f::new(b.x, b.y, b.z) - Vec3f::new(a.x, a.y, a.z);
        let ac = Vec3f::new(c.x, c.y, c.z) - Vec3f::new(a.x, a.y, a.z);
        let n = Vec3f::cross(&ab, &ac).normalize();
        triangles.push(Triangle {
            a: Vertex {
                pos: a,
                pos_world: Point::origin(),
                color: color_a,
                n,
                tex: [0.0, 0.0]
            },
            b: Vertex {
                pos: b,
                pos_world: Point::origin(),
                color: color_b,
                n,
                tex: [0.0, 0.0]
            },
            c: Vertex {
                pos: c,
                pos_world: Point::origin(),
                color: color_c,
                n,
                tex: [0.0, 0.0]
            },
            tex: None,
        });
    }

    for (i, tri) in vt_indices.iter().enumerate() {
        let tex_a = tex_coords[tri.0[0] as usize];
        let tex_b = tex_coords[tri.0[1] as usize];
        let tex_c = tex_coords[tri.0[2] as usize];

        triangles[i].a.tex = tex_a;
        triangles[i].b.tex = tex_b;
        triangles[i].c.tex = tex_c;
        triangles[i].tex = tri.1;
    }

    for (i, tri) in vn_indices.iter().enumerate() {
        let n_a = normals[tri[0] as usize];
        let n_b = normals[tri[1] as usize];
        let n_c = normals[tri[2] as usize];

        triangles[i].a.n = n_a;
        triangles[i].b.n = n_b;
        triangles[i].c.n = n_c;
    }

    triangles
}