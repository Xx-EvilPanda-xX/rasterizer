use std::{str::FromStr, collections::HashMap};

use core::fmt::Debug;
use image::RgbaImage;

use crate::MtlData;
use crate::renderer::{Triangle, Vertex, Clip};
use crate::math::{Point3d, Vec3f};

pub struct Config<'a> {
    pub width: u32,
    pub height: u32,
    pub clear_color: [u8; 3],
    pub legacy: bool,
    pub fullscreen: bool,
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
    pub light_pos: Point3d,
    pub light_scale: f64,
    pub ambient: f64,
    pub diffuse: f64,
    pub specular: f64,
    pub light_diss: f64,
    pub shininess: u32,
    pub render_shadows: bool,
    pub tex_sample_lerp: bool,
    pub render_threads: u32,
    pub show_progress: bool,
    pub do_rotation: bool,
    pub wireframe: bool,
    pub triangles: Vec<Triangle<'a>>,
}

impl<'a> Config<'a> {

    // text: <name>.txt, obj: <name>.obj
    pub fn new(text: &str, obj: &Option<String>, textures: &'a HashMap<String, MtlData>) -> Self {
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
        let mut img_config = sections.next().expect("Render configurations are not present");

        let width = eval(field("width", &mut img_config), false, "");
        let height = eval(field("height", &mut img_config), false, "");
        let clear_color = eval(field("clear_color", &mut img_config), false, "");
        let legacy = eval(field("legacy", &mut img_config), false, "");

        let legacy = legacy.parse().expect("Failed to parse legacy");

        let fullscreen = eval(field("fullscreen", &mut img_config), legacy, "false");
        let color_freq = eval(field("color_freq", &mut img_config), legacy, "0.0");
        let shade_mode = eval(field("shade_mode", &mut img_config), legacy, "0");
        let fov = eval(field("fov", &mut img_config), legacy, "0.0");
        let n = eval(field("n", &mut img_config), legacy, "0.0");
        let f = eval(field("f", &mut img_config), legacy, "0.0");
        let scale = eval(field("scale", &mut img_config), legacy, "0.0");
        let rot_x = eval(field("rot_x", &mut img_config), legacy, "0.0");
        let rot_y = eval(field("rot_y", &mut img_config), legacy, "0.0");
        let rot_z = eval(field("rot_z", &mut img_config), legacy, "0.0");
        let trans_x = eval(field("trans_x", &mut img_config), legacy, "0.0");
        let trans_y = eval(field("trans_y", &mut img_config), legacy, "0.0");
        let trans_z = eval(field("trans_z", &mut img_config), legacy, "0.0");
        let light_pos = eval(field("light_pos", &mut img_config), legacy, "[0.0,0.0,0.0]");
        let light_scale = eval(field("light_scale", &mut img_config), legacy, "0.0");
        let ambient = eval(field("ambient", &mut img_config), legacy, "1.0");
        let diffuse = eval(field("diffuse", &mut img_config), legacy, "0.0");
        let specular = eval(field("specular", &mut img_config), legacy, "0.0");
        let light_diss = eval(field("light_diss", &mut img_config), legacy, "0.0");
        let shininess = eval(field("shininess", &mut img_config), legacy, "0");
        let render_shadows = eval(field("render_shadows", &mut img_config), legacy, "false");
        let tex_sample_lerp = eval(field("tex_sample_lerp", &mut img_config), legacy, "false");
        let render_threads = eval(field("render_threads", &mut img_config), legacy, "16");
        let show_progess = eval(field("show_progress", &mut img_config), legacy, "false");
        let do_rotation = eval(field("do_rotation", &mut img_config), legacy, "false");
        let wireframe = eval(field("wireframe", &mut img_config), legacy, "false");

        let triangles = if let Some(text) = obj {
            load_obj(
                &text,
                textures,
                color_freq.parse().expect("Failed to parse color_freq"),
                shade_mode.parse().expect("Failed to parse shade_mode")
            )
        } else {
            sections.into_iter().map(|section| Triangle::from_config(section)).collect()
        };

        Self {
            width: width.parse().expect("Failed to parse width"),
            height: height.parse().expect("Failed to parse height"),
            clear_color: parse_arr(clear_color),
            legacy,
            fullscreen: fullscreen.parse().expect("Failed to parse fullscreen"),
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
            light_pos: Point3d::from_arr(parse_arr(light_pos)),
            light_scale: light_scale.parse().expect("Failed to parse light_scale"),
            ambient: ambient.parse().expect("Failed to parse ambient"),
            diffuse: diffuse.parse().expect("Failed to parse diffuse"),
            specular: specular.parse().expect("Failed to parse specular"),
            light_diss: light_diss.parse().expect("Failed to parse light_diss"),
            shininess: shininess.parse().expect("Failed to parse shininess"),
            render_shadows: render_shadows.parse().expect("Failed to parse render_shadows"),
            tex_sample_lerp: tex_sample_lerp.parse().expect("Failed to parse tex_sample_lerp"),
            render_threads: render_threads.parse().expect("Failed to parse render_threads"),
            show_progress: show_progess.parse().expect("Failed to parse show_progress"),
            do_rotation: do_rotation.parse().expect("Failed to parse do_rotation"),
            wireframe: wireframe.parse().expect("Failed to parse wireframe"),
            triangles,
        }
    }
}

impl<'a> Triangle<'a> {
    pub fn from_config(mut config: &str) -> Self {
        let a = Point3d::from_arr(parse_arr(eval(field("a", &mut config), false, "")));
        let b = Point3d::from_arr(parse_arr(eval(field("b", &mut config), false, "")));
        let c = Point3d::from_arr(parse_arr(eval(field("c", &mut config), false, "")));
        let color_a = parse_arr(eval(field("color_a", &mut config), false, ""));
        let color_b = parse_arr(eval(field("color_b", &mut config), false, ""));
        let color_c = parse_arr(eval(field("color_c", &mut config), false, ""));

        let ab = Vec3f::new(b.x, b.y, b.z) - Vec3f::new(a.x, a.y, a.z);
        let ac = Vec3f::new(c.x, c.y, c.z) - Vec3f::new(a.x, a.y, a.z);
        let n = Vec3f::cross(&ab, &ac).normalize();
        Self {
            a: Vertex {
                pos: a,
                pos_world: Point3d::origin(),
                pos_clip: Point3d::origin(),
                color: color_a,
                n,
                tex: [0.0, 0.0],
            },
            b: Vertex {
                pos: b,
                pos_world: Point3d::origin(),
                pos_clip: Point3d::origin(),
                color: color_b,
                n,
                tex: [0.0, 0.0],
            },
            c: Vertex {
                pos: c,
                pos_world: Point3d::origin(),
                pos_clip: Point3d::origin(),
                color: color_c,
                n,
                tex: [0.0, 0.0],
            },
            tex: None,
            clip: Clip::Zero,
        }
    }
}

fn eval<'a>(field: Result<&'a str, FieldError>, legacy: bool, default: &'a str) -> &'a str {
    match field {
        Ok(field) => field,
        Err(e) => {
            if legacy {
                default
            } else {
                match e {
                    FieldError::Missing(missed) => {
                        println!("Missing field `{}`", missed);
                        print_field_help();
                        std::process::exit(-1);
                    }
                    FieldError::MissingNewline => {
                        println!("Missing newline after field");
                        std::process::exit(-1);
                    }
                }
            }
        }
    }
}

fn print_field_help() {
    println!("\nFields should be laid out like so for the render config:");
    println!("width = u32
height = u32
clear_color = [u8, u8, u8]
legacy = bool
fullscreen = bool
color_freq = f64
shade_mode = u32
fov = f64
n = f64
f = f64
scale = f64
rot_x = f64
rot_y = f64
rot_z = f64
trans_x = f64
trans_y = f64
trans_z = f64
light_pos = [f64, f64, f64]
light_scale = f64
ambient = f64
diffuse = f64
specular = f64
light_diss = f64
shininess = u32
render_shadows = bool
tex_sample_lerp = bool
render_threads = u32
show_progress = bool
do_rotation = bool
wireframe = bool");
    println!("\nAnd like so for triangle configs:");
    println!("a = [f64, f64, f64]
b = [f64, f64, f64]
c = [f64, f64, f64]
color_a = [u8, u8, u8]
color_a = [u8, u8, u8]
color_a = [u8, u8, u8]");
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
        out[i] = e.parse().expect("failed to parse array element");
    }

    if elems.next().is_some() {
        panic!("too many array elements");
    }

    out
}

enum FieldError {
    Missing(String),
    MissingNewline,
}

fn field<'a>(name: &str, args: &mut &'a str) -> Result<&'a str, FieldError> {
    *args = if let Some(new_args) = args.strip_prefix(&format!("{}=", name)) {
        new_args
    } else {
        return Err(FieldError::Missing(name.to_string()));
    };

    let newline = if let Some(newline) = args.find('\n') {
        newline
    } else {
        return Err(FieldError::MissingNewline);
    };

    let f = &args[..newline];
    *args = &args[newline + 1..];
    Ok(f)
}

fn load_obj<'a>(obj: &str, mtls: &'a HashMap<String, MtlData>, color_freq: f64, shade_mode: i32) -> Vec<Triangle<'a>> {
    // vertex data
    let mut vertices = Vec::new();
    // indices into vertex data
    let mut v_indices: Vec<([u32; 3], Option<[u8; 3]>)> = Vec::new();
    // normal data
    let mut normals = Vec::new();
    // indices into normal data and the index of the triangle it belongs to
    let mut vn_indices: Vec<([u32; 3], usize)> = Vec::new();
    // texture coordinates
    let mut tex_coords = Vec::new();
    // indices into texture coordinates and corresponding texture and the index of the triangle it belongs to
    let mut vt_indices: Vec<([u32; 3], Option<&'a RgbaImage>, usize)> = Vec::new();

    let mut current_tex = None;
    let mut current_mtl_color = None;

    for line in obj.lines() {
        if line.starts_with("v ") {
            let v = line.strip_prefix("v ").unwrap();
            let mut elems = v.split(' ');
            let x = elems.next().unwrap().trim().parse().unwrap();
            let y = elems.next().unwrap().trim().parse().unwrap();
            let z = elems.next().unwrap().trim().parse().unwrap();
            vertices.push(Point3d::new(x, y, z));
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

            for (i, &string) in idx.iter().enumerate() {
                let mut s = string.split('/');

                v[i] = s.next().unwrap().parse::<u32>().expect("Failed to parse vertex position index") - 1;

                if let Some(s) = s.next() {
                    if let Ok(vt_index) = s.parse::<u32>() {
                        vt.0[i] = vt_index - 1;
                        vt.1 = true;
                    }
                }

                if let Some(s) = s.next() {
                    if let Ok(vn_index) = s.parse::<u32>() {
                        vn.0[i] = vn_index - 1;
                        vn.1 = true;
                    }
                }
            }

            // every trianlge MUST have a position attrib
            v_indices.push((v, current_mtl_color));

            // but not a tex or norm attrib
            if vt.1 {
                vt_indices.push((vt.0, current_tex, v_indices.len() - 1)); // v_indices.len() - 1 is essentially the triangle index since we make a tri for every element
            }

            if vn.1 {
                vn_indices.push((vn.0, v_indices.len() - 1));
            }
        } else if line.starts_with("usemtl ") {
            let k = line.strip_prefix("usemtl ").unwrap().trim();
            if let Some(mtl) = mtls.get(k) {
                current_tex = mtl.tex.as_ref();
                current_mtl_color = Some(mtl.color);
            } else {
                current_tex = None;
                current_mtl_color = None;
            }
        }
    }

    let mut triangles = Vec::new();

    for (tri, color) in v_indices.iter() {
        let a = vertices[tri[0] as usize];
        let b = vertices[tri[1] as usize];
        let c = vertices[tri[2] as usize];

        let (color_a, color_b, color_c) = get_color(shade_mode, &a, &b, &c, color_freq);

        let ab = Vec3f::new(b.x, b.y, b.z) - Vec3f::new(a.x, a.y, a.z);
        let ac = Vec3f::new(c.x, c.y, c.z) - Vec3f::new(a.x, a.y, a.z);
        let n = Vec3f::cross(&ab, &ac).normalize();
        triangles.push(Triangle {
            a: Vertex {
                pos: a,
                // these values are intialized in the vertex shader
                pos_world: Point3d::origin(),
                pos_clip: Point3d::origin(),

                color: color.unwrap_or(color_a),
                // both of these may or not be initialized with a different value in the below loops
                n,
                tex: [0.0, 0.0]
            },
            b: Vertex {
                pos: b,
                pos_world: Point3d::origin(),
                pos_clip: Point3d::origin(),
                color: color.unwrap_or(color_b),
                n,
                tex: [0.0, 0.0]
            },
            c: Vertex {
                pos: c,
                pos_world: Point3d::origin(),
                pos_clip: Point3d::origin(),
                color: color.unwrap_or(color_c),
                n,
                tex: [0.0, 0.0],
            },
            tex: None,
            // these values are intitalized in the vertex shader
            clip: Clip::Zero,
        });
    }

    for &(tri, tex, i) in vt_indices.iter() {
        triangles[i].a.tex = tex_coords[tri[0] as usize];
        triangles[i].b.tex = tex_coords[tri[1] as usize];
        triangles[i].c.tex = tex_coords[tri[2] as usize];
        triangles[i].tex = tex;
    }

    for &(tri, i) in vn_indices.iter() {
        triangles[i].a.n = normals[tri[0] as usize];
        triangles[i].b.n = normals[tri[1] as usize];
        triangles[i].c.n = normals[tri[2] as usize];
    }

    triangles
}

fn get_color(shade_mode: i32, a: &Point3d, b: &Point3d, c: &Point3d, color_freq: f64) -> ([u8; 3], [u8; 3], [u8; 3]) {
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

    (color_a, color_b, color_c)
}