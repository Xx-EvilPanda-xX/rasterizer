use std::mem::swap;
use crate::{math::*, SubBuffer, COLOR_BUF_CHANNELS};
use image::RgbaImage;

#[derive(Clone, Debug)]
pub struct Triangle<'a> {
    pub a: Vertex,
    pub b: Vertex,
    pub c: Vertex,
    pub tex: Option<&'a RgbaImage>,
    pub clip: Clip<'a>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Vertex {
    pub pos: Point3d, // x and y are in screen space for rasterization, z is still in clip space
    pub pos_world: Point3d,
    pub pos_clip: Point3d,
    pub color: [u8; 3],
    pub n: Vec3f,
    pub tex: [f64; 2],
}

#[derive(Clone, Debug)]
pub enum Clip<'a> {
    Zero,
    // when only one vertex triangle is clipped, two triangles need to be created. One will be stored in this box and one in the containing triangle.
    One(Box<Triangle<'a>>),
    Two,
    Three,
}

#[derive(Clone, Copy, Debug)]
enum ClipLerp {
    AB,
    AC,
    BC
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
    pos_x: Plane,
    pos_y: Plane,
    pos_z: Plane,
}

#[derive(Default)]
pub struct Uniforms {
    pub model: Mat4f,
    pub view: Mat4f,
    pub proj: Mat4f,
    pub shadow_view: Mat4f,
    pub shadow_proj: Mat4f,
    pub inv_view: Mat4f,
    pub inv_proj: Mat4f,
    pub shadow_inv_view: Mat4f,
    pub shadow_inv_proj: Mat4f,
    pub near_clipping_plane: f64,
    pub light_pos: Point3d,
    pub cam_pos: Point3d,
    pub ambient: f64,
    pub diffuse: f64,
    pub specular: f64,
    pub light_dissipation: f64,
    pub light_intensity: f64,
    pub shininess: u32,
    pub legacy: bool,
    pub shadow_mode: u32,
    pub shadow_buf_width: u32,
    pub shadow_buf_height: u32,
    pub shadow_perspective: Option<Perspective>,
    pub tex_sample_lerp: bool,
    pub wireframe: bool,
}

// above the frame
fn in_section_1(p: &Point2d, start_y: u32) -> bool {
    p.y < start_y as f64
}

// to the right of the frame
fn in_section_2(p: &Point2d, dims: (u32, u32)) -> bool {
    p.x > dims.0 as f64
}

// below the frame
fn in_section_3(p: &Point2d, dims: (u32, u32), start_y: u32) -> bool {
    p.y > (start_y + dims.1) as f64
}

// to the left of the frame
fn in_section_4(p: &Point2d) -> bool {
    p.x < 0.0
}

pub fn rasterize(buf: &mut SubBuffer, tri: &Triangle, occ: &[[Point3d; 3]], shadow_depth: Option<&[f64]>, view_inv: &Mat4f, proj_inv: &Mat4f, u: &Uniforms) -> u32 {
    let dims = buf.dims;
    let (a, b, c) = (&tri.a.pos.into_2d(), &tri.b.pos.into_2d(), &tri.c.pos.into_2d());

    // simple check before hand to see if the triangle will ever be visible
    if (in_section_1(a, buf.start_y) && in_section_1(b, buf.start_y) && in_section_1(c, buf.start_y)) ||
        (in_section_2(a, dims) && in_section_2(b, dims) && in_section_2(c, dims)) ||
        (in_section_3(a, dims, buf.start_y) && in_section_3(b, dims, buf.start_y) && in_section_3(c, dims, buf.start_y)) ||
        (in_section_4(a) && in_section_4(b) && in_section_4(c)) {
        return 0;
    }

    let (a_world, b_world, c_world) = (&tri.a.pos_world, &tri.b.pos_world, &tri.c.pos_world);
    let planes = AttributePlanes {
        color_r: plane_from_points(&Point3d::new(a.x, a.y, tri.a.color[0] as f64), &Point3d::new(b.x, b.y, tri.b.color[0] as f64), &Point3d::new(c.x, c.y, tri.c.color[0] as f64), PlaneSolveType::Z),
        color_g: plane_from_points(&Point3d::new(a.x, a.y, tri.a.color[1] as f64), &Point3d::new(b.x, b.y, tri.b.color[1] as f64), &Point3d::new(c.x, c.y, tri.c.color[1] as f64), PlaneSolveType::Z),
        color_b: plane_from_points(&Point3d::new(a.x, a.y, tri.a.color[2] as f64), &Point3d::new(b.x, b.y, tri.b.color[2] as f64), &Point3d::new(c.x, c.y, tri.c.color[2] as f64), PlaneSolveType::Z),
        n_x: attrib_plane(a_world, b_world, c_world, tri.a.n.x, tri.b.n.x, tri.c.n.x),
        n_y: attrib_plane(a_world, b_world, c_world, tri.a.n.y, tri.b.n.y, tri.c.n.y),
        n_z: attrib_plane(a_world, b_world, c_world, tri.a.n.z, tri.b.n.z, tri.c.n.z),
        tex_x: attrib_plane(a_world, b_world, c_world, tri.a.tex[0], tri.b.tex[0], tri.c.tex[0]),
        tex_y: attrib_plane(a_world, b_world, c_world, tri.a.tex[1], tri.b.tex[1], tri.c.tex[1]),
        pos_x: plane_from_points(&Point3d::new(a.x, a.y, tri.a.pos_clip.x), &Point3d::new(b.x, b.y, tri.b.pos_clip.x), &Point3d::new(c.x, c.y, tri.c.pos_clip.x), PlaneSolveType::Z),
        pos_y: plane_from_points(&Point3d::new(a.x, a.y, tri.a.pos_clip.y), &Point3d::new(b.x, b.y, tri.b.pos_clip.y), &Point3d::new(c.x, c.y, tri.c.pos_clip.y), PlaneSolveType::Z),
        pos_z: plane_from_points(&Point3d::new(a.x, a.y, tri.a.pos_clip.z), &Point3d::new(b.x, b.y, tri.b.pos_clip.z), &Point3d::new(c.x, c.y, tri.c.pos_clip.z), PlaneSolveType::Z),
    };

    let mut pixels_shaded = 0;
    let color_buf_exists = buf.color.is_some();

    let mut pixel = |x, y| {
        if x >= dims.0 || y < buf.start_y || y >= buf.start_y + dims.1 {
            return;
        }

        // localize the index into the sub buffer to the current chunk
        let i = ((y - buf.start_y) * dims.0 + x) as usize;

        // we can safely pass in 0 for the z here because we will never being solving for anything other than z with this plane
        let depth = lerp_fast(&planes.pos_z, x as f64, y as f64, 0.0);
        if depth < buf.depth[i] && depth >= super::SCREEN_Z {
            if let Some(color_buf) = &mut buf.color {
                let color = pixel_shader(tri, occ, shadow_depth.unwrap(), view_inv, proj_inv, u, &planes, x, y);

                // discard transparency
                if color[3] == 0 {
                    return;
                }

                color_buf[i * COLOR_BUF_CHANNELS] = color[0];
                color_buf[i * COLOR_BUF_CHANNELS + 1] = color[1];
                color_buf[i * COLOR_BUF_CHANNELS + 2] = color[2];
            }

            buf.depth[i] = depth;
            pixels_shaded += 1;
        }
    };

    let mut draw_line = |p1: &Point2d, p2: &Point2d| {
        let (mut start_x, mut start_y) = (p1.x.ceil() as i32, p1.y.ceil() as i32);
        let (mut end_x, mut end_y) = (p2.x.ceil() as i32, p2.y.ceil() as i32);

        // if the line is vertical dont bother scanning x
        if start_x == end_x {
            for y in start_y..end_y {
                // rule out any negative pixels because casting to u32 causes problems
                if start_x < 0 || y < 0 {
                    continue;
                }

                pixel(start_x as u32, y as u32);
            }

            return;
        }

        if start_x > end_x {
            swap(&mut start_x, &mut end_x);
            swap(&mut start_y, &mut end_y);
        }

        let line = line_2d_from_points(&Point2d::new(start_x as f64, start_y as f64), &Point2d::new(end_x as f64, end_y as f64));

        let mut current_y = start_y;

        for current_x in start_x..=end_x {
            let target_y = solve_y(&line, current_x as f64);
            let mut diff = current_y as f64 - target_y;

            // rule out any negative pixels because casting to u32 causes problems
            if current_x >= 0 && current_y >= 0 && diff.abs() < 0.5 {
                pixel(current_x as u32, current_y as u32);
            }

            while diff.abs() > 0.5 {
                if diff.is_sign_positive() {
                    current_y -= 1;
                }

                if diff.is_sign_negative() {
                    current_y += 1;
                }

                // rule out any negative pixels because casting to u32 causes problems
                if current_x >= 0 && current_y >= 0 {
                    pixel(current_x as u32, current_y as u32);
                }

                diff = current_y as f64 - target_y;
            }
        }
    };

    // check for existent color buf since we dont want to ever do a wireframe shadow depth render
    if u.wireframe && color_buf_exists {
        draw_line(a, b);
        draw_line(b, c);
        draw_line(a, c);
    } else {
        for i in 0..2 {
            // we use .ceil() as our method of rounding here because it allows for the correct (pixel-perfect) start and stop values
            let (mut start_y, mut end_y) = match i {
                0 => (a.y.ceil() as u32, b.y.ceil() as u32),
                1 => (b.y.ceil() as u32, c.y.ceil() as u32),
                _ => unreachable!(),
            };

            start_y = start_y.clamp(buf.start_y, buf.start_y + dims.1);
            end_y = end_y.clamp(buf.start_y, buf.start_y + dims.1);
            for y in start_y..end_y {
                let (start_x, end_x) = match i {
                    0 => top_scanline(tri, y),
                    1 => bottom_scanline(tri, y),
                    _ => unreachable!(),
                };

                // YOU CANNOT ROUND HERE (.round()). It creates situtations where the start and end x are outside our tri
                let mut start_x = start_x.ceil() as u32;
                let mut end_x = end_x.ceil() as u32;
                if start_x > end_x {
                    swap(&mut start_x, &mut end_x);
                }

                start_x = start_x.clamp(0, dims.0);
                end_x = end_x.clamp(0, dims.0);

                for x in start_x..end_x {
                    pixel(x, y);
                };
            }
        }
    }

    fn top_scanline(tri: &Triangle, y: u32) -> (f64, f64) {
        let (a, b, c) = (&tri.a.pos.into_2d(), &tri.b.pos.into_2d(), &tri.c.pos.into_2d());
        // if tri has a flat top side, we don't have a top half of the tri
        if a.y == b.y {
            return (0.0, 0.0);
        }

        let ab = a.x == b.x;
        let ac = a.x == c.x;
        if !ab && !ac {
            let ab = line_2d_from_points(a, b);
            let ac = line_2d_from_points(a, c);
            let start = solve_x(&ab, y as f64);
            let end = solve_x(&ac, y as f64);
            (start, end)
        } else if !ab && ac {
            let ab = line_2d_from_points(a, b);
            let start = solve_x(&ab, y as f64);
            let end = c.x;
            (start, end)
        } else if ab && !ac {
            let ac = line_2d_from_points(a, c);
            let start = b.x;
            let end = solve_x(&ac, y as f64);
            (start, end)
        } else {
            // this doesn't matter since this part of the tri will be invisible
            (0.0, 0.0)
        }
    }

    fn bottom_scanline(tri: &Triangle, y: u32) -> (f64, f64) {
        let (a, b, c) = (&tri.a.pos.into_2d(), &tri.b.pos.into_2d(), &tri.c.pos.into_2d());
        // if tri has a flat bottom side, we don't have a bottom half of the tri
        if b.y == c.y {
            return (0.0, 0.0);
        }

        let cb = c.x == b.x;
        let ca = c.x == a.x;
        if !cb && !ca {
            let cb = line_2d_from_points(c, b);
            let ca = line_2d_from_points(c, a);
            let start = solve_x(&cb, y as f64);
            let end = solve_x(&ca, y as f64);
            (start, end)
        } else if !cb && ca {
            let cb = line_2d_from_points(c, b);
            let start = solve_x(&cb, y as f64);
            let end = a.x;
            (start, end)
        } else if cb && !ca {
            let ca = line_2d_from_points(c, a);
            let start = b.x;
            let end = solve_x(&ca, y as f64);
            (start, end)
        } else {
            // this doesn't matter since this part of the tri will be invisible
            (0.0, 0.0)
        }
    }

    pixels_shaded
}

// takes triangle in model space and returns it in raster space, also returns pure model transform
pub fn vertex_shader<'a>(
    tri: &Triangle<'a>,
    model: &Mat4f,
    view: &Mat4f,
    proj: &Mat4f,
    inv_view: &Mat4f,
    u: &Uniforms,
    dims: (u32, u32)
) -> (Triangle<'a>, [Point3d; 3]) {
    let (a, b, c) = (&tri.a, &tri.b, &tri.c);

    // model transform
    let mut pos_a = mul_point_matrix(&a.pos, model);
    let mut pos_b = mul_point_matrix(&b.pos, model);
    let mut pos_c = mul_point_matrix(&c.pos, model);

    // save model space position
    let pos_world_a = pos_a;
    let pos_world_b = pos_b;
    let pos_world_c = pos_c;

    // no non-uniform scaling is actually done to our points, so its fine to multiply normals by the model too.
    let n_a = mul_point_matrix(&a.n.into_point(), &model.no_trans()).into_vec().normalize();
    let n_b = mul_point_matrix(&b.n.into_point(), &model.no_trans()).into_vec().normalize();
    let n_c = mul_point_matrix(&c.n.into_point(), &model.no_trans()).into_vec().normalize();

    // view transform
    pos_a = mul_point_matrix(&pos_a, view);
    pos_b = mul_point_matrix(&pos_b, view);
    pos_c = mul_point_matrix(&pos_c, view);

    // here we create vertices just for clipping (we might need to interpolate between attributes)
    // anything in clip space is left unused, but pos is used to store the view space position (clipping is done in view space)
    let v_a = Vertex {
        pos: pos_a,
        pos_world: pos_world_a,
        color: a.color,
        n: n_a,
        tex: a.tex,
        ..Default::default()
    };
    let v_b = Vertex {
        pos: pos_b,
        pos_world: pos_world_b,
        color: b.color,
        n: n_b,
        tex: b.tex,
        ..Default::default()
    };
    let v_c = Vertex {
        pos: pos_c,
        pos_world: pos_world_c,
        color: c.color,
        n: n_c,
        tex: c.tex,
        ..Default::default()
    };

    let clip_dist = u.near_clipping_plane;
    let (num_clipped, clipped_points, non_clipped_points) = get_clipped(&v_a, &v_b, &v_c, -clip_dist);
    // if Some, holds one of the new tris from clipping, to replace the original triangle
    let mut tri_override = None;

    let mut clip = match num_clipped {
        0 => {
            Clip::Zero
        }
        1 => {
            let a = &non_clipped_points[0];
            let b = &non_clipped_points[1];
            // gaurantee c to the only clipped vertex
            let c = &clipped_points[0];

            let ray1 = ray_3d_from_points(&a.pos, &c.pos);
            let ray2 = ray_3d_from_points(&b.pos, &c.pos);
            // solve for the points where our triangle is barely not clippable
            let inter1 = solve_ray_3d_z(&ray1, -clip_dist);
            let inter2 = solve_ray_3d_z(&ray2, -clip_dist);

            // find the factor by which to lerp attribs in the newly found points
            let lerp_val_a = (dist_3d(&a.pos, &inter1) / dist_3d(&a.pos, &c.pos)).sqrt();
            let lerp_val_b = (dist_3d(&b.pos, &inter2) / dist_3d(&b.pos, &c.pos)).sqrt();

            // 1st new triangle
            tri_override = Some(make_clipped_triangle(a, b, c, tri.tex, &[a, b], &[(&inter2, lerp_val_b, ClipLerp::BC)], inv_view));
            Clip::One(
                Box::new(
                    // 2nd new triangle
                    make_clipped_triangle(a, b, c, tri.tex, &[a], &[(&inter1, lerp_val_a, ClipLerp::AC), (&inter2, lerp_val_b, ClipLerp::BC)], inv_view)
                )
            )
        }
        2 => {
            // guarantee a to be the only non clipped vertex
            let a = &non_clipped_points[0];
            let b = &clipped_points[0];
            let c = &clipped_points[1];

            let ray1 = ray_3d_from_points(&a.pos, &b.pos);
            let ray2 = ray_3d_from_points(&a.pos, &c.pos);
            let inter1 = solve_ray_3d_z(&ray1, -clip_dist);
            let inter2 = solve_ray_3d_z(&ray2, -clip_dist);

            let lerp_val_b = dist_3d(&a.pos, &inter1).sqrt() / dist_3d(&a.pos, &b.pos).sqrt();
            let lerp_val_c = dist_3d(&a.pos, &inter2).sqrt() / dist_3d(&a.pos, &c.pos).sqrt();

            // only new triangle
            tri_override = Some(make_clipped_triangle(a, b, c, tri.tex, &[a], &[(&inter1, lerp_val_b, ClipLerp::AB), (&inter2, lerp_val_c, ClipLerp::AC)], inv_view));
            Clip::Two
        }
        3 => {
            Clip::Three
        }
        _ => unreachable!()
    };

    // no clipping in legacy mode
    if u.legacy {
        tri_override = None;
        clip = Clip::Zero;
    }

    // in the case of an override, we already have a half contructed triangle and simply need to complete it
    if let Some(mut tri) = tri_override {
        // do the same clip/raster space calculations for the clip override if it exsists (and the optional newly boxed triangle), then return it
        clip_space_calc(tri.a.pos, &mut tri.a.pos, &mut tri.a.pos_clip, proj, dims);
        clip_space_calc(tri.b.pos, &mut tri.b.pos, &mut tri.b.pos_clip, proj, dims);
        clip_space_calc(tri.c.pos, &mut tri.c.pos, &mut tri.c.pos_clip, proj, dims);

        if let Clip::One(tri) = &mut clip {
            clip_space_calc(tri.a.pos, &mut tri.a.pos, &mut tri.a.pos_clip, proj, dims);
            clip_space_calc(tri.b.pos, &mut tri.b.pos, &mut tri.b.pos_clip, proj, dims);
            clip_space_calc(tri.c.pos, &mut tri.c.pos, &mut tri.c.pos_clip, proj, dims);
        }

        tri.clip = clip;
        (tri, [pos_world_a, pos_world_b, pos_world_c])
    } else {
        let mut pos_clip_a = Point3d::default();
        let mut pos_clip_b = Point3d::default();
        let mut pos_clip_c = Point3d::default();

        clip_space_calc(pos_a, &mut pos_a, &mut pos_clip_a, proj, dims);
        clip_space_calc(pos_b, &mut pos_b, &mut pos_clip_b, proj, dims);
        clip_space_calc(pos_c, &mut pos_c, &mut pos_clip_c, proj, dims);

        (Triangle {
            a: Vertex {
                pos: pos_a,
                pos_world: pos_world_a,
                pos_clip: pos_clip_a,
                color: a.color,
                n: n_a,
                tex: a.tex,
            },
            b: Vertex {
                pos: pos_b,
                pos_world: pos_world_b,
                pos_clip: pos_clip_b,
                color: b.color,
                n: n_b,
                tex: b.tex,
            },
            c: Vertex {
                pos: pos_c,
                pos_world: pos_world_c,
                pos_clip: pos_clip_c,
                color: c.color,
                n: n_c,
                tex: c.tex,
            },
            tex: tri.tex,
            clip,
        }, [pos_world_a, pos_world_b, pos_world_c])
    }
}

fn clip_space_calc(pos_view: Point3d, pos: &mut Point3d, pos_clip: &mut Point3d, proj: &Mat4f, dims: (u32, u32)) {
    // projection transform
    *pos = mul_point_matrix(&pos_view, proj);
    // save clip space position
    *pos_clip = *pos;
    // normalize to 0 to 1 and scale to raster space
    pos.x = (pos.x + 1.0) / 2.0 * dims.0 as f64;
    pos.y = (pos.y + 1.0) / 2.0 * dims.1 as f64;
}

// find the vertices in a triangle that need to be clipped or not and retain their respective attrib info
fn get_clipped(a: &Vertex, b: &Vertex, c: &Vertex, clip_dist: f64) -> (usize, [Vertex; 3], [Vertex; 3]) {
    let mut num_clipped = 0;
    let mut num_non_clipped = 0;
    let mut clipped_points = [Vertex::default(); 3];
    let mut non_clipped_points = [Vertex::default(); 3];
    let points = [a, b, c];

    for p in points {
        if p.pos.z >= clip_dist {
            clipped_points[num_clipped] = *p;
            num_clipped += 1;
        } else {
            non_clipped_points[num_non_clipped] = *p;
            num_non_clipped += 1;
        };
    }

    (num_clipped, clipped_points, non_clipped_points)
}

// a, b, c are the orignal points of our triangle
// non_clipped is a slice of all the non clipped vertices we want in our newly constructed triangle
// clipped is slice of all the new points made by intersecting with the xy plane
fn make_clipped_triangle<'a>(
    a: &Vertex,
    b: &Vertex,
    c: &Vertex,
    tex: Option<&'a RgbaImage>,
    non_clipped: &[&Vertex],
    // (the clipped point, lerp value, lerp type)
    clipped: &[(&Point3d, f64, ClipLerp)],
    inv_view: &Mat4f,
) -> Triangle<'a> {
    assert_eq!(non_clipped.len() + clipped.len(), 3);
    assert_ne!(non_clipped.len(), 0);
    assert_ne!(clipped.len(), 0);

    let mut out = [Vertex::default(); 3];
    let mut idx = 0;

    for &v in non_clipped {
        out[idx] = *v;
        idx += 1;
    }

    for &(&pos, lerp_val, lerp_type) in clipped {
        let (v1, v2) = match lerp_type {
            ClipLerp::AB => (a, b),
            ClipLerp::AC => (a, c),
            ClipLerp::BC => (b, c),
        };

        let color = [
            lerp(v1.color[0] as f64, v2.color[0] as f64, lerp_val) as u8,
            lerp(v1.color[1] as f64, v2.color[1] as f64, lerp_val) as u8,
            lerp(v1.color[2] as f64, v2.color[2] as f64, lerp_val) as u8,
        ];

        let n = Vec3f::new(
            lerp(v1.n.x, v2.n.x, lerp_val),
            lerp(v1.n.y, v2.n.y, lerp_val),
            lerp(v1.n.z, v2.n.z, lerp_val),
        );

        let tex = [
            lerp(v1.tex[0], v2.tex[0], lerp_val),
            lerp(v1.tex[1], v2.tex[1], lerp_val)
        ];

        let pos_world = mul_point_matrix(&pos, inv_view);
        out[idx] = Vertex {
            pos,
            pos_world,
            color,
            n,
            tex,
            ..Default::default()
        };

        idx += 1;
    }

    Triangle {
        a: out[0],
        b: out[1],
        c: out[2],
        tex,
        clip: Clip::Zero, // this functions purpose is to contruct a triangle THAT HAS ALREADY been clipped
    }
}

pub const FULLY_OPAQUE: u8 = 255;
const INTERPOLATE_FAST: bool = true;

/*
Interpolate_fast treats our attributes as the z values of our
triangles, then solves for z when x and y are known at any
arbitrary point in that plane.

Interpolate_slow finds the maximum distance away from a vertex
that a point on our triangle can be, as well as the actual distance.
From there, it can produce a "weight" for each vertex, to be
multiplied with our attributes.

An interesting thing to note is that if a 1 or more verts of a tri have perfectly equal x and y values in world space,
interpolation won't actually work. This is because any interpolation plane created with this triangle
in mind will have an undefined slope. Therefore, trying to solve for a z value on this plane from
a know x and y will result in a `NaN` or `inf`. The only way to solve this would be to do interpolation in 4
dimensions and solve for w from a know x, y, and z, which im definetly not gonna do. (FIXED)
*/

fn pixel_shader(tri: &Triangle, occ: &[[Point3d; 3]], shadow_depth: &[f64], inv_view: &Mat4f, inv_proj: &Mat4f, u: &Uniforms, planes: &AttributePlanes, x1: u32, y1: u32) -> [u8; 4] {
    let (x, y) = (x1 as f64, y1 as f64);
    const SPEC_COLOR: [u8; 4] = [255, 255, 255, 255];

    let (norm, pix_world_pos, base_color) = if INTERPOLATE_FAST {
        // position of our pixel in clip space
        let pix_clip_pos = Point3d::new(
            lerp_fast(&planes.pos_x, x, y, 0.0),
            lerp_fast(&planes.pos_y, x, y, 0.0),
            lerp_fast(&planes.pos_z, x, y, 0.0),
        );

        // position of our pixel in view space
        let pix_world_pos = mul_point_matrix(&pix_clip_pos, inv_proj);
        // position of our pixel in world space
        let pix_world_pos = mul_point_matrix(&pix_world_pos, inv_view);
        let (x_world, y_world, z_world) = (pix_world_pos.x, pix_world_pos.y, pix_world_pos.z);

        let norm = Vec3f::new(
            lerp_fast(&planes.n_x, x_world, y_world, z_world),
            lerp_fast(&planes.n_y, x_world, y_world, z_world),
            lerp_fast(&planes.n_z, x_world, y_world, z_world),
        ).normalize(); // !!!!!

        let base_color = if let Some(tex) = tri.tex {
            // we calculate texture coords with respect to our pixel's world space position rather than it's clip or raster space position
            // because otherwise we get perspective-incorrect results. we do the same with some other attributes too.
            let vt_x = lerp_fast(&planes.tex_x, x_world, y_world, z_world);
            let vt_y = lerp_fast(&planes.tex_y, x_world, y_world, z_world);
            tex_sample(tex, vt_x, vt_y, u.tex_sample_lerp)
        } else {
            [lerp_fast(&planes.color_r, x, y, 0.0).round() as u8,
            lerp_fast(&planes.color_g, x, y, 0.0).round() as u8,
            lerp_fast(&planes.color_b, x, y, 0.0).round() as u8,
            FULLY_OPAQUE]
        };

        (norm, pix_world_pos, base_color)
    } else {
        let weights_clip = lerp_slow([&tri.a.pos, &tri.b.pos, &tri.c.pos], x, y);
        let pix_clip_pos = Point3d::new(
            tri.a.pos_clip.x * weights_clip.a + tri.b.pos_clip.x * weights_clip.b + tri.c.pos_clip.x * weights_clip.c,
            tri.a.pos_clip.y * weights_clip.a + tri.b.pos_clip.y * weights_clip.b + tri.c.pos_clip.y * weights_clip.c,
            tri.a.pos_clip.z * weights_clip.a + tri.b.pos_clip.z * weights_clip.b + tri.c.pos_clip.z * weights_clip.c,
        );
        let pix_world_pos = mul_point_matrix(&pix_clip_pos, inv_proj);
        let pix_world_pos = mul_point_matrix(&pix_world_pos, inv_view);
        let weights_world = lerp_slow([&tri.a.pos_world, &tri.b.pos_world, &tri.c.pos_world], pix_world_pos.x, pix_world_pos.y);

        let base_color = if let Some(tex) = tri.tex {
            let vt_x = tri.a.tex[0] * weights_world.a + tri.b.tex[0] * weights_world.b + tri.c.tex[0] * weights_world.c;
            let vt_y = tri.a.tex[1] * weights_world.a + tri.b.tex[1] * weights_world.b + tri.c.tex[1] * weights_world.c;
            tex_sample(tex, vt_x, vt_y, u.tex_sample_lerp)
        } else {
            let a = [tri.a.color[0] as f64, tri.a.color[1] as f64, tri.a.color[2] as f64];
            let b = [tri.b.color[0] as f64, tri.b.color[1] as f64, tri.b.color[2] as f64];
            let c = [tri.c.color[0] as f64, tri.c.color[1] as f64, tri.c.color[2] as f64];
            [(a[0] * weights_clip.a + b[0] * weights_clip.b + c[0] * weights_clip.c).round() as u8,
            (a[1] * weights_clip.a + b[1] * weights_clip.b + c[1] * weights_clip.c).round() as u8,
            (a[2] * weights_clip.a + b[2] * weights_clip.b + c[2] * weights_clip.c).round() as u8,
            FULLY_OPAQUE]
        };

        let norm = Vec3f::new(
            tri.a.n.x * weights_world.a + tri.b.n.x * weights_world.b + tri.c.n.x * weights_world.c,
            tri.a.n.y * weights_world.a + tri.b.n.y * weights_world.b + tri.c.n.y * weights_world.c,
            tri.a.n.z * weights_world.a + tri.b.n.z * weights_world.b + tri.c.n.z * weights_world.c,
        ).normalize();

        (norm, pix_world_pos, base_color)
    };

    let debug = x1 == 10 && y1 == 11300;

    let (ambient, diffuse, specular) = calc_lighting(&norm, &pix_world_pos, u, occ, shadow_depth, debug);
    let ambient = mul_color(&base_color, ambient);
    let diffuse = mul_color(&base_color, diffuse);
    let specular = mul_color(&SPEC_COLOR, specular);
    let mut color = add_color(&ambient, &diffuse);
    color = add_color(&color, &specular);
    color[3] = base_color[3];
    color
}

fn calc_lighting(norm: &Vec3f, pix_pos: &Point3d, u: &Uniforms, occ: &[[Point3d; 3]], shadow_depth: &[f64], debug: bool) -> (f64, f64, f64) {
    // pixel to light
    let light_dir = (u.light_pos.into_vec() - pix_pos.into_vec()).normalize();

    let ambient = u.ambient;
    let diffuse_theta = Vec3f::dot(norm, &light_dir).max(0.0);
    let diffuse = diffuse_theta * u.diffuse;

    let view_dir = (pix_pos.into_vec() - u.cam_pos.into_vec()).normalize();
    let reflected = reflect(&light_dir.inv(), norm);
    let specular = Vec3f::dot(&view_dir.inv(), &reflected).max(0.0).powi(u.shininess as i32) * u.specular;

    let d = dist_3d(pix_pos, &u.light_pos).sqrt();
    let x = d / u.light_dissipation + 1.0;
    let diss = u.light_intensity / (x * x);

    let shadow = match u.shadow_mode {
        0 => 1.0,
        1 => perfect_shadow(occ, &u.light_pos, pix_pos),
        2 => approx_shadow(shadow_depth, u, pix_pos, norm, debug),
        _ => panic!("Invalid shadow mode"),
    };

    (ambient, diffuse * shadow * diss, specular * shadow * diss)
}

fn perfect_shadow(occ: &[[Point3d; 3]], light_pos: &Point3d, pix_pos: &Point3d) -> f64 {
    let line = line_3d_from_points(light_pos, pix_pos);
    let mut shadow = 1.0;

    // check for a collision with every triangle in the scene
    for tri in occ {
        let (a, b, c) = (&tri[0], &tri[1], &tri[2]);
        // the solve type here doesn't actually matter since we are never actually solving for this plane, just intersecting it
        let plane = plane_from_points(a, b, c, PlaneSolveType::Z);
        let inter = solve_line_plane(&line, &plane);

        // is the point of intersection within the triangle and between the pixel and the light?
        if is_point_between3d(pix_pos, light_pos, &inter) {
            if point_in_tri(&inter, [a, b, c]) {
                shadow = 0.0;
                break;
            }
        }
    }

    shadow
}

fn approx_shadow(shadow_depth: &[f64], u: &Uniforms, pix_pos: &Point3d, norm: &Vec3f, debug: bool) -> f64 {

    // position of the pixel in the clip space of the light source
    let shadow_view_pos = mul_point_matrix(pix_pos, &u.shadow_view);
    let shadow_proj_pos = mul_point_matrix(&shadow_view_pos, &u.shadow_proj);

    if shadow_proj_pos.x > 1.0 || shadow_proj_pos.x < -1.0 || shadow_proj_pos.y > 1.0 || shadow_proj_pos.y < -1.0 || shadow_proj_pos.z > 1.0 {
        return 0.0;
    }

    // retrieve z of the closest pixel to the camera at that x,y
    let x = (((shadow_proj_pos.x + 1.0) / 2.0) * u.shadow_buf_width as f64) as usize;
    let y = (((shadow_proj_pos.y + 1.0) / 2.0) * u.shadow_buf_height as f64) as usize;
    let shadow_depth = shadow_depth[(y * u.shadow_buf_width as usize) + x];

    if debug {
        println!("\n\nshadow_depth: {shadow_depth}");
    }

    // convert to view space z
    let shadow_pos_proj = Point3d::new(shadow_proj_pos.x, shadow_proj_pos.y, shadow_depth);
    let shadow_depth_view = mul_point_matrix(&shadow_pos_proj, &u.shadow_inv_proj).z;

    if debug {
        println!("shadow_depth_view: {shadow_depth_view}");
    }

    // find pixel pos in view space
    let pix_pos_view = mul_point_matrix(pix_pos, &u.shadow_view);

    if debug {
        println!("pix_pos_view: {pix_pos_view}");
    }

    // size of a pixel at the z of the shadow depth
    let per = u.shadow_perspective.as_ref().expect("Missing shadow perspective");
    let s = (2.0 * per.t * -shadow_depth_view) / (per.n * u.shadow_buf_height as f64);

    // cos(angle between light to pixel and the normal)
    let pix_to_light = Vec3f::new(0.0, 0.0, 1.0);
    let norm_view = mul_point_matrix(&norm.into_point(), &u.shadow_view.no_trans()).into_vec(); // transform norm to shadow view space
    let cos_theta = Vec3f::dot(&pix_to_light, &norm_view);

    // tiny value to add to delta to account for limited precision
    const MU: f64 = 0.000001;

    // calculate change in z to fully conceal all shadow map pixels under the polygon
    // = s / cot(cos-1(cos_theta)) = s / (cos_theta / sqrt(1 - cos_theta^2)) = s*sqrt(1 - cos_theta^2) / cos_theta
    // let delta = s * (1.0 - cos_theta*cos_theta).sqrt() / cos_theta;
    let delta = s * cos_theta.acos().tan();

    if debug {
        println!("s: {s}");
        println!("pix_to_light: {pix_to_light}");
        println!("norm_view: {norm_view}");
        println!("cos_theta: {cos_theta}");
        println!("delta: {delta}");
    }

    // println!("{}", delta);

    let diff = pix_pos_view.z - (shadow_depth_view - 3.0 * delta);
    // let diff = shadow_proj_pos.z - (shadow_depth + 0.001);

    if debug {
        println!("diff: {diff}");
    }

    if diff < 0.0 { // TODO: add a variable amount proportional to distance from cam
        0.0
    } else {
        1.0
    }
}

// all parameters must be normalized
fn reflect(incoming: &Vec3f, norm: &Vec3f) -> Vec3f {
    let inc = incoming.inv();
    // the cos of the angle between our incoming vec and our norm
    let cos_theta = Vec3f::dot(&inc, norm);

    // the point along our norm with distance `cos_theta` from the origin
    // angle > 90 handles itself because our norm will be inverted due to a negative cos
    let norm_int = Point3d::new(
        lerp(0.0, norm.x, cos_theta),
        lerp(0.0, norm.y, cos_theta),
        lerp(0.0, norm.z, cos_theta)
    ).into_vec();

    // vector from our incoming vec to the above point
    let inc_norm_int = norm_int - inc;
    let reflected = norm_int + inc_norm_int;
    reflected
}

// slowest function by far
// optimizations needed (likely not possible, the slowness is from slow memory access)
fn tex_sample(tex: &RgbaImage, x: f64, y: f64, sample_lerp: bool) -> [u8; 4] {
    // confine coords to be between 0 and 1
    // adding 0.5 to each coord essentially puts texel coords in the center of pixels instead of the bottom left corner
    // that allows us to really sample the nearest texel
    let x = x.fract() * (tex.width() - 1) as f64 + 0.5;
    let y = y.fract() * (tex.height() - 1) as f64 + 0.5;

    if sample_lerp {
        let t = (x as u32, y as u32);
        let t_x = (x as u32 + 1, y as u32);
        let t_y = (x as u32, y as u32 + 1);
        let t_xy = (x as u32 + 1, y as u32 + 1);

        // the four texels we care about
        const BLACK: image::Rgba<u8> = image::Rgba([0; 4]);
        let this = tex.get_pixel_checked(t.0, t.1).unwrap_or(&BLACK).0;
        let this_x = tex.get_pixel_checked(t_x.0, t_x.1).unwrap_or(&BLACK).0;
        let this_y = tex.get_pixel_checked(t_y.0, t_y.1).unwrap_or(&BLACK).0;
        let this_xy = tex.get_pixel_checked(t_xy.0, t_xy.1).unwrap_or(&BLACK).0;

        // how close our pixel to the bounds of the texel
        let dx = 1.0 - ((x + 1.0).trunc() - x);
        let dy = 1.0 - ((y + 1.0).trunc() - y);

        // color at the lerped x values
        let lerp_x_low = lerp_color(&this, &this_x, dx);
        let lerp_x_high = lerp_color(&this_y, &this_xy, dx);

        lerp_color(&lerp_x_low, &lerp_x_high, dy)
    } else {
        tex.get_pixel(x as u32, y as u32).0
    }
}

fn lerp_color(c1: &[u8; 4], c2: &[u8; 4], x: f64) -> [u8; 4] {
    [lerp(c1[0] as f64, c2[0] as f64, x) as u8,
    lerp(c1[1] as f64, c2[1] as f64, x) as u8,
    lerp(c1[2] as f64, c2[2] as f64, x) as u8,
    lerp(c1[3] as f64, c2[3] as f64, x) as u8]
}

fn add_color(c1: &[u8; 4], c2: &[u8; 4]) -> [u8; 4] {
    [c1[0].saturating_add(c2[0]),
    c1[1].saturating_add(c2[1]),
    c1[2].saturating_add(c2[2]),
    c1[3].saturating_add(c2[3])]
}

fn mul_color(color: &[u8; 4], x: f64) -> [u8; 4] {
    [(color[0] as f64 * x) as u8,
    (color[1] as f64 * x) as u8,
    (color[2] as f64 * x) as u8,
    color[3]]
}

pub fn sort_tri_points_y(tri: &mut Triangle) {
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