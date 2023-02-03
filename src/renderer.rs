use std::mem::swap;
use crate::{math::*, SubBuffer};
use image::RgbaImage;

#[derive(Clone, Debug)]
pub struct Triangle<'a> {
    pub a: Vertex,
    pub b: Vertex,
    pub c: Vertex,
    pub tex: Option<&'a RgbaImage>,
    pub clipped: bool,

    // directions of vertices to other vertices (used for point_in_tri())
    pub ab: Vec3f,
    pub ba: Vec3f,
    pub ac: Vec3f,
    pub bc: Vec3f,
}

#[derive(Clone, Debug)]
pub struct Vertex {
    pub pos: Point3d, // x and y are in screen space for rasterization, z is still in clip space
    pub pos_world: Point3d,
    pub pos_clip: Point3d,
    pub color: [u8; 3],
    pub n: Vec3f,
    pub tex: [f64; 2],
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

pub struct Uniforms {
    pub model: Mat4f,
    pub proj: Mat4f,
    pub inv_proj: Mat4f,
    pub light_pos: Point3d,
    pub ambient: f64,
    pub diffuse: f64,
    pub specular: f64,
    pub shininess: u32,
    pub legacy: bool,
    pub render_shadows: bool,
}

pub fn rasterize(buf: &mut SubBuffer, tri: &Triangle, occ: &[Triangle], u: &Uniforms) -> u32 {
    let dims = buf.dims;
    let (a, b, c) = (&tri.a.pos, &tri.b.pos, &tri.c.pos);
    let (a_world, b_world, c_world) = (&tri.a.pos_world, &tri.b.pos_world, &tri.c.pos_world);
    let planes = AttributePlanes {
        color_r: plane_from_points_z(a, b, c, tri.a.color[0] as f64, tri.b.color[0] as f64, tri.c.color[0] as f64),
        color_g: plane_from_points_z(a, b, c, tri.a.color[1] as f64, tri.b.color[1] as f64, tri.c.color[1] as f64),
        color_b: plane_from_points_z(a, b, c, tri.a.color[2] as f64, tri.b.color[2] as f64, tri.c.color[2] as f64),
        n_x: plane_from_points_z(a_world, b_world, c_world, tri.a.n.x, tri.b.n.x, tri.c.n.x),
        n_y: plane_from_points_z(a_world, b_world, c_world, tri.a.n.y, tri.b.n.y, tri.c.n.y),
        n_z: plane_from_points_z(a_world, b_world, c_world, tri.a.n.z, tri.b.n.z, tri.c.n.z),
        tex_x: plane_from_points_z(a_world, b_world, c_world, tri.a.tex[0], tri.b.tex[0], tri.c.tex[0]),
        tex_y: plane_from_points_z(a_world, b_world, c_world, tri.a.tex[1], tri.b.tex[1], tri.c.tex[1]),
        pos_x: plane_from_points_z(a, b, c, tri.a.pos_clip.x, tri.b.pos_clip.x, tri.c.pos_clip.x),
        pos_y: plane_from_points_z(a, b, c, tri.a.pos_clip.y, tri.b.pos_clip.y, tri.c.pos_clip.y),
        pos_z: plane_from_points_z(a, b, c, tri.a.pos_clip.z, tri.b.pos_clip.z, tri.c.pos_clip.z),
    };

    let mut pixels_shaded = 0;
    for i in 0..2 {
        let (mut start_y, mut end_y) = match i {
            0 => (tri.a.pos.y.ceil() as u32, tri.b.pos.y.ceil() as u32),
            1 => (tri.b.pos.y.ceil() as u32, tri.c.pos.y.ceil() as u32),
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

            // YOU CANNOT ROUND HERE. It creates situtations where the start and end x are outside our tri
            let mut start_x = start_x.ceil() as u32;
            let mut end_x = end_x.ceil() as u32;
            if start_x > end_x {
                swap(&mut start_x, &mut end_x);
            }

            start_x = start_x.clamp(0, dims.0 - 1);
            end_x = end_x.clamp(0, dims.0 - 1);
            for x in start_x..end_x {
                // localize the index to the sub buffer
                let i = (y - buf.start_y) * dims.0 + x;
                let depth = lerp_fast(&planes.pos_z, x as f64, y as f64);
                if depth < buf.depth[i as usize] && depth >= super::SCREEN_Z {
                    let color = pixel_shader(tri, occ, u, &planes, x, y);

                    // discard transparency
                    if color[3] == 0 {
                        continue;
                    }

                    buf.color[i as usize * 3] = color[0];
                    buf.color[i as usize * 3 + 1] = color[1];
                    buf.color[i as usize * 3 + 2] = color[2];
                    buf.depth[i as usize] = depth;
                    pixels_shaded += 1;
                }
            };
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

// mutates the triangle in place and leaves it in raster space
pub fn vertex_shader(tri: &mut Triangle, u: &Uniforms, dims: (u32, u32)) {
    let (a, b, c) = (&mut tri.a, &mut tri.b, &mut tri.c);

    a.pos = mul_point_matrix(&a.pos, &u.model);
    b.pos = mul_point_matrix(&b.pos, &u.model);
    c.pos = mul_point_matrix(&c.pos, &u.model);

    a.pos_world = a.pos;
    b.pos_world = b.pos;
    c.pos_world = c.pos;

    // no non-uniform scaling is actually done to our points, so normals will be fine too.
    a.n = mul_point_matrix(&a.n.into_point(), &u.model.no_trans()).into_vec().normalize();
    b.n = mul_point_matrix(&b.n.into_point(), &u.model.no_trans()).into_vec().normalize();
    c.n = mul_point_matrix(&c.n.into_point(), &u.model.no_trans()).into_vec().normalize();

    // primitive implementation of clipping (so z !>= 0 for perspective division, otherwise weird stuff unfolds)
    if (a.pos.z >= 0.0 || b.pos.z >= 0.0 || c.pos.z >= 0.0) && !u.legacy {
        // triangle (at least one vertex) was clipped, we cannot render it
        tri.clipped = true;
        return;
    }

    a.pos = mul_point_matrix(&a.pos, &u.proj);
    b.pos = mul_point_matrix(&b.pos, &u.proj);
    c.pos = mul_point_matrix(&c.pos, &u.proj);

    a.pos_clip = a.pos;
    b.pos_clip = b.pos;
    c.pos_clip = c.pos;

    // normalize to 0 to 1 and scale to raster space
    a.pos.x = (a.pos.x + 1.0) / 2.0 * dims.0 as f64;
    b.pos.x = (b.pos.x + 1.0) / 2.0 * dims.0 as f64;
    c.pos.x = (c.pos.x + 1.0) / 2.0 * dims.0 as f64;
    a.pos.y = (a.pos.y + 1.0) / 2.0 * dims.1 as f64;
    b.pos.y = (b.pos.y + 1.0) / 2.0 * dims.1 as f64;
    c.pos.y = (c.pos.y + 1.0) / 2.0 * dims.1 as f64;
}

const FULLY_OPAQUE: u8 = 255;
const INTERPOLATE_FAST: bool = true;

/*
Interpolate_fast treats our attributes as the z values of our
triangles, then solves for z when x and y are known at any
arbitrary point in that plane.

Interpolate_slow finds the maximum distance away from a vertex
that a point on our triangle can be, as well as the actual distance.
From there, it can produce a "weight" for each vertex, to be
multiplied with our attributes.

An interesting thing to note is that if a 2 or more verts of a tri have perfectly equal x and y values in world space,
interpolation won't actually work. This is because any interpolation plane created with this triangle
in mind will have an undefined slope. Therefore, trying to solve for a z value on this plane from
a know x and y will result in a `NaN` or `inf`. The only way to solve this would be to do interpolation in 4
dimensions and solve for w from a know x, y, and z, which im definetly not gonna do.
*/

fn pixel_shader(tri: &Triangle, occ: &[Triangle], u: &Uniforms, planes: &AttributePlanes, x: u32, y: u32) -> [u8; 4] {
    let (x, y) = (x as f64, y as f64);

    if INTERPOLATE_FAST {
        // position of our pixel in clip space
        let pix_clip_pos = Point3d::new(
            lerp_fast(&planes.pos_x, x, y),
            lerp_fast(&planes.pos_y, x, y),
            lerp_fast(&planes.pos_z, x, y),
        );

        // position of our pixel in world space
        let pix_world_pos = mul_point_matrix(&pix_clip_pos, &u.inv_proj);
        let (x_world, y_world) = (pix_world_pos.x, pix_world_pos.y);

        let norm = Vec3f::new(
            lerp_fast(&planes.n_x, x_world, y_world),
            lerp_fast(&planes.n_y, x_world, y_world),
            lerp_fast(&planes.n_z, x_world, y_world),
        );

        let base_color = if let Some(tex) = tri.tex {
            // we calculate texture coords with respect to our pixel's world space position rather than it's clip or raster space position
            // because otherwise we get incorrect results. we do the same with some other attributes too.
            let vt_x = lerp_fast(&planes.tex_x, x_world, y_world);
            let vt_y = lerp_fast(&planes.tex_y, x_world, y_world);
            tex_sample(tex, vt_x, vt_y)
        } else {
            [lerp_fast(&planes.color_r, x, y).round() as u8,
            lerp_fast(&planes.color_g, x, y).round() as u8,
            lerp_fast(&planes.color_b, x, y).round() as u8,
            FULLY_OPAQUE]
        };

        let color = mul_color(&base_color, calc_lighting(&norm, &pix_world_pos, u, occ));
        color
    } else {
        let weights_clip = lerp_slow([&tri.a.pos, &tri.b.pos, &tri.c.pos], x, y);
        let pix_clip_pos = Point3d::new(
            tri.a.pos_clip.x * weights_clip.a + tri.b.pos_clip.x * weights_clip.b + tri.c.pos_clip.x * weights_clip.c,
            tri.a.pos_clip.y * weights_clip.a + tri.b.pos_clip.y * weights_clip.b + tri.c.pos_clip.y * weights_clip.c,
            tri.a.pos_clip.z * weights_clip.a + tri.b.pos_clip.z * weights_clip.b + tri.c.pos_clip.z * weights_clip.c,
        );
        let pix_world_pos = mul_point_matrix(&pix_clip_pos, &u.inv_proj);
        let weights_world = lerp_slow([&tri.a.pos_world, &tri.b.pos_world, &tri.c.pos_world], pix_world_pos.x, pix_world_pos.y);

        let base_color = if let Some(tex) = tri.tex {
            let vt_x = tri.a.tex[0] * weights_world.a + tri.b.tex[0] * weights_world.b + tri.c.tex[0] * weights_world.c;
            let vt_y = tri.a.tex[1] * weights_world.a + tri.b.tex[1] * weights_world.b + tri.c.tex[1] * weights_world.c;
            tex_sample(tex, vt_x, vt_y)
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
        );

        let color = mul_color(&base_color, calc_lighting(&norm, &pix_world_pos, u, occ));
        color
    }
}

fn calc_lighting(norm: &Vec3f, pix_pos: &Point3d, u: &Uniforms, occ: &[Triangle]) -> f64 {
    // pixel to light
    let light_dir = (u.light_pos.into_vec() - pix_pos.into_vec()).normalize();

    let ambient = u.ambient;
    let diffuse_theta = Vec3f::dot(norm, &light_dir).max(0.0);
    let diffuse = diffuse_theta * u.diffuse;

    // our cam is always at the origin, so view dir is just the pixel pos (cam to pixel)
    let view_dir = pix_pos.into_vec().normalize();
    let reflected = reflect(&light_dir.inv(), norm);
    // multiply with ceil of diffuse theta to stop physically inaccurate instances of specular lighting when the light source is just barely behind the tri
    let specular = Vec3f::dot(&view_dir.inv(), &reflected).max(0.0).powi(u.shininess as i32) * u.specular * diffuse_theta.ceil();

    let shadow = if u.render_shadows {
        shadow(occ, &u.light_pos, pix_pos)
    } else {
        1.0
    };

    ambient + (diffuse + specular) * shadow
}

// theres something wrong with pix_pos
fn shadow(occ: &[Triangle], light_pos: &Point3d, pix_pos: &Point3d) -> f64 {
    let line = line_3d_from_points(light_pos, pix_pos);
    let mut shadow = 1.0;

    // check for a collision with every triangle in the scene
    for tri in occ {
        let (a, b, c) = (&tri.a.pos_world, &tri.b.pos_world, &tri.c.pos_world);
        let plane = plane_from_points(a, b, c);
        let inter = solve_line_plane(&line, &plane);

        // is the point of intersection within the triangle and between the pixel and the light?
        if is_point_between(pix_pos, light_pos, &inter) {
            if point_in_tri(&inter, [a, b, c], &tri.ab, &tri.ba, &tri.ac, &tri.bc) {
                shadow = 0.0;
                break;
            }
        }
    }

    shadow
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

fn mul_color(color: &[u8; 4], x: f64) -> [u8; 4] {
    [(color[0] as f64 * x) as u8,
    (color[1] as f64 * x) as u8,
    (color[2] as f64 * x) as u8,
    color[3]]
}

// slowest function by far
// optimizations needed (likely not possible)
fn tex_sample(tex: &RgbaImage, x: f64, y: f64) -> [u8; 4] {
    // confine coords to be between 0 and 1
    // adding 0.5 to each coord essentially puts texel coords in the center of pixels instead of the bottom left corner
    // that allows us to really sample the nearest texel
    let x = x.fract() * (tex.width() - 1) as f64 + 0.5;
    let y = y.fract() * (tex.height() - 1) as f64 + 0.5;

    let t = (x as u32, y as u32);
    let t_x = (x as u32 + 1, y as u32);
    let t_y = (x as u32, y as u32 + 1);
    let t_xy = (x as u32 + 1, y as u32 + 1);

    // the four texels we care about
    let this = if t.0 < tex.width() && t.1 < tex.height() {
        tex.get_pixel(t.0, t.1).0
    } else {
        [0; 4]
    };
    let this_x = if t_x.0 < tex.width() && t_x.1 < tex.height() {
        tex.get_pixel(t_x.0, t_x.1).0
    } else {
        [0; 4]
    };
    let this_y = if t_y.0 < tex.width() && t_y.1 < tex.height() {
        tex.get_pixel(t_y.0, t_y.1).0
    } else {
        [0; 4]
    };
    let this_xy = if t_xy.0 < tex.width() && t_xy.1 < tex.height() {
        tex.get_pixel(t_xy.0, t_xy.1).0
    } else {
        [0; 4]
    };

    // how close our pixel is the four texels
    let dx = 1.0 - ((x + 1.0).trunc() - x);
    let dy = 1.0 - ((y + 1.0).trunc() - y);

    // color at the lerped x values
    let lerp_x_low = lerp_color(&this, &this_x, dx);
    let lerp_x_high = lerp_color(&this_y, &this_xy, dx);

    lerp_color(&lerp_x_low, &lerp_x_high, dy)
}

fn lerp_color(c1: &[u8; 4], c2: &[u8; 4], x: f64) -> [u8; 4] {
    [lerp(c1[0] as f64, c2[0] as f64, x) as u8,
    lerp(c1[1] as f64, c2[1] as f64, x) as u8,
    lerp(c1[2] as f64, c2[2] as f64, x) as u8,
    lerp(c1[3] as f64, c2[3] as f64, x) as u8]
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