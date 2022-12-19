use std::fmt::{Display, Formatter};
use super::Point;

pub struct Perspective {
    n: f64,
    f: f64,
    r: f64,
    l: f64,
    t: f64,
    b: f64,
}

pub fn get_perspective(fov: f64, aspect: f64, n: f64, f: f64) -> Perspective {
    let scale = (fov / 2.0).to_radians().tan() * n;
    let r = aspect * scale;
    let t = scale;
    Perspective {
        n,
        f,
        r,
        l: -r,
        t,
        b: -t
    }
}

pub fn frustum(p: &Perspective) -> Mat4f {
    let mut mat = Mat4f::new();
    let (n, f, r, l, t, b) = (p.n, p.f, p.r, p.l, p.t, p.b);

    mat.r0c0 = (2.0 * n) / (r - l);
    mat.r0c1 = 0.0;
    mat.r0c2 = (r + l) / (r - l);
    mat.r0c3 = 0.0;

    mat.r1c0 = 0.0;
    mat.r1c1 = (2.0 * n) / (t - b);
    mat.r1c2 = (t + b) / (t - b);
    mat.r1c3 = 0.0;

    mat.r2c0 = 0.0;
    mat.r2c1 = 0.0;
    mat.r2c2 = -(f + n) / (f - n);
    mat.r2c3 = -2.0 * f * n / (f - n);

    mat.r3c0 = 0.0;
    mat.r3c1 = 0.0;
    mat.r3c2 = -1.0;
    mat.r3c3 = 0.0;

    mat
}

pub fn mul_point_matrix(point: &Point, mat: &Mat4f) -> Point {
    let mut out = Point::new(0.0, 0.0, 0.0);

    out.x = point.x * mat.r0c0 + point.y * mat.r0c1 + point.z * mat.r0c2 + mat.r0c3;
    out.y = point.x * mat.r1c0 + point.y * mat.r1c1 + point.z * mat.r1c2 + mat.r1c3;
    out.z = point.x * mat.r2c0 + point.y * mat.r2c1 + point.z * mat.r2c2 + mat.r2c3;
    let w = point.x * mat.r3c0 + point.y * mat.r3c1 + point.z * mat.r3c2 + mat.r3c3;

    out.x /= w;
    out.y /= w;
    out.z /= w;

    out
}

pub struct Mat4f {
    pub r0c0: f64,
    pub r0c1: f64,
    pub r0c2: f64,
    pub r0c3: f64,
    pub r1c0: f64,
    pub r1c1: f64,
    pub r1c2: f64,
    pub r1c3: f64,
    pub r2c0: f64,
    pub r2c1: f64,
    pub r2c2: f64,
    pub r2c3: f64,
    pub r3c0: f64,
    pub r3c1: f64,
    pub r3c2: f64,
    pub r3c3: f64,
}

impl Mat4f {
    pub fn new() -> Self {
        Self {
            r0c0: 1.0, r0c1: 0.0, r0c2: 0.0, r0c3: 0.0,
            r1c0: 0.0, r1c1: 1.0, r1c2: 0.0, r1c3: 0.0,
            r2c0: 0.0, r2c1: 0.0, r2c2: 1.0, r2c3: 0.0,
            r3c0: 0.0, r3c1: 0.0, r3c2: 0.0, r3c3: 1.0,
        }
    }
}

impl Display for Mat4f {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        f.write_fmt(format_args!(
            "[{}] [{}] [{}] [{}]\n[{}] [{}] [{}] [{}]\n[{}] [{}] [{}] [{}]\n[{}] [{}] [{}] [{}]",
            self.r0c0, self.r0c1, self.r0c2, self.r0c3,
            self.r1c0, self.r1c1, self.r1c2, self.r1c3,
            self.r2c0, self.r2c1, self.r2c2, self.r2c3,
            self.r3c0, self.r3c1, self.r3c2, self.r3c3,
        ))
    }
}