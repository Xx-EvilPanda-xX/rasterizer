use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut, Sub, Add};

const MU: f64 = 0.0000001;

// `p` is assumed to lie on the same plane as `tri`
pub fn point_in_tri(p: &Point3d, tri: [&Point3d; 3]) -> bool {
    let (a, b, c) = (tri[0].into_vec(), tri[1].into_vec(), tri[2].into_vec());

    let ap = (p.into_vec() - a).normalize();
    let bp = (p.into_vec() - b).normalize();

    // all direction vecs of our tri
    let ab = (b - a).normalize();
    let ba = (a - b).normalize();
    let ac = (c - a).normalize();
    let bc = (c - b).normalize();

    // angle at point a and point b on our tri
    let theta_a = Vec3f::dot(&ab, &ac);
    let theta_b = Vec3f::dot(&ba, &bc);

    // angles between our point of intersection and the sides of our tri
    let theta_iab = Vec3f::dot(&ap, &ab);
    let theta_iac = Vec3f::dot(&ap, &ac);
    let theta_iba = Vec3f::dot(&bp, &ba);
    let theta_ibc = Vec3f::dot(&bp, &bc);

    // we invert the comparison becuase cos is, in some sense, proportional to the negative of the angle
    (theta_iab > theta_a && theta_iac > theta_a) && (theta_iba > theta_b && theta_ibc > theta_b)
}

// finds the point of intersection between a line and a plane
pub fn solve_line_plane(l: &Line3d, p: &Plane) -> Point3d {
    let x = (p.d - p.b * l.b_y - p.c * l.b_z) / (p.a + p.b * l.m_y + p.c * l.m_z);
    let y = l.m_y * x + l.b_y;
    let z = l.m_z * x + l.b_z;
    Point3d::new(x, y, z)
}

// simple linear interpolation between two scalar values
pub fn lerp(a: f64, b: f64, c: f64) -> f64 {
    a + c * (b - a)
}

// solve for z given x and y with respect to a plane
pub fn lerp_fast(p: &Plane, x: f64, y: f64) -> f64 {
    let point = Point2d::new(x as f64, y as f64);
    let z = (p.d - p.a * point.x - p.b * point.y) / p.c;
    z
}

pub struct VertexWeights {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

// calculates a "weight" for each vertex in our tri by comparing the distances between the vertices and our point
pub fn lerp_slow(tri: [&Point3d; 3], x: f64, y: f64) -> VertexWeights {
    let (a, b, c) = (&tri[0].into_2d(), &tri[1].into_2d(), &tri[2].into_2d());
    let p = Point2d::new(x, y);

    // sides of our triangle
    let ab = line_2d_from_points(a, b);
    let bc = line_2d_from_points(b, c);
    let ac = line_2d_from_points(a, c);

    // distance from each vertex
    let dist_a = dist_2d(&p, a);
    let dist_b = dist_2d(&p, b);
    let dist_c = dist_2d(&p, c);

    // line passing through each vertex and our point, then find the point of intersection with opposite side
    let ap = line_2d_from_points(a, &p);
    let ap_bc = solve_lines(&ap, &bc);
    let max_dist_a = dist_2d(a, &ap_bc);

    let bp = line_2d_from_points(b, &p);
    let bp_ac = solve_lines(&bp, &ac);
    let max_dist_b = dist_2d(b, &bp_ac);

    let cp = line_2d_from_points(c, &p);
    let cp_ab = solve_lines(&cp, &ab);
    let max_dist_c = dist_2d(c, &cp_ab);

    // weight vertices based off distance from the point
    let a_weight = 1.0 - (dist_a / max_dist_a).sqrt();
    let b_weight = 1.0 - (dist_b / max_dist_b).sqrt();
    let c_weight = 1.0 - (dist_c / max_dist_c).sqrt();

    VertexWeights { a: a_weight, b: b_weight, c: c_weight }
}

// distance squared in 2d
pub fn dist_2d(a: &Point2d, b: &Point2d) -> f64 {
    let diff_x = a.x - b.x;
    let diff_y = a.y - b.y;
    diff_x * diff_x + diff_y * diff_y
}

// distance squared in 3d
pub fn dist_3d(a: &Point3d, b: &Point3d) -> f64 {
    let diff_x = a.x - b.x;
    let diff_y = a.y - b.y;
    let diff_z = a.z - b.z;
    diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
}

// find m and b from two points
pub fn line_2d_from_points(p1: &Point2d, p2: &Point2d) -> Line2d {
    let dy = p1.y - p2.y;
    let dx = p1.x - p2.x;

    if dx == 0.0 {
        return Line2d { m: p1.x, b: f64::INFINITY };
    }

    let m = dy / dx;
    let b = p1.y - dy * p1.x / dx;

    Line2d { m, b }
}

// find y and z slope and intercepts from two points
pub fn line_3d_from_points(p1: &Point3d, p2: &Point3d) -> Line3d {
    let line_y = line_2d_from_points(&Point2d::new(p1.x, p1.y), &Point2d::new(p2.x, p2.y));
    let line_z = line_2d_from_points(&Point2d::new(p1.x, p1.z), &Point2d::new(p2.x, p2.z));

    Line3d {
        m_y: line_y.m,
        b_y: line_y.b,
        m_z: line_z.m,
        b_z: line_z.b,
    }
}

// checks whether `between` is between p1 and p2 (assumes all inputs are colinear)
pub fn is_point_between(p1: &Point3d, p2: &Point3d, between: &Point3d) -> bool {
    // small correction term to prevent floating point precision errors (negated to prefer returning false when its close)
    let max_dist = dist_3d(p1, p2) - MU;
    let to_p1 = dist_3d(p1, between);
    let to_p2 = dist_3d(p2, between);

    to_p1 < max_dist && to_p2 < max_dist
}

// only the x and y components of the points being passed in here are used, the z coming from the last args
pub fn plane_from_points_z(p1: &Point3d, p2: &Point3d, p3: &Point3d, z1: f64, z2: f64, z3: f64) -> Plane {
    plane_from_points(&Point3d::new(p1.x, p1.y, z1), &Point3d::new(p2.x, p2.y, z2), &Point3d::new(p3.x, p3.y, z3))
}

// finds plane in the form of ax + by + cz = d from three points
pub fn plane_from_points(p1: &Point3d, p2: &Point3d, p3: &Point3d) -> Plane {
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
pub fn solve_lines(l1: &Line2d, l2: &Line2d) -> Point2d {
    // account for lines that are edge cases to the below formula
    if l1.b.is_infinite() {
        return Point2d::new(l1.m, solve_y(l2, l1.m));
    }
    if l2.b.is_infinite() {
        return Point2d::new(l2.m, solve_y(l1, l2.m));
    }
    if l1.m == 0.0 {
        return Point2d::new(solve_x(l2, l1.b), l1.b);
    }
    if l2.m == 0.0 {
        return Point2d::new(solve_x(l1, l2.b), l2.b);
    }

    let y = (l2.b * l1.m - l1.b * l2.m) / (l1.m - l2.m);
    let x = solve_x(l1, y);
    Point2d::new(x, y)
}

pub fn solve_x(l: &Line2d, y: f64) -> f64 {
    (y - l.b) / l.m
}

pub fn solve_y(l: &Line2d, x: f64) -> f64 {
    l.m * x + l.b
}

// calculate the perspective projection matrix given the n, f, t, b, r, and l planes
pub fn frustum(p: &Perspective) -> Mat4f {
    let mut mat = Mat4f::new();
    let (n, f, r, l, t, b) = (p.n, p.f, p.r, p.l, p.t, p.b);

    mat[0][0] = (2.0 * n) / (r - l);
    mat[0][1] = 0.0;
    mat[0][2] = (r + l) / (r - l);
    mat[0][3] = 0.0;

    mat[1][0] = 0.0;
    mat[1][1] = (2.0 * n) / (t - b);
    mat[1][2] = (t + b) / (t - b);
    mat[1][3] = 0.0;

    mat[2][0] = 0.0;
    mat[2][1] = 0.0;
    mat[2][2] = -(f + n) / (f - n);
    mat[2][3] = -2.0 * f * n / (f - n);

    mat[3][0] = 0.0;
    mat[3][1] = 0.0;
    mat[3][2] = -1.0;
    mat[3][3] = 0.0;

    mat
}

// multiply a 3d point with a 4x4 matrix
pub fn mul_point_matrix(point: &Point3d, mat: &Mat4f) -> Point3d {
    let mut out = Point3d::origin();

    out.x = point.x * mat[0][0] + point.y * mat[0][1] + point.z * mat[0][2] + mat[0][3];
    out.y = point.x * mat[1][0] + point.y * mat[1][1] + point.z * mat[1][2] + mat[1][3];
    out.z = point.x * mat[2][0] + point.y * mat[2][1] + point.z * mat[2][2] + mat[2][3];
    let w = point.x * mat[3][0] + point.y * mat[3][1] + point.z * mat[3][2] + mat[3][3];

    out.x /= w;
    out.y /= w;
    out.z /= w;

    out
}

// multiply a 4x4 matrix with a 4x4 matrix
pub fn mul_matrix_matrix(m1: &Mat4f, m2: &Mat4f) -> Mat4f {
    let mut out = Mat4f::new();

    for i in 0..4 {
        for j in 0..4 {
            out[j][i] = row_col_mul(m1, m2, j, i);
        }
    }

    out
}

fn row_col_mul(m1: &Mat4f, m2: &Mat4f, row: usize, col: usize) -> f64 {
    let mut out = 0.0;

    for i in 0..4 {
        out += m1[row][i] * m2[i][col]
    }

    out
}

pub struct Perspective {
    n: f64,
    f: f64,
    r: f64,
    l: f64,
    t: f64,
    b: f64,
}

// calculate n, f, r, l, t, and b from fov, aspect, n, and f
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

#[derive(Clone, Copy, Debug)]
pub struct Point3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Point2d {
    pub x: f64,
    pub y: f64,
}

// if undefined slope, m = x intercept and b = infinity
#[derive(Clone, Copy, Debug)]
pub struct Line2d {
    pub m: f64,
    pub b: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Line3d {
    pub m_y: f64,
    pub b_y: f64,
    pub m_z: f64,
    pub b_z: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mat4f {
    pub mat: [[f64; 4]; 4],
}

#[derive(Debug, Clone, Copy)]
pub struct Vec3f {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3d {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn origin() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn from_arr(a: [f64; 3]) -> Self {
        Self { x: a[0], y: a[1], z: a[2] }
    }

    pub fn into_vec(&self) -> Vec3f {
        Vec3f { x: self.x, y: self.y, z: self.z }
    }

    pub fn into_2d(&self) -> Point2d {
        Point2d { x: self.x, y: self.y }
    }
}

impl Point2d {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl Mat4f {
    pub fn new() -> Self {
        Self {
            mat: [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
        }
    }

    pub fn no_trans(&self) -> Self {
        let mut new = self.clone();
        new[0][3] = 0.0;
        new[1][3] = 0.0;
        new[2][3] = 0.0;
        new
    }

    // Gauss-Jordan Method
    pub fn inverse(&self) -> Self {
        let mut aug_mat = [[0.0; 8]; 4];

        // augment an identity matrix onto the end of the matrix
        for i in 0..4 {
            for j in 0..4 {
                aug_mat[i][j] = self[i][j];
            }
            aug_mat[i][i + 4] = 1.0;
        }

        for i in 0..4 {
            for j in 0..4 {
                if j != i {
                    let div = aug_mat[j][i] / aug_mat[i][i];
                    for k in 0..8 {
                        aug_mat[j][k] -= aug_mat[i][k] * div;
                    }
                }
            }
        }

        for i in 0..4 {
            let diag = aug_mat[i][i];
            for j in 0..8 {
                aug_mat[i][j] /= diag;
            }
        }

        let mut out = Self::new();

        for i in 0..4 {
            for j in 0..4 {
                out[i][j] = aug_mat[i][j + 4];
            }
        }

        out
    }
}

impl Index<usize> for Mat4f {
    type Output = [f64; 4];

    fn index(&self, index: usize) -> &Self::Output {
        &self.mat[index]
    }
}

impl IndexMut<usize> for Mat4f {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.mat[index]
    }
}

impl Display for Mat4f {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        f.write_fmt(format_args!(
            "[{:.2}] [{:.2}] [{:.2}] [{:.2}]\n[{:.2}] [{:.2}] [{:.2}] [{:.2}]\n[{:.2}] [{:.2}] [{:.2}] [{:.2}]\n[{:.2}] [{:.2}] [{:.2}] [{:.2}]",
            self[0][0], self[0][1], self[0][2], self[0][3],
            self[1][0], self[1][1], self[1][2], self[1][3],
            self[2][0], self[2][1], self[2][2], self[2][3],
            self[3][0], self[3][1], self[3][2], self[3][3],
        ))
    }
}

impl Vec3f {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
        }
    }

    pub fn dot(a: &Vec3f, b: &Vec3f) -> f64 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    pub fn cross(a: &Vec3f, b: &Vec3f) -> Self {
        Self {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }

    pub fn normalize(&self) -> Self {
        let len = self.mag();
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    pub fn mag(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn inv(&self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    pub fn into_point(&self) -> Point3d {
        Point3d { x: self.x, y: self.y, z: self.z }
    }
}

impl Sub for Vec3f {
    type Output = Vec3f;

    fn sub(self, rhs: Self) -> Self::Output {
        Vec3f {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Add for Vec3f {
    type Output = Vec3f;

    fn add(self, rhs: Self) -> Self::Output {
        Vec3f {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Default for Vec3f {
    fn default() -> Self {
        Vec3f { x: 0.0, y: 0.0, z: 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::Mat4f;

    #[test]
    fn test_mul_matrix_matrix() {
        let mut m1 = Mat4f::new();
        let mut m2 = Mat4f::new();
        let mut result = Mat4f::new();

        m1[0][0] = 1.0;
        m1[0][1] = 2.0;
        m1[0][2] = 3.0;
        m1[1][0] = 4.0;
        m1[1][1] = 5.0;
        m1[1][2] = 6.0;
        m1[2][0] = 7.0;
        m1[2][1] = 8.0;
        m1[2][2] = 9.0;

        m2[0][0] = 1.0;
        m2[0][1] = 2.0;
        m2[0][2] = 3.0;
        m2[1][0] = 4.0;
        m2[1][1] = 5.0;
        m2[1][2] = 6.0;
        m2[2][0] = 7.0;
        m2[2][1] = 8.0;
        m2[2][2] = 9.0;

        result[0][0] = 30.0;
        result[0][1] = 36.0;
        result[0][2] = 42.0;
        result[1][0] = 66.0;
        result[1][1] = 81.0;
        result[1][2] = 96.0;
        result[2][0] = 102.0;
        result[2][1] = 126.0;
        result[2][2] = 150.0;

        assert_eq!(result, super::mul_matrix_matrix(&m1, &m2));
    }

}