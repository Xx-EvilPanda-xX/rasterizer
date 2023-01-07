use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut, Sub, Add};
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

pub fn mul_point_matrix(point: &Point, mat: &Mat4f) -> Point {
    let mut out = Point::new(0.0, 0.0, 0.0);

    out.x = point.x * mat[0][0] + point.y * mat[0][1] + point.z * mat[0][2] + mat[0][3];
    out.y = point.x * mat[1][0] + point.y * mat[1][1] + point.z * mat[1][2] + mat[1][3];
    out.z = point.x * mat[2][0] + point.y * mat[2][1] + point.z * mat[2][2] + mat[2][3];
    let w = point.x * mat[3][0] + point.y * mat[3][1] + point.z * mat[3][2] + mat[3][3];

    out.x /= w;
    out.y /= w;
    out.z /= w;

    out
}

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

    pub fn from_arr(x: [f64; 3]) -> Self {
        Self {
            x: x[0],
            y: x[1],
            z: x[2],
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

    pub fn into_point(&self) -> Point {
        Point { x: self.x, y: self.y, z: self.z }
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