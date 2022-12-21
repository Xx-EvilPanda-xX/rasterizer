use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut};
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

    out[0][0] = row_col_mul(m1, m2, 0, 0);
    out[1][0] = row_col_mul(m1, m2, 1, 0);
    out[2][0] = row_col_mul(m1, m2, 2, 0);
    out[3][0] = row_col_mul(m1, m2, 3, 0);

    out[0][1] = row_col_mul(m1, m2, 0, 1);
    out[1][1] = row_col_mul(m1, m2, 1, 1);
    out[2][1] = row_col_mul(m1, m2, 2, 1);
    out[3][1] = row_col_mul(m1, m2, 3, 1);

    out[0][2] = row_col_mul(m1, m2, 0, 2);
    out[1][2] = row_col_mul(m1, m2, 1, 2);
    out[2][2] = row_col_mul(m1, m2, 2, 2);
    out[3][2] = row_col_mul(m1, m2, 3, 2);

    out[0][3] = row_col_mul(m1, m2, 0, 3);
    out[1][3] = row_col_mul(m1, m2, 1, 3);
    out[2][3] = row_col_mul(m1, m2, 2, 3);
    out[3][3] = row_col_mul(m1, m2, 3, 3);

    out
}

fn row_col_mul(m1: &Mat4f, m2: &Mat4f, row: usize, col: usize) -> f64 {
    let mut out = 0.0;

    for i in 0..4 {
        out += m1[row][i] * m2[i][col]
    }

    out
}

#[derive(Debug, PartialEq)]
pub struct Mat4f {
    pub mat: [[f64; 4]; 4],
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