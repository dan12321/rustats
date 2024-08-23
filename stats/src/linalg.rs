
use std::{cmp::min, error::Error, fmt::Display};

use anyhow::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    // Declaring minimum element traits needed for each operation gets verbose
    // and tedious. Starting with just f64.
    elements: Vec<f64>,
    width: usize,
    height: usize,
}

impl Matrix {
    pub fn new(elements: Vec<f64>, height: usize, width: usize) -> Result<Self> {
        if elements.len() != width * height {
            return Err(MatrixError::SizeMismatch.into());
        }

        Ok(Self {
            elements,
            height,
            width,
        })
    }

    pub fn mul(&self, matrix: &Matrix) -> Result<Matrix> {
        if self.width != matrix.height {
            return Err(MatrixError::SizeMismatch.into());
        }

        let mut elements: Vec<f64> = Vec::with_capacity(self.height * matrix.width);
        for ij in 0..self.height * matrix.width {
            let column = ij % matrix.width;
            let row = ij / matrix.width;
            let mut value = 0.0;
            for i in 0..self.width {
                let a = self.get_unchecked(row, i);
                let b = matrix.get_unchecked(i, column);
                value += a * b;
            }

            elements.push(value);
        }
        Ok(Matrix {
            elements,
            width: matrix.width,
            height: self.height,
        })
    }

    pub fn scalar_mul(&self, value: f64) -> Self {
        let elements: Vec<f64> = self.elements.iter()
            .map(|a| a * value)
            .collect();

        Self {
            elements,
            width: self.width,
            height: self.height,
        }
    }

    pub fn sub(&self, matrix: Self) -> Result<Self> {
        if self.width != matrix.width || self.height != matrix.height {
            return Err(MatrixError::SizeMismatch.into());
        }
        let elements = self.elements.iter()
            .zip(matrix.elements.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(Self {
            elements,
            width: self.width,
            height: self.height,
        })
    }

    pub fn add(&self, matrix: Self) -> Result<Self> {
        if self.width != matrix.width || self.height != matrix.height {
            return Err(MatrixError::SizeMismatch.into());
        }
        let elements = self.elements.iter()
            .zip(matrix.elements.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Self {
            elements,
            width: self.width,
            height: self.height,
        })
    }

    pub fn get_col(&self, i: usize) -> Result<Vec<f64>, MatrixError> {
        if i >= self.width {
            return Err(MatrixError::OutOfBounds);
        }
        let mut elements: Vec<f64> = Vec::with_capacity(self.height);
        for j in 0..self.height {
            elements.push(self.get_unchecked(j, i));
        }
        Ok(elements)
    }

    pub fn set_col(&mut self, i: usize, column: &Vec<f64>) -> Result<(), MatrixError> {
        if i >= self.width {
            return Err(MatrixError::OutOfBounds);
        }

        for j in 0..self.height {
            self.set_unchecked(j, i, column[j]);
        }

        Ok(())
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<(), MatrixError> {
        if row >= self.height || col >= self.width {
            Err(MatrixError::OutOfBounds)
        } else {
            self.set_unchecked(row, col, value);
            Ok(())
        }
    }

    pub fn set_unchecked(&mut self, row: usize, col: usize, value: f64) {
        self.elements[row * self.width + col] = value;
    }

    pub fn get_unchecked(&self, row: usize, col: usize) -> f64 {
        self.elements[row * self.width + col]
    }

    pub fn transpose(&self) -> Matrix {
        let mut elements: Vec<f64> = Vec::with_capacity(self.elements.len());
        for i in 0..self.width {
            for j in 0..self.height {
                elements.push(self.get_unchecked(j, i));
            }
        }
        Matrix {
            elements,
            width: self.height,
            height: self.width,
        }
    }

    pub fn round(&self, places: i32) -> Self {
        let shift = 10.0_f64.powi(places);
        let elements = self.elements.iter()
            .map(|a| (a * shift).round() / shift)
            .collect();
        Self {
            elements,
            width: self.width,
            height: self.height,
        }
    }

    pub fn mean_row(&self) -> Vec<f64> {
        let mut elements = vec![0.0;self.width];
        for i in 0..self.height {
            for j in 0..self.width {
                elements[j] += self.get_unchecked(i, j) / self.height as f64;
            }
        }
        elements
    }

    // TODO: make this a macro
    pub fn identity(width: usize, height: usize) -> Self {
        let mut elements = vec![0.0; width * height];
        let n = min(width, height);
        for i in 0..n {
            elements[i * width + i] = 1.0;
        }
        Self {
            elements,
            width,
            height,
        }
    }
}

struct SquareMatrix {
    matrix: Matrix,
    n: usize, 
}

impl SquareMatrix {
    pub fn qr_decomp(&self) -> QR {
        let q_elements = vec![0.0; self.n * self.n];
        let r_elements = vec![0.0; self.n * self.n];
        let mut q = Matrix::new(q_elements, self.n, self.n).unwrap();
        let mut r = Matrix::new(r_elements, self.n, self.n).unwrap();

        for i in 0..self.n {
            let ai = self.matrix.get_col(i).unwrap();
            let mut adjustments = vec![0.0; self.n];
            for j in 0..i {
                let qj = q.get_col(j).unwrap();
                let coef = dot(&ai,&qj);
                adjustments = add(&adjustments, &scalar_mul(coef, &qj));
                r.set(j, i, coef).unwrap();
            }
            let ai_perp = sub(&ai, &adjustments);
            let ai_perp_norm = norm(&ai_perp);
            r.set(i, i, ai_perp_norm).unwrap();
            q.set_col(i, &scalar_mul(1.0 / ai_perp_norm, &ai_perp)).unwrap();
        }
        QR {
            q,
            r,
        }
    }

    /// Compute the eigen values of a matrix using the qr algorithm.
    ///
    /// `threshold`: how close the lower triangle should be to 0.
    /// `max_iter`: maximum number of iterations
    pub fn eigen_values(&self, threshold: f64, max_iter: usize) -> Vec<f64> {
        let mut a = SquareMatrix {
            matrix: self.matrix.clone(),
            n: self.n,
        };
        for _ in 0..max_iter {
            let QR { q, r } = a.qr_decomp();
            a = r.mul(&q).unwrap().try_into().unwrap();
            if a.non_upper_triangle_within_threshold(threshold) {
                break;
            }
        }
        a.get_diagonal()
    }

    fn get_diagonal(&self) -> Vec<f64> {
        let mut elements = Vec::with_capacity(self.n);
        for i in 0..self.n {
            elements.push(self.matrix.get_unchecked(i, i));
        }
        elements
    }

    fn non_upper_triangle_within_threshold(&self, threshold: f64) -> bool {
        for i in 0..self.n {
            for j in i+1..self.n {
                if self.matrix.get_unchecked(j, i).abs() > threshold {
                    return false;
                }
            }
        }
        true
    }
}

impl TryFrom<Matrix> for SquareMatrix {
    type Error = MatrixError;
    fn try_from(value: Matrix) -> std::result::Result<Self, Self::Error> {
        if value.height != value.width {
            Err(MatrixError::NotSquare)
        } else {
            Ok(SquareMatrix {
                n: value.width,
                matrix: value,
            })
        }
    }
}

fn dot(u: &Vec<f64>, v: &Vec<f64>) -> f64 {
    u.iter()
        .zip(v.iter())
        .fold(0.0, |acc, (a, b)| acc + a * b)
}

fn norm(u: &Vec<f64>) -> f64 {
    u.iter()
        .fold(0.0, |acc, a| acc + a * a)
        .sqrt()
}

fn add(u: &Vec<f64>, v: &Vec<f64>) -> Vec<f64> {
    u.iter()
        .zip(v.iter())
        .map(|(a, b)| a + b)
        .collect()
}

fn sub(u: &Vec<f64>, v: &Vec<f64>) -> Vec<f64> {
    u.iter()
        .zip(v.iter())
        .map(|(a, b)| a - b)
        .collect()
}

fn scalar_mul(a: f64, u: &Vec<f64>) -> Vec<f64> {
    u.iter()
        .map(|b| a * b)
        .collect()
}

pub fn round(u: &Vec<f64>, places: i32) -> Vec<f64> {
    let shift = 10.0_f64.powi(places);
    let elements = u.iter()
        .map(|a| (a * shift).round() / shift)
        .collect();
    elements
}

pub struct QR {
    // Orthonormal
    q: Matrix,
    // Upper Triangular
    r: Matrix,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatrixError {
    SizeMismatch,
    OutOfBounds,
    NotSquare,
}

impl Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for MatrixError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_new_size_mismatch() {
        let a = Matrix::new(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 10.0,
            ],
            3,
            3,
        );

        let err: Option<MatrixError> = a.err().map(|e| e.downcast().unwrap());
        assert_eq!(err, Some(MatrixError::SizeMismatch));
    }

    #[test]
    fn test_matrix_identity() {
        let result = Matrix::identity(3, 3);
        let expected_result = Matrix::new(
            vec![
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
            3,
            3,
        ).unwrap();
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_multiply_square_matrix() {
        let a = Matrix::new(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
            3,
            3,
        ).unwrap();
        let b = Matrix::new(
            vec![
                10.0, 11.0, 12.0,
                13.0, 14.0, 15.0,
                16.0, 17.0, 18.0,
            ],
            3,
            3,
        ).unwrap();

        let result = a.mul(&b).unwrap();

        let expected_result = Matrix::new(
            vec![
                84.0, 90.0, 96.0,
                201.0, 216.0, 231.0,
                318.0, 342.0, 366.0,
            ],
            3,
            3,
        ).unwrap();
        assert_eq!(result, expected_result)
    }

    #[test]
    fn test_transpose() {
        let a = Matrix::new(
            vec![
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
            ],
            3,
            2,
        ).unwrap();

        let result = a.transpose();
        let expected_result = Matrix::new(
            vec![
                1.0, 3.0, 5.0,
                2.0, 4.0, 6.0,
            ],
            2,
            3,
        ).unwrap();

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_round() {
        let a = Matrix::new(
            vec![
                1.0001,
                2.0005,
            ],
            2,
            1,
        ).unwrap();

        let result = a.round(3);

        let expected = Matrix::new(
            vec![
                1.0,
                2.001,
            ],
            2,
            1
        ).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_qr_decomp() {
        let a: SquareMatrix = Matrix::new(
            vec![
                1.0, 2.0,
                3.0, 4.0,
            ],
            2,
            2,
        ).unwrap().try_into().unwrap();

        let result = a.qr_decomp();
        
        let expected_q = Matrix::new(vec![
            0.316, 0.949,
            0.949, -0.316
        ], 2, 2).unwrap();
        let expected_r = Matrix::new(vec![
            3.162, 4.427,
            0.0, 0.632,
        ], 2, 2).unwrap();

        // should probably check the delta is small but this gives better
        // error messages
        assert_eq!(result.q.round(3), expected_q);
        assert_eq!(result.r.round(3), expected_r);
    }

    #[test]
    fn test_eigen_values() {
        let a: SquareMatrix = Matrix::new(vec![
            1.0, 2.0,
            3.0, 4.0,
        ], 2, 2).unwrap().try_into().unwrap();

        let result = round(&a.eigen_values(0.01, 5), 3);

        let expected_result = vec![5.372, -0.372];
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_mean_row() {
        let a = Matrix::new(vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 9.0,
        ], 3, 2).unwrap();

        let result = a.mean_row();

        let expected_result = vec![3.0, 5.0];

        assert_eq!(result, expected_result);
    }
}