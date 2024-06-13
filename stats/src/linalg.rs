#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<const ROWS: usize, const COLS: usize> {
    // Declaring minimum element traits needed for each operation gets verbose
    // and tedious. Starting with just f64.
    elements: Vec<f64>,
}

pub type Vector<const N: usize> = Matrix<N, 1>;

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    pub fn new(elements: Vec<f64>) -> Result<Self, MatrixError> {
        if elements.len() != ROWS * COLS {
            Err(MatrixError::SizeMismatch)
        } else {
            Ok(Self { elements })
        }
    }

    pub fn mul<const M: usize>(&self, matrix: &Matrix<COLS, M>) -> Matrix<ROWS, M> {
        let mut elements: Vec<f64> = Vec::with_capacity(ROWS * M);
        for ij in 0..ROWS * M {
            let column = ij % M;
            let row = ij / M;
            let mut value = 0.0;
            for i in 0..COLS {
                let a = self.get_unchecked(row, i);
                let b = matrix.get_unchecked(i, column);
                value += a * b;
            }

            elements.push(value);
        }
        Matrix::<ROWS, M> {
            elements,
        }
    }

    pub fn scalar_mul(&self, value: f64) -> Self {
        let elements: Vec<f64> = self.elements.iter()
            .map(|a| a * value)
            .collect();

        Self {
            elements,
        }
    }

    pub fn sub(&self, matrix: Self) -> Self {
        let elements = self.elements.iter()
            .zip(matrix.elements.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self {
            elements,
        }
    }

    pub fn add(&self, matrix: Self) -> Self {
        let elements = self.elements.iter()
            .zip(matrix.elements.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self {
            elements,
        }
    }

    pub fn get_col(&self, i: usize) -> Result<Vector<ROWS>, MatrixError> {
        if i >= COLS {
            return Err(MatrixError::OutOfBounds);
        }
        let mut elements: Vec<f64> = Vec::with_capacity(ROWS);
        for j in 0..ROWS {
            elements.push(self.get_unchecked(j, i));
        }
        Ok(Vector::<ROWS> {
            elements
        })
    }

    pub fn set_col(&mut self, i: usize, column: &Vector<ROWS>) -> Result<(), MatrixError> {
        if i >= COLS {
            return Err(MatrixError::OutOfBounds);
        }

        for j in 0..ROWS {
            self.set_unchecked(j, i, column.get_unchecked(j, 0));
        }

        Ok(())
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<(), MatrixError> {
        if row >= ROWS || col >= COLS {
            Err(MatrixError::OutOfBounds)
        } else {
            self.set_unchecked(row, col, value);
            Ok(())
        }
    }

    pub fn set_unchecked(&mut self, row: usize, col: usize, value: f64) {
        self.elements[row * COLS + col] = value;
    }

    pub fn get_unchecked(&self, row: usize, col: usize) -> f64 {
        self.elements[row * COLS + col]
    }

    pub fn transpose(&self) -> Matrix<COLS, ROWS> {
        let mut elements: Vec<f64> = Vec::with_capacity(ROWS * COLS);
        for i in 0..COLS {
            for j in 0..ROWS {
                elements.push(self.get_unchecked(j, i));
            }
        }
        Matrix::<COLS, ROWS> {
            elements,
        }
    }

    pub fn round(&self, places: i32) -> Self {
        let shift = 10.0_f64.powi(places);
        let elements = self.elements.iter()
            .map(|a| (a * shift).round() / shift)
            .collect();
        Self {
            elements
        }
    }
}

impl<const N: usize> Matrix<N, N> {
    // TODO: make this a macro
    pub fn identity() -> Self {
        let mut elements = vec![0.0; N * N];
        for i in 0..N {
            elements[i * N + i] = 1.0;
        }
        Self {
            elements,
        }
    }

    pub fn qr_decomp(&self) -> QR<N> {
        let q_elements = vec![0.0; N * N];
        let r_elements = vec![0.0; N * N];
        let mut q = Self::new(q_elements).unwrap();
        let mut r = Self::new(r_elements).unwrap();

        for i in 0..N {
            let ai = self.get_col(i).unwrap();
            let mut adjustments = Vector::<N>::new(vec![0.0; N]).unwrap();
            for j in 0..i {
                let qj = q.get_col(j).unwrap();
                let coef = ai.dot(&qj);
                adjustments = adjustments.add(qj.scalar_mul(coef));
                r.set(j, i, coef).unwrap();
            }
            let ai_perp = ai.sub(adjustments);
            let ai_perp_norm = ai_perp.norm();
            r.set(i, i, ai_perp_norm).unwrap();
            q.set_col(i, &ai_perp.scalar_mul(1.0 / ai_perp_norm)).unwrap();
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
    pub fn eigen_values(&self, threshold: f64, max_iter: usize) -> Vector<N> {
        let mut a = self.clone();
        for _ in 0..max_iter {
            let QR { q, r } = a.qr_decomp();
            a = r.mul(&q);
            if a.non_upper_triangle_within_threshold(threshold) {
                break;
            }
        }
        a.get_diagonal()
    }

    fn get_diagonal(&self) -> Vector<N> {
        let mut elements = Vec::with_capacity(N);
        for i in 0..N {
            elements.push(self.get_unchecked(i, i));
        }
        Vector::<N> {
            elements,
        }
    }

    fn non_upper_triangle_within_threshold(&self, threshold: f64) -> bool {
        for i in 0..N {
            for j in i+1..N {
                if self.get_unchecked(j, i).abs() > threshold {
                    return false;
                }
            }
        }
        true
    }
}

impl<const N: usize> Vector<N> {
    pub fn dot(&self, vector: &Vector<N>) -> f64 {
        self.elements.iter()
            .zip(vector.elements.iter())
            .fold(0.0, |acc, (a, b)| acc + a * b)
    }

    pub fn norm(&self) -> f64 {
        self.elements.iter()
            .fold(0.0, |acc, a| acc + a * a)
            .sqrt()
    }
}

pub struct QR<const N: usize> {
    // Orthonormal
    q: Matrix<N, N>,
    // Upper Triangular
    r: Matrix<N, N>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatrixError {
    SizeMismatch,
    OutOfBounds,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_new_size_mismatch() {
        let a = Matrix::<3, 3>::new(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 10.0,
            ],
        );

        assert_eq!(a.err(), Some(MatrixError::SizeMismatch));
    }

    #[test]
    fn test_matrix_identity() {
        let result = Matrix::<3, 3>::identity();
        let expected_result = Matrix::<3, 3>::new(vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]).unwrap();
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_multiply_square_matrix() {
        let a = Matrix::<3, 3>::new(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
        ).unwrap();
        let b = Matrix::<3, 3>::new(
            vec![
                10.0, 11.0, 12.0,
                13.0, 14.0, 15.0,
                16.0, 17.0, 18.0,
            ],
        ).unwrap();

        let result = a.mul(&b);

        let expected_result = Matrix::<3, 3>::new(vec![
            84.0, 90.0, 96.0,
            201.0, 216.0, 231.0,
            318.0, 342.0, 366.0,
        ]).unwrap();
        assert_eq!(result, expected_result)
    }

    #[test]
    fn test_transpose() {
        let a = Matrix::<3,2>::new(vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]).unwrap();

        let result = a.transpose();
        let expected_result = Matrix::<2,3>::new(vec![
            1.0, 3.0, 5.0,
            2.0, 4.0, 6.0,
        ]).unwrap();

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_round() {
        let a = Matrix::<2, 1>::new(vec![
            1.0001,
            2.0005,
        ]).unwrap();

        let result = a.round(3);

        let expected = Matrix::<2, 1>::new(vec![
            1.0,
            2.001,
        ]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_qr_decomp() {
        let a = Matrix::<2,2>::new(vec![
            1.0, 2.0,
            3.0, 4.0,
        ]).unwrap();

        let result = a.qr_decomp();
        
        let expected_q = Matrix::<2,2>::new(vec![
            0.316, 0.949,
            0.949, -0.316
        ]).unwrap();
        let expected_r = Matrix::<2,2>::new(vec![
            3.162, 4.427,
            0.0, 0.632,
        ]).unwrap();

        // should probably check the delta is small but this gives better
        // error messages
        assert_eq!(result.q.round(3), expected_q);
        assert_eq!(result.r.round(3), expected_r);
    }

    #[test]
    fn test_eigen_values() {
        let a = Matrix::<2,2>::new(vec![
            1.0, 2.0,
            3.0, 4.0,
        ]).unwrap();

        let result = a.eigen_values(0.01, 5).round(3);

        let expected_result = vec![5.372, -0.372];
        assert_eq!(result.elements, expected_result);
    }
}