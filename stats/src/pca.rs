use crate::linalg::{matrix_rows, Matrix, SquareMatrix};

use anyhow::Result;

pub fn pca(data: Matrix) -> Result<Matrix> {
    let mean = matrix_rows(&data.mean_row(), data.height());
    let centered_data = data.sub(mean).unwrap();
    let cov = centered_data.transpose().mul(&centered_data).unwrap();
    let cov: SquareMatrix = cov.try_into().unwrap();
    let eigen_vectors = cov.eigen_vectors(0.001, 10, 100);
    let pca = centered_data.mul(&eigen_vectors.into()).unwrap();
    Ok(pca)
}