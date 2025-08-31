mod sparse;

use crate::sparse::sparse_coo::SparseCOO;
use crate::sparse::sparse_csc::SparseCSC;
use crate::sparse::sparse_csr::SparseCSR;
use crate::sparse::sparse_matrix::SparseMatrixTrait;
use std::time::Instant;

pub fn dense_matrix_multiply(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let rows_a = a.len();
    let cols_a = a[0].len();
    let cols_b = b[0].len();

    assert_eq!(
        cols_a,
        b.len(),
        "Matrix dimensions must be compatible for multiplication"
    );

    let mut result = vec![vec![0.0; cols_b]; rows_a];

    for i in 0..rows_a {
        for j in 0..cols_b {
            for k in 0..cols_a {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

fn print_dense(v: Vec<Vec<f32>>) {
    for row in v {
        for val in row {
            print!("{:.3}  ", val);
        }
        println!();
    }
}
fn main() {
    println!("Hello, world!");

    let n = 100000;
    let density = 0.0001;

    let mut start = Instant::now();
    let mat = SparseCOO::random(n, n, density);
    let mat2 = SparseCOO::random(n, n, density);
    println!(
        "Instantiated matrices in {:?} nnz {}",
        start.elapsed(),
        mat.nnz()
    );

    start = Instant::now();
    let result2 = mat.multiply(&mat2);
    println!(
        "Multiplied matrices COO in {:?} nnz {}",
        start.elapsed(),
        result2.nnz()
    );

    let mut start = Instant::now();
    let mat = SparseCSR::random(n, n, density);
    let mat2 = SparseCSR::random(n, n, density);
    println!(
        "CSR-CSR Instantiated matrices in {:?} nnz {}",
        start.elapsed(),
        mat.nnz()
    );

    start = Instant::now();
    let result2 = mat.multiply_to_flat_csr(&mat2);
    println!(
        "Multiplied matrices CSR-CSR in {:?} nnz {}",
        start.elapsed(),
        result2.0.len()
    );

    // start = Instant::now();
    // let result = mat.multiply(&mat2);
    // println!("Multiplied matrices in {:?}", start.elapsed());

    // let mut start = Instant::now();
    // let mut mat3 = SparseCSC::random(n, n, density);
    // println!("Instantiated matrices in {:?}", start.elapsed());

    // let mut mat4 = SparseCSC::random(5, 5, 1.0);
    // mat4.print();

    // print_dense(mat4.to_dense());
    // println!("{}", mat4.get(0, 0));
    // println!("{}", mat4.get(0, 1));
    // println!("{}", mat4.get(1, 0));
    // println!("{}", mat4.get(3, 3));

    let dense = vec![
        vec![1.0, 2.0, 0.0],
        vec![0.0, 3.0, 4.0],
        vec![5.0, 0.0, 6.0],
    ];

    let mut mat5 = SparseCSR::from_dense(dense.clone());
    let mat6 = SparseCSR::from_dense(dense.clone());

    let (indexes, values) = mat5.multiply_to_flat_csr(&mat6);
    let res = SparseCOO::from_flat_indices(3, 3, indexes, values);

    print_dense(res.to_dense());

    let dense_result = dense_matrix_multiply(&dense, &dense);

    print_dense(dense_result);
    mat5.print();
}
