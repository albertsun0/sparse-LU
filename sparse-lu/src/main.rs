mod sparse;

use crate::sparse::sparse_coo::SparseCOO;
use crate::sparse::sparse_csc::SparseCSC;
use crate::sparse::sparse_matrix::SparseMatrixTrait;
use std::time::Instant;

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

    // let mut start = Instant::now();
    // let mat = SparseCOO::random(n, n, density);
    // let mat2 = SparseCOO::random(n, n, density);
    // println!(
    //     "Instantiated matrices in {:?} nnz {}",
    //     start.elapsed(),
    //     mat.nnz()
    // );

    // start = Instant::now();
    // let result2 = mat.multiply(&mat2);
    // println!(
    //     "Multiplied matrices COO in {:?} nnz {}",
    //     start.elapsed(),
    //     result2.nnz()
    // );

    // start = Instant::now();
    // let result = mat.multiply(&mat2);
    // println!("Multiplied matrices in {:?}", start.elapsed());
    let mut start = Instant::now();
    let mut mat3 = SparseCSC::random(n, n, density);
    println!("Instantiated matrices in {:?}", start.elapsed());

    let mut mat4 = SparseCSC::random(5, 5, 1.0);
    mat4.print();

    print_dense(mat4.to_dense());
    println!("{}", mat4.get(0, 0));
    println!("{}", mat4.get(0, 1));
    println!("{}", mat4.get(1, 0));
    println!("{}", mat4.get(3, 3));
}
