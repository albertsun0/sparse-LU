mod sparse;

use crate::sparse::sparse_coo::SparseCOO;
use crate::sparse::sparse_matrix::SparseMatrixTrait;
use std::time::Instant;


fn main() {
    println!("Hello, world!");

    let n = 100000;
    let density = 0.0001;
    
    let mut start = Instant::now();
    let mat = SparseCOO::random(n, n, density);
    let mat2 = SparseCOO::random(n, n, density);
    println!("Instantiated matrices in {:?}", start.elapsed());
    
    // start = Instant::now();
    // let result2 = mat.multiply_unoptimized(&mat2);
    // println!("Multiplied matrices unoptimized in {:?}", start.elapsed());

    start = Instant::now();
    let result = mat.multiply(&mat2);
    println!("Multiplied matrices in {:?}", start.elapsed());
    
    // println!("{:?}", result.to_dense());

    // let mat2 = SparseCOO::random(3, 3, 0.5);
    // let result = mat.multiply(&mat2);
    // println!("{:?}", result.to_dense());
    // let result2 = mat.multiply_unoptimized(&mat2);
    // println!("{:?}", result2.to_dense());
}
