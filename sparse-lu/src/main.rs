mod sparse;

use crate::sparse::sparse_coo::SparseCOO;
use crate::sparse::sparse_matrix::SparseMatrixTrait;

fn main() {
    println!("Hello, world!");

    let mut mat = SparseCOO::new(3, 3);
    mat.set(0, 0, 1.0);
    mat.set(0, 1, 2.0);
    mat.set(0, 2, 3.0);
    mat.set(1, 0, 4.0);
    mat.set(1, 1, 5.0);
    mat.set(1, 2, 6.0);
    mat.set(2, 0, 7.0);
    mat.set(2, 1, 8.0);
    mat.set(2, 2, 9.0);
    println!("{}", mat.get(0, 0));

    let mat2 = SparseCOO::from_dense(vec![
        vec![1.0, 0.0, 3.0],
        vec![4.0, 5.0, 0.0],
        vec![7.0, 0.0, 9.0],
    ]);
    let result = mat.multiply(&mat2);
    println!("{:?}", result.to_dense());
}
