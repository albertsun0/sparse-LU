use crate::tests::test_utils::get_dense_simple;

use crate::sparse::{
    sparse_coo::SparseCOO, sparse_csc::SparseCSC, sparse_matrix::SparseMatrixTrait,
};

#[test]
fn test_coo_to_csc() {
    let dense_simple = get_dense_simple();
    let sparse_coo = SparseCOO::from_dense(dense_simple.clone());
    let sparse_csc = sparse_coo.to_csc();
    assert_eq!(sparse_csc.to_dense(), dense_simple);
}

#[test]
fn test_csc_to_coo() {
    let dense_simple = get_dense_simple();
    let sparse_csc = SparseCSC::from_dense(dense_simple.clone());
    let sparse_coo = sparse_csc.to_coo();
    assert_eq!(sparse_coo.to_dense(), dense_simple);
}

#[test]
fn stress_test_coo_to_csc() {
    let sparse_coo = SparseCOO::random(30, 30, 0.2);
    let sparse_csc = sparse_coo.to_csc();
    assert_eq!(sparse_csc.to_dense(), sparse_coo.to_dense());
}

#[test]
fn stress_test_csc_to_coo() {
    let sparse_csc = SparseCSC::random(30, 30, 0.2);
    let sparse_coo = sparse_csc.to_coo();
    assert_eq!(sparse_coo.to_dense(), sparse_csc.to_dense());
}
