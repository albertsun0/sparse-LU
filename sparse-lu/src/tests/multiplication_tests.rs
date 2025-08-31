use crate::sparse::{
    sparse_csc::SparseCSC, sparse_csr::SparseCSR, sparse_matrix::SparseMatrixTrait,
};
use crate::tests::test_utils::{
    dense_matrix_multiply, dense_random_floats, get_dense_simple, get_dense_simple_b,
};

fn test_sparse_csr_csc_multiplication(dense_a: Vec<Vec<f32>>, dense_b: Vec<Vec<f32>>) {
    let sparse_a = SparseCSR::from_dense(dense_a.clone());
    let sparse_b = SparseCSR::from_dense(dense_b.clone());

    // Multiply sparse matrices
    let sparse_result = sparse_a.multiply_csr(&sparse_b);

    // Multiply dense matrices for verification
    let dense_result = dense_matrix_multiply(&dense_a, &dense_b);

    println!("dense_result: {:?}", dense_result);
    // Convert sparse result to dense for comparison
    let sparse_result_dense = sparse_result.to_dense();

    println!("sparse_result_dense: {:?}", sparse_result_dense);
    // Compare results
    assert_eq!(sparse_result_dense, dense_result);
}

#[test]
fn test_sparse_csr_csc_multiplication_simple() {
    let dense_a = get_dense_simple();
    let dense_b = get_dense_simple_b();
    test_sparse_csr_csc_multiplication(dense_a, dense_b);
}

#[test]
fn test_sparse_csr_csc_multiplication_random() {
    let dense_a = dense_random_floats(20, 16);
    let dense_b = dense_random_floats(16, 12);
    test_sparse_csr_csc_multiplication(dense_a, dense_b);
}
