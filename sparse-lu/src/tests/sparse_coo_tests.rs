use crate::sparse::{sparse_coo::SparseCOO, sparse_matrix::SparseMatrixTrait};
use crate::tests::test_utils::dense_matrix_multiply;

#[test]
fn test_sparse_coo_multiplication() {
    let dense_a = vec![
        vec![1.0, 2.0, 0.0],
        vec![0.0, 3.0, 4.0],
        vec![5.0, 0.0, 6.0],
    ];
    let dense_b = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];

    let sparse_a = SparseCOO::from_dense(dense_a.clone());
    let sparse_b = SparseCOO::from_dense(dense_b.clone());

    // Multiply sparse matrices
    let sparse_result = sparse_a.multiply(&sparse_b);

    // Multiply dense matrices for verification
    let dense_result = dense_matrix_multiply(&dense_a, &dense_b);

    println!("dense_result: {:?}", dense_result);
    // Convert sparse result to dense for comparison
    let sparse_result_dense = sparse_result.to_dense();

    // Compare results
    assert_eq!(sparse_result_dense, dense_result);

    // Verify dimensions
    assert_eq!(sparse_result.size(), (3, 2));
}

#[test]
fn test_sparse_coo_multiplication_edge_cases() {
    // Test multiplication with identity matrix
    let test_matrix = vec![
        vec![1.0, 2.0, 0.0],
        vec![0.0, 3.0, 4.0],
        vec![5.0, 0.0, 6.0],
    ];
    let identity = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    let sparse_test = SparseCOO::from_dense(test_matrix.clone());
    let sparse_identity = SparseCOO::from_dense(identity);

    let result = sparse_test.multiply(&sparse_identity);
    let result_dense = result.to_dense();

    // Result should equal the original matrix
    assert_eq!(result_dense, test_matrix);

    // Test multiplication with zero matrix
    let zero_matrix = vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]];

    let sparse_zero = SparseCOO::from_dense(zero_matrix);
    let result_zero = sparse_test.multiply(&sparse_zero);

    // Result should be all zeros
    let result_zero_dense = result_zero.to_dense();
    let expected_zero = vec![vec![0.0; 2]; 3];
    assert_eq!(result_zero_dense, expected_zero);
}

#[test]
fn test_sparse_coo_set_operations() {
    let mut sparse = SparseCOO::new(3, 3);

    // Test setting new elements
    sparse.set(0, 0, 1.0);
    sparse.set(1, 1, 2.0);
    sparse.set(2, 2, 3.0);

    // Test getting set elements
    assert_eq!(sparse.get(0, 0), 1.0);
    assert_eq!(sparse.get(1, 1), 2.0);
    assert_eq!(sparse.get(2, 2), 3.0);

    // Test getting unset elements (should return 0.0)
    assert_eq!(sparse.get(0, 1), 0.0);
    assert_eq!(sparse.get(1, 0), 0.0);

    // Test updating existing elements
    sparse.set(0, 0, 5.0);
    assert_eq!(sparse.get(0, 0), 5.0);

    // Test nnz count
    assert_eq!(sparse.nnz(), 3);
}
