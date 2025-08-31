use crate::sparse::{sparse_csc::SparseCSC, sparse_matrix::SparseMatrixTrait};

#[test]
fn test_sparse_csc_column_operations() {
    let dense = vec![
        vec![1.0, 2.0, 0.0],
        vec![0.0, 3.0, 4.0],
        vec![5.0, 0.0, 6.0],
    ];
    let sparse = SparseCSC::from_dense(dense);

    // Test column operations
    assert_eq!(sparse.num_nnz_in_column(0), 2); // Column 0 has 2 non-zeros
    assert_eq!(sparse.num_nnz_in_column(1), 2); // Column 1 has 2 non-zeros
    assert_eq!(sparse.num_nnz_in_column(2), 2); // Column 2 has 2 non-zeros

    // Test column range
    let (start, end) = sparse.get_column_range(0);
    assert_eq!(end - start, 2);

    let (start, end) = sparse.get_column_range(1);
    assert_eq!(end - start, 2);

    let (start, end) = sparse.get_column_range(2);
    assert_eq!(end - start, 2);
}
