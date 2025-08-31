use crate::sparse::{
    sparse_coo::SparseCOO, sparse_csc::SparseCSC, sparse_matrix::SparseMatrixTrait,
};

fn test_from_to_dense<T: SparseMatrixTrait>(dense: Vec<Vec<f32>>) {
    let sparse = T::from_dense(dense.clone());

    // Test size
    assert_eq!(sparse.size(), (dense.len(), dense[0].len()));
    assert_eq!(
        sparse.nnz(),
        dense.iter().flatten().filter(|x| **x != 0.0).count()
    );

    // Test conversion back to dense
    let converted_back = sparse.to_dense();
    assert_eq!(converted_back, dense);
}

fn test_sparse_get<T: SparseMatrixTrait>(dense: Vec<Vec<f32>>) {
    let sparse = T::from_dense(dense.clone());

    // Test size
    assert_eq!(sparse.size(), (dense.len(), dense[0].len()));
    assert_eq!(
        sparse.nnz(),
        dense.iter().flatten().filter(|x| **x != 0.0).count()
    );

    for i in 0..dense.len() {
        for j in 0..dense[0].len() {
            assert_eq!(sparse.get(i, j), dense[i][j]);
        }
    }
}

fn test_sparse_edge_cases<T: SparseMatrixTrait>() {
    // Test empty matrix
    let empty_dense = vec![vec![0.0; 3]; 3];
    let empty_sparse = T::from_dense(empty_dense);

    assert_eq!(empty_sparse.nnz(), 0);

    // Test matrix with all ones
    let ones_dense = vec![vec![1.0; 3]; 3];
    let ones_sparse = T::from_dense(ones_dense);

    assert_eq!(ones_sparse.nnz(), 9);

    // Test single element matrix
    let single_dense = vec![vec![5.0]];
    let single_sparse = T::from_dense(single_dense);

    assert_eq!(single_sparse.size(), (1, 1));
    assert_eq!(single_sparse.get(0, 0), 5.0);
}

fn test_sparse_random_generation<T: SparseMatrixTrait>(rows: usize, cols: usize, density: f32) {
    let random_sparse = T::random(rows, cols, density);

    // Test dimensions
    assert_eq!(random_sparse.size(), (rows, cols));

    // Test that density is approximately correct
    let expected_nnz = (rows * cols) as f32 * density;

    assert!(random_sparse.nnz() == expected_nnz as usize);

    // Test that random matrix can be converted to dense
    let sparse_dense = random_sparse.to_dense();

    assert_eq!(sparse_dense.len(), rows);
    assert_eq!(sparse_dense[0].len(), cols);
}

fn get_dense_simple() -> Vec<Vec<f32>> {
    vec![
        vec![1.0, 2.0, 0.0],
        vec![0.0, 3.0, 4.0],
        vec![5.0, 0.0, 6.0],
    ]
}

fn dense_random_floats(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut rng = fastrand::Rng::new();
    vec![vec![rng.f32(); cols]; rows]
}

#[test]
fn test_from_to_dense_coo() {
    let dense_simple = get_dense_simple();
    test_from_to_dense::<SparseCOO>(dense_simple);
    let dense_random = dense_random_floats(20, 18);
    test_from_to_dense::<SparseCOO>(dense_random);
}

#[test]
fn test_get_coo() {
    let dense_simple = get_dense_simple();
    test_sparse_get::<SparseCOO>(dense_simple);
    let dense_random = dense_random_floats(20, 18);
    test_sparse_get::<SparseCOO>(dense_random);
}

#[test]
fn test_edge_cases_coo() {
    test_sparse_edge_cases::<SparseCOO>();
}

#[test]
fn test_random_generation_coo() {
    test_sparse_random_generation::<SparseCOO>(10, 8, 0.3);
}

#[test]
fn test_from_to_dense_csc() {
    let dense_simple = get_dense_simple();
    test_from_to_dense::<SparseCSC>(dense_simple);
    let dense_random = dense_random_floats(20, 18);
    test_from_to_dense::<SparseCSC>(dense_random);
}

#[test]
fn test_get_csc() {
    let dense_simple = get_dense_simple();
    test_sparse_get::<SparseCSC>(dense_simple);
    let dense_random = dense_random_floats(20, 18);
    test_sparse_get::<SparseCSC>(dense_random);
}

#[test]
fn test_edge_cases_csc() {
    test_sparse_edge_cases::<SparseCSC>();
}

#[test]
fn test_random_generation_csc() {
    test_sparse_random_generation::<SparseCSC>(10, 8, 0.3);
}
