/// Dense matrix multiplication for verification
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

pub fn get_dense_simple() -> Vec<Vec<f32>> {
    vec![
        vec![1.0, 2.0, 0.0],
        vec![0.0, 3.0, 4.0],
        vec![5.0, 0.0, 6.0],
    ]
}

pub fn get_dense_simple_b() -> Vec<Vec<f32>> {
    vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]
}

pub fn dense_random_floats(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut rng = fastrand::Rng::new();
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.f32()).collect())
        .collect()
}
