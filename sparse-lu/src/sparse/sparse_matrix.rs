// too lazy to support generic types
pub trait SparseMatrixTrait {
    fn get(&self, i: usize, j: usize) -> f32;
    fn size(&self) -> (usize, usize);
    fn nnz(&self) -> usize;
    fn set(&mut self, i: usize, j: usize, value: f32);
    fn new(rows: usize, cols: usize) -> Self;
    fn random(rows: usize, cols: usize, density: f32) -> Self;
}

//TODO: create sparseMatrix class that handles conversion between different sparse matrix formats
