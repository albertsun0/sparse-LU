use crate::sparse::sparse_coo::SparseCOO;
use crate::sparse::sparse_csc::SparseCSC;
use crate::sparse::sparse_matrix::SparseMatrixTrait;
use std::collections::HashSet;

/*
    Compressed sparse row

    rowptr[i] = index of first element in colind for row i
    colind[i] = column index
    values[i] = value

    to get number of nonzeros in row i, use rowptr[i+1] - rowptr[i]

    NOTE: not needed for LU, but CSR x CSC is the most efficent multiplication format
*/

pub struct SparseCSR {
    pub nrows: usize,
    pub ncols: usize,
    pub rowptr: Vec<usize>, // length = nrows + 1
    pub colind: Vec<usize>, // length = nnz
    pub values: Vec<f32>,   // length = nnz
}

impl SparseMatrixTrait for SparseCSR {
    fn get(&self, i: usize, j: usize) -> f32 {
        self.check_bounds(i, j);
        let index = self.get_container_index(i, j);
        match index {
            Some(index) => self.values[index],
            None => 0.0,
        }
    }
    fn size(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
    fn nnz(&self) -> usize {
        self.colind.len()
    }
    fn set(&mut self, i: usize, j: usize, value: f32) {
        self.check_bounds(i, j);
        let index = self.get_container_index(i, j);
        match index {
            Some(index) => {
                self.values[index] = value;
            }
            None => {
                println!("inserting new indices not supported: ({}, {})", i, j);
            }
        }
    }
    // this fn is not very useful, since we disallow setting new indices
    fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            rowptr: vec![0; nrows + 1],
            colind: Vec::new(),
            values: Vec::new(),
        }
    }
    fn random(nrows: usize, ncols: usize, density: f32) -> Self {
        let mut rng = fastrand::Rng::new();
        let nnz = ((nrows * ncols) as f32 * density).floor() as usize;

        let mut flat_indices: HashSet<usize> = HashSet::with_capacity(nnz);
        let mut x = Vec::with_capacity(nnz);

        while flat_indices.len() < nnz {
            let index = rng.usize(..nrows * ncols);
            if flat_indices.insert(index) {
                x.push(index);
            }
        }

        let mut rowptr = vec![0; nrows + 1];
        rowptr[nrows] = nnz;
        let mut colind = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        x.sort_unstable();

        let mut current_row: usize = 0;
        for flat_index in x {
            let row = flat_index / ncols;
            let col = flat_index % ncols;
            while row > current_row {
                rowptr[current_row + 1] = colind.len();
                current_row += 1;
            }
            colind.push(col);
            values.push(rng.f32());
        }

        while current_row < nrows {
            rowptr[current_row + 1] = colind.len();
            current_row += 1;
        }

        Self {
            nrows,
            ncols,
            rowptr,
            colind,
            values,
        }
    }
    fn from_dense(dense: Vec<Vec<f32>>) -> Self {
        let nrows = dense.len();
        let ncols = dense[0].len();
        let mut rowptr = vec![0; nrows + 1];
        let mut colind = Vec::new();
        let mut values = Vec::new();

        for i in 0..nrows {
            for j in 0..ncols {
                let value = dense[i][j];
                if value != 0.0 {
                    colind.push(j);
                    values.push(value);
                }
            }
            rowptr[i + 1] = colind.len();
        }

        Self {
            nrows,
            ncols,
            rowptr,
            colind,
            values,
        }
    }
    fn to_dense(&self) -> Vec<Vec<f32>> {
        let mut dense = vec![vec![0.0; self.ncols]; self.nrows];
        let mut row = 0;
        for i in 0..self.nnz() {
            if i >= self.rowptr[row + 1] {
                row += 1;
            }
            dense[row][self.colind[i]] = self.values[i];
        }
        dense
    }
}

impl SparseCSR {
    fn get_container_index(&self, i: usize, j: usize) -> Option<usize> {
        // binary search column
        let start = self.rowptr[i];
        let end = self.rowptr[i + 1];

        match self.colind[start..end].binary_search(&j) {
            Ok(index) => Some(start + index),
            Err(_) => None,
        }
    }

    fn check_bounds(&self, i: usize, j: usize) -> bool {
        if i >= self.nrows || j >= self.ncols {
            panic!("Index out of bounds: ({}, {})", i, j);
        }
        true
    }

    pub fn print(&self) {
        println!("SparseCSR matrix:");
        println!("nrows: {}", self.nrows);
        println!("ncols: {}", self.ncols);
        println!("rowptr: {:?}", self.rowptr);
        println!("colind: {:?}", self.colind);
        println!("values: {:?}", self.values);
    }

    pub fn num_nnz_in_row(&self, i: usize) -> usize {
        self.rowptr[i + 1] - self.rowptr[i]
    }

    pub fn get_row_range(&self, i: usize) -> (usize, usize) {
        (self.rowptr[i], self.rowptr[i + 1])
    }

    // TODO: take optional flat_values
    pub fn from_flat_indices(
        nrows: usize,
        ncols: usize,
        flat_indices: Vec<usize>,
        flat_values: Vec<f32>,
    ) -> Self {
        assert_eq!(flat_indices.len(), flat_values.len());

        let mut flat_pairs = flat_indices
            .iter()
            .zip(flat_values.iter())
            .collect::<Vec<_>>();

        flat_pairs.sort_unstable_by_key(|pair| pair.0);

        let (sorted_flat_indices, sorted_flat_values): (Vec<usize>, Vec<f32>) =
            flat_pairs.into_iter().unzip();

        let nnz = sorted_flat_indices.len();

        let mut rowptr = vec![0; nrows + 1];
        rowptr[nrows] = nnz;
        let mut colind = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        let mut current_row: usize = 0;
        for flat_index in sorted_flat_indices {
            let row = flat_index / ncols;
            let col = flat_index % ncols;
            while row > current_row {
                rowptr[current_row + 1] = colind.len();
                current_row += 1;
            }
            colind.push(col);
            values.push(sorted_flat_values[flat_index]);
        }

        while current_row < nrows {
            rowptr[current_row + 1] = colind.len();
            current_row += 1;
        }

        Self {
            nrows,
            ncols,
            rowptr,
            colind,
            values,
        }
    }

    pub fn to_flat_indices(&self) -> (Vec<usize>, Vec<f32>) {
        let mut row = 0;
        let mut flat_indices = Vec::with_capacity(self.nnz());

        for i in 0..self.nnz() {
            if i >= self.rowptr[row + 1] {
                row += 1;
            }
            flat_indices.push(row * self.ncols + self.colind[i]);
        }

        (flat_indices, self.values.clone())
    }

    pub fn to_coo(&self) -> SparseCOO {
        let (flat_indices, values) = self.to_flat_indices();

        SparseCOO::from_flat_indices(self.nrows, self.ncols, flat_indices, values)
    }

    pub fn multiply_to_flat_csc(&self, other: &SparseCSC) -> (Vec<usize>, Vec<f32>) {
        assert_eq!(self.ncols, other.nrows);
        let target_cols = other.ncols;

        let mut result_flat_indices = Vec::new();
        let mut result_values = Vec::new();

        for i in 0..self.nrows {
            let (r_start, r_end) = self.get_row_range(i);
            if r_start == r_end {
                continue;
            }
            for j in 0..other.ncols {
                let (c_start, c_end) = other.get_column_range(j);
                let mut acc = 0.0;

                // two pointers - rowrange and colrange are sorted
                let mut r_ptr = r_start;
                let mut c_ptr = c_start;

                while r_ptr < r_end && c_ptr < c_end {
                    let rowind = other.rowind[c_ptr];
                    let colind = self.colind[r_ptr];
                    if rowind == colind {
                        acc += self.values[r_ptr] * other.values[c_ptr];
                        r_ptr += 1;
                        c_ptr += 1;
                    } else if rowind < colind {
                        c_ptr += 1;
                    } else {
                        r_ptr += 1;
                    }
                }
                result_flat_indices.push(i * target_cols + j);
                result_values.push(acc);
            }
        }

        (result_flat_indices, result_values)
    }
    pub fn multiply_csc(&self, other: &SparseCSC) -> SparseCSC {
        let (flat_indices, values) = self.multiply_to_flat_csc(other);
        SparseCSC::from_flat_indices(self.nrows, other.ncols, flat_indices, values)
    }
}
