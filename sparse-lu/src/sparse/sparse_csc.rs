use crate::sparse::sparse_coo::SparseCOO;
use crate::sparse::sparse_matrix::SparseMatrixTrait;
use std::collections::HashSet;

/*
    Compressed sparse column

    colptr[j] = index of first element in rowind for column j
    rowind[i] = row index
    values[i] = value

    to get number of nonzeros in column j, use colptr[j+1] - colptr[j]
*/

pub struct SparseCSC {
    pub nrows: usize,
    pub ncols: usize,
    pub colptr: Vec<usize>, // length = ncols + 1
    pub rowind: Vec<usize>, // length = nnz
    pub values: Vec<f32>,   // length = nnz
}

impl SparseMatrixTrait for SparseCSC {
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
        self.rowind.len()
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
            colptr: vec![0; ncols + 1],
            rowind: Vec::new(),
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

        let mut colptr = vec![0; ncols + 1];
        colptr[ncols] = nnz;
        let mut rowind = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        x.sort_unstable();

        let mut current_col: usize = 0;
        for flat_index in x {
            let i = flat_index % nrows;
            let j = flat_index / nrows;
            while j > current_col {
                colptr[current_col + 1] = rowind.len();
                current_col += 1;
            }
            rowind.push(i);
            values.push(rng.f32());
        }

        while current_col < ncols {
            colptr[current_col + 1] = rowind.len();
            current_col += 1;
        }

        Self {
            nrows,
            ncols,
            colptr,
            rowind,
            values,
        }
    }
    fn from_dense(dense: Vec<Vec<f32>>) -> Self {
        let nrows = dense.len();
        let ncols = dense[0].len();
        let mut colptr = vec![0; ncols + 1];
        let mut rowind = Vec::new();
        let mut values = Vec::new();

        for i in 0..ncols {
            for j in 0..nrows {
                let value = dense[j][i];
                if value != 0.0 {
                    rowind.push(j);
                    values.push(value);
                }
            }
            colptr[i + 1] = rowind.len();
        }

        Self {
            nrows,
            ncols,
            colptr,
            rowind,
            values,
        }
    }
    fn to_dense(&self) -> Vec<Vec<f32>> {
        let mut dense = vec![vec![0.0; self.ncols]; self.nrows];
        let mut col = 0;
        for i in 0..self.nnz() {
            if i >= self.colptr[col + 1] {
                col += 1;
            }
            dense[self.rowind[i]][col] = self.values[i];
        }
        dense
    }
}

impl SparseCSC {
    fn get_container_index(&self, i: usize, j: usize) -> Option<usize> {
        // binary search column
        let start = self.colptr[j];
        let end = self.colptr[j + 1];

        match self.rowind[start..end].binary_search(&i) {
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
        println!("SparseCSC matrix:");
        println!("nrows: {}", self.nrows);
        println!("ncols: {}", self.ncols);
        println!("colptr: {:?}", self.colptr);
        println!("rowind: {:?}", self.rowind);
        println!("values: {:?}", self.values);
    }

    pub fn num_nnz_in_column(&self, j: usize) -> usize {
        self.colptr[j + 1] - self.colptr[j]
    }

    pub fn get_column_range(&self, j: usize) -> (usize, usize) {
        (self.colptr[j], self.colptr[j + 1])
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
            .map(|x| flat_index_to_column_major(*x, nrows, ncols))
            .zip(flat_values.iter())
            .collect::<Vec<_>>();

        flat_pairs.sort_unstable_by_key(|pair| pair.0);

        let (sorted_flat_indices, sorted_flat_values): (Vec<usize>, Vec<f32>) =
            flat_pairs.into_iter().unzip();

        let nnz = sorted_flat_indices.len();

        let mut colptr = vec![0; ncols + 1];
        colptr[ncols] = nnz;
        let mut rowind = Vec::with_capacity(nnz);

        let mut current_col: usize = 0;

        for flat_index in sorted_flat_indices {
            let row = flat_index % nrows;
            let col = flat_index / nrows;
            while col > current_col {
                colptr[current_col + 1] = rowind.len();
                current_col += 1;
            }
            rowind.push(row);
        }

        while current_col < ncols {
            colptr[current_col + 1] = rowind.len();
            current_col += 1;
        }

        Self {
            nrows,
            ncols,
            colptr,
            rowind,
            values: sorted_flat_values,
        }
    }

    pub fn to_coo(&self) -> SparseCOO {
        let mut col = 0;
        let mut flat_indices = Vec::with_capacity(self.nnz());

        for i in 0..self.nnz() {
            if i >= self.colptr[col + 1] {
                col += 1;
            }
            flat_indices.push(self.rowind[i] * self.ncols + col);
        }

        SparseCOO::from_flat_indices(self.nrows, self.ncols, flat_indices, self.values.clone())
    }

    pub fn nonzero_columns(&self) -> Vec<usize> {
        let mut nonzero_columns = Vec::new();
        for i in 0..self.ncols {
            if self.num_nnz_in_column(i) > 0 {
                nonzero_columns.push(i);
            }
        }
        nonzero_columns
    }
}

pub fn flat_index_to_column_major(flat_index: usize, nrows: usize, ncols: usize) -> usize {
    /*
       turn a row major flat_index = row * ncols + col
       into column major flat_index = col * nrows + row
    */

    let row = flat_index / ncols;
    let col = flat_index % ncols;
    return col * nrows + row;
}
