use crate::sparse::sparse_matrix::SparseMatrixTrait;
use rand::Rng;
use rand::seq::index::sample;
use std::collections::HashMap;
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

    pub fn to_dense(&self) -> Vec<Vec<f32>> {
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

    pub fn num_nnz_in_column(&self, j: usize) -> usize {
        self.colptr[j + 1] - self.colptr[j]
    }

    pub fn get_column_range(&self, j: usize) -> (usize, usize) {
        (self.colptr[j], self.colptr[j + 1])
    }

}


