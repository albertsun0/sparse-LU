use crate::sparse::{sparse_csc::SparseCSC, sparse_matrix::SparseMatrixTrait};
use rand::seq::index::sample;
use std::{collections::HashMap, iter::repeat_with};

pub struct SparseCOO {
    pub nrows: usize,
    pub ncols: usize,
    pub rowind: Vec<usize>, // length = nnz
    pub colind: Vec<usize>, // length = nnz
    pub values: Vec<f32>,   // length = nnz
}

impl SparseMatrixTrait for SparseCOO {
    // WARNING: get is O(n)
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
    // WARNING: set is O(n)
    fn set(&mut self, i: usize, j: usize, value: f32) {
        self.check_bounds(i, j);
        let index = self.get_container_index(i, j);
        match index {
            Some(index) => self.values[index] = value,
            None => {
                self.rowind.push(i);
                self.colind.push(j);
                self.values.push(value);
            }
        }
    }
    fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            rowind: Vec::new(),
            colind: Vec::new(),
            values: Vec::new(),
        }
    }
    fn random(nrows: usize, ncols: usize, density: f32) -> Self {
        let mut rng = fastrand::Rng::new();
        let mut old_rng = rand::rng();
        let nnz = ((nrows * ncols) as f32 * density).floor() as usize;

        // better way to sample using fastrand?
        let flat_indices = sample(&mut old_rng, nrows * ncols, nnz);

        let rowind: Vec<_> = flat_indices.iter().map(|x| x % nrows).collect();
        let colind: Vec<_> = flat_indices.iter().map(|x| x / nrows).collect();
        let values: Vec<f32> = repeat_with(|| rng.f32()).take(nnz).collect();

        Self {
            nrows,
            ncols,
            rowind,
            colind,
            values,
        }
    }
    fn from_dense(dense: Vec<Vec<f32>>) -> Self {
        let nrows = dense.len();
        let ncols = dense[0].len();
        let mut rowind = Vec::new();
        let mut colind = Vec::new();
        let mut values = Vec::new();
        for (i, row) in dense.iter().enumerate().take(nrows) {
            for (j, value) in row.iter().enumerate().take(ncols) {
                if *value != 0.0 {
                    rowind.push(i);
                    colind.push(j);
                    values.push(*value);
                }
            }
        }
        Self {
            nrows,
            ncols,
            rowind,
            colind,
            values,
        }
    }

    fn to_dense(&self) -> Vec<Vec<f32>> {
        let mut dense = vec![vec![0.0; self.ncols]; self.nrows];
        for i in 0..self.nnz() {
            dense[self.rowind[i]][self.colind[i]] = self.values[i];
        }
        dense
    }
}

impl SparseCOO {
    fn get_container_index(&self, i: usize, j: usize) -> Option<usize> {
        self.check_bounds(i, j);
        self.rowind
            .iter()
            .zip(self.colind.iter())
            .position(|(x, y)| *x == i && *y == j)
            .map(|x| x)
    }

    fn check_bounds(&self, i: usize, j: usize) -> bool {
        if i >= self.nrows || j >= self.ncols {
            panic!("Index out of bounds: ({}, {})", i, j);
        }
        true
    }

    pub fn print(&self) {
        println!("SparseCOO matrix:");
        println!("nrows: {}", self.nrows);
        println!("ncols: {}", self.ncols);
        println!("rowind: {:?}", self.rowind);
        println!("colind: {:?}", self.colind);
        println!("values: {:?}", self.values);
    }

    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.ncols, other.nrows);
        let target_cols = other.ncols;
        self.print();
        other.print();
        let mut other_row_map: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();

        for i in 0..other.nnz() {
            other_row_map
                .entry(other.rowind[i])
                .or_default()
                .push((other.colind[i], other.values[i]));
        }

        let mut result_map: HashMap<usize, f32> = HashMap::new();

        for i in 0..self.nnz() {
            let row = self.rowind[i];
            let col = self.colind[i];
            let value = self.values[i];
            for (other_row, other_value) in other_row_map.get(&col).unwrap_or(&Vec::new()) {
                let e = result_map
                    .entry(row * target_cols + *other_row)
                    .or_insert(0.0);
                *e += value * other_value;
            }
        }

        println!("result_map: {:?}", result_map);

        Self::from_flat_map(self.nrows, other.ncols, result_map)
    }

    fn from_flat_map(nrows: usize, ncols: usize, map: HashMap<usize, f32>) -> Self {
        let (flat_indexes, values): (Vec<usize>, Vec<f32>) = map.into_iter().unzip();
        let rowind: Vec<_> = flat_indexes.iter().map(|x| x / ncols).collect();
        let colind: Vec<_> = flat_indexes.iter().map(|x| x % ncols).collect();

        println!("rowind: {:?}", rowind);
        println!("colind: {:?}", colind);
        println!("values: {:?}", values);

        Self {
            nrows,
            ncols,
            rowind,
            colind,
            values,
        }
    }
}
