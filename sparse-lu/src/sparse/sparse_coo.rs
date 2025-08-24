use crate::sparse::sparse_matrix::SparseMatrixTrait;
use std::collections::HashMap;
use rand::Rng;
use std::collections::HashSet;


pub struct SparseCOO {
    nrows: usize,
    ncols: usize,
    rowind: Vec<usize>, // length = nnz
    colind: Vec<usize>, // length = nnz
    values: Vec<f32>,   // length = nnz
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
        let mut rng = rand::rng();
        let need = ((nrows * ncols) as f32 * density).floor() as usize;

        // We know capacity - remove dynamic allocation overhead
        let mut pairs: HashSet<(usize, usize)> = HashSet::with_capacity(need);
        while pairs.len() < need {
            let i = rng.random_range(0..nrows);
            let j = rng.random_range(0..ncols);
            pairs.insert((i, j));
        }

        // Convert to vectors and generate corresponding values
        let coords: Vec<(usize, usize)> = pairs.into_iter().collect();
        let mut values: Vec<f32> = Vec::with_capacity(need);
            
        for _ in 0..need {
            values.push(rng.random::<f32>());
        }

        let (rowind, colind): (Vec<usize>, Vec<usize>) = coords.into_iter().unzip();

        Self {
            nrows,
            ncols,
            rowind,
            colind,
            values,
        }
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
    pub fn from_dense(dense: Vec<Vec<f32>>) -> Self {
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
    pub fn to_dense(&self) -> Vec<Vec<f32>> {
        let mut dense = vec![vec![0.0; self.ncols]; self.nrows];
        for i in 0..self.nnz() {
            dense[self.rowind[i]][self.colind[i]] = self.values[i];
        }
        dense
    }
    
    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.ncols, other.nrows);
        let mut other_row_map: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for i in 0..other.nnz() {
            other_row_map
                .entry(other.rowind[i])
                .or_default()
                .push((other.colind[i], other.values[i]));
        }

        let mut result_map: HashMap<(usize, usize), f32> = HashMap::new();

        for i in 0..self.nnz() {
            let row = self.rowind[i];
            let col = self.colind[i];
            let value = self.values[i];
            for (other_row, other_value) in other_row_map.get(&col).unwrap_or(&Vec::new()) {
                let e = result_map.entry((row, *other_row)).or_insert(0.0);
                *e += value * other_value;
            }
        }

        Self::from_map(self.nrows, other.ncols, result_map)
    }

    fn from_map(nrows: usize, ncols: usize, map: HashMap<(usize, usize), f32>) -> Self {
        let (keys, values): (Vec<(usize, usize)>, Vec<f32>) = map.into_iter().unzip();

        let (rowind, colind)  = keys.into_iter().unzip();

        Self {
            nrows,
            ncols,
            rowind,
            colind,
            values,
        }
    }
}
