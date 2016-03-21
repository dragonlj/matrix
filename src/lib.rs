//! 实现了两个Matrix：
//!
//! Vec<Vec<f64>> 返回皆为值 ，符合trait Matrix
//!
//! VecMatrix 返回皆为引用，符合trait Matrix
//! VecMatrix 函数返回的SubMatrix为一个引用的Matrix
//! 

use std::ops::{ Add, Sub, Range, Index };

pub trait LinearSpace {
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn zero(&self) -> Self;
    fn mul(&self, rhs: f64) -> Self;
}

pub trait Matrix<'a, T: Add<T> + Sub<T>> : Sized {
    type RowVector;
    type ColVector;
    type MatrixSlice;
    type RowIter;
    type ColIter;

    fn row_len(&self) -> usize;
    fn col_len(&self) -> usize;
    
    fn get(&self, i: usize, j: usize) -> T;

    fn row(&'a self, i: usize) -> Self::RowVector;
    fn col(&'a self, i: usize) -> Self::ColVector;

    fn sub_matrix(&'a self, row: Range<usize>, col: Range<usize>) -> Self::MatrixSlice;

    fn row_iter(&'a self) -> Self::RowIter;
    fn col_iter(&'a self) -> Self::ColIter;

    //fn as_row_major_vector(&self) -> Self::RowVector;
    //fn as_col_major_vector(&self) -> Self::ColVector;

    //fn mul_vec(&self, rhs: &Self::Vector) -> Self::Vector;
    //fn mul_mat(&self, rhs: &Self) -> Self;
    
}

pub trait MatrixMut<'a, T: Add<T> + Sub<T>>: Matrix<'a, T> {
    fn set(&mut self, i: usize, j: usize, x: &T);
    fn set_row(&mut self, i: usize, r: &Self::RowVector);
    fn set_col(&mut self, i: usize, c: &Self::ColVector);
    fn set_sub_matrix(&'a mut self, row: Range<usize>, col: Range<usize>, m: &Self);
}

pub struct RowIterator<'a> {
    v: &'a Vec<Vec<f64>>,
    row: usize,
} 

impl<'a> Iterator for RowIterator<'a> {
    type Item = Vec<f64>;
    fn next(&mut self) -> Option<Self::Item> {
        let v = self.v[self.row].clone();
        self.row = self.row + 1;
        if self.row <= self.v.len() {
            Some(v)
        } else {
            None
        }
    }
}

pub struct ColIterator<'a> {
    v: &'a Vec<Vec<f64>>,
    col: usize,
} 

impl<'a> Iterator for ColIterator<'a> {
    type Item = Vec<f64>;
    fn next(&mut self) -> Option<Self::Item> {
        let v = self.v.iter().map(|x| x[self.col]).collect::<Vec<f64>>();
        self.col = self.col + 1;
        if self.col <= self.v[0].len() {
            Some(v)
        } else {
            None
        }
    }
}

impl<'a> Matrix<'a, f64> for Vec<Vec<f64>> {
    type RowVector = Vec<f64>;
    type ColVector = Vec<f64>;
    type MatrixSlice = Self;
    type RowIter = RowIterator<'a>;
    type ColIter = ColIterator<'a>;

    fn row_len(&self) -> usize {
        self.len()
    }
    
    fn col_len(&self) -> usize {
        self[0].len()
    } 
    
    fn get(&self, i: usize, j: usize) -> f64 {
        self[i][j]
    }

    fn row(&self, i: usize) -> Self::RowVector {
        self[i].clone()
    }

    fn col(&self, j: usize) -> Self::ColVector {
        self.iter().map(|x| x[j]).collect()
    }

    fn sub_matrix(&'a self, row: Range<usize>, col: Range<usize>) -> Self {
        let mut m = vec![ vec![0f64; col.len()]; row.len() ];
        for (i, row_i) in row.enumerate() {
            for (j, col_j) in col.clone().enumerate() {
                m[i][j] = self[row_i][col_j]
            }
        }
        m
    }

    fn row_iter(&'a self) -> Self::RowIter {
        RowIterator { v: self, row: 0 }
    }

    fn col_iter(&'a self) -> Self::ColIter {
        ColIterator { v: self, col: 0 }
    }

    
    //fn as_row_major_vector(&self) -> Self::RowVector {
    //    self.iter().flat_map(|x| x.clone().into_iter()).collect()
    //}
    // 
    //fn as_col_major_vector(&self) -> Self::ColVector {
    //    let mut m = Vec::new();
    //    for i in 0..self.col_len() {
    //        for j in 0..self.row_len() {
    //            m.push(self[j][i]);
    //        }
    //    }
    //    m
    //}
}


impl<'a> MatrixMut<'a, f64> for Vec<Vec<f64>> {
    fn set(&mut self, i: usize, j: usize, x: &f64) {
        self[i][j] = *x
    }

    fn set_row(&mut self, i: usize, r: &Self::RowVector) {
        self[i] = r.clone();
    }
     
    fn set_col(&mut self, j: usize, c: &Self::ColVector) {
        for i in 0..self.len() {
            self[i][j] = c[i];
        }
    }
    
    fn set_sub_matrix(&'a mut self, row: Range<usize>, col: Range<usize>, m: &Self) {
        for (i, row_i) in row.enumerate() {
            for (j, col_j) in col.clone().enumerate() {
                self[row_i][col_j] = m[i][j];
            }
        }
    }
}

pub struct SubRowSlice<'a> {
    m: &'a VecMatrix,
    row: usize,

    col_start: usize,
    col_end: usize,
}

impl<'a> SubRowSlice<'a> {
    pub fn iter(&self) -> SubRowSliceIter {
        SubRowSliceIter {
            m: self.m,
            row: self.row,
            at: 0,
            end: self.len() }
    }

    pub fn len(&self) -> usize {
        self.col_end - self.col_start
    }
}

impl<'a> Index<usize> for SubRowSlice<'a> {
    type Output = f64;
    fn index(&self, index: usize) -> &f64 {
        &self.m.v[self.row][self.col_start + index]
    }
}

pub struct SubColSlice<'a> {
    m: &'a VecMatrix,
    col: usize,

    row_start: usize,
    row_end: usize,
}

impl<'a> SubColSlice<'a> {
    pub fn iter(&self) -> SubColSliceIter {
        SubColSliceIter {
            m: self.m,
            col: self.col,
            at: 0,
            end: self.len() }
    }

    pub fn len(&self) -> usize {
        self.row_end - self.row_start
    }
}

impl<'a> Index<usize> for SubColSlice<'a> {
    type Output = f64;
    fn index(&self, index: usize) -> &f64 {
        &self.m.v[self.row_start + index][self.col]
    }
}

pub struct SubRowSliceIter<'a> {
    m: &'a VecMatrix,
    row: usize, //指向的行
    
    at: usize, //指向行中元素位置
    end: usize, 
}

impl<'a> Iterator for SubRowSliceIter<'a> {
    type Item = &'a f64;
    fn next(&mut self) -> Option<Self::Item> {
        let at = self.at;
        self.at += 1;
        if at < self.end {
            Some(&self.m.v[self.row][at])
        } else {
            None
        }
    }
}

pub struct SubColSliceIter<'a> {
    m: &'a VecMatrix,
    col: usize, //指向的列
    
    at: usize, //指向列中元素位置
    end: usize, 
}

impl<'a> Iterator for SubColSliceIter<'a> {
    type Item = &'a f64;
    fn next(&mut self) -> Option<Self::Item> {
        let at = self.at;
        self.at += 1;
        if at < self.end {
            Some(&self.m.v[at][self.col])
        } else {
            None
        }
    }
}

pub struct SubMatRowIter<'a> {
    m: &'a VecMatrix,

    row_at: usize,
    row_end: usize,

    col_start: usize,
    col_end: usize,
}

impl<'a> Iterator for SubMatRowIter<'a> {
    type Item = SubRowSlice<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.row_at;
        self.row_at = self.row_at + 1;
        if i < self.row_end {
            Some(self.m.sub_row(i, self.col_start, self.col_end))
        } else {
            None
        }
    }
}

pub struct SubMatColIter<'a> {
    m: &'a VecMatrix,

    col_at: usize,
    col_end: usize,

    row_start: usize,
    row_end: usize,
}

impl<'a> Iterator for SubMatColIter<'a> {
    type Item = SubColSlice<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.col_at;
        self.col_at = self.col_at + 1;
        if i < self.col_end {
            Some(self.m.sub_col(i, self.row_start, self.row_end))
        } else {
            None
        }
    }
}

///f64的矩阵
pub struct VecMatrix {
    v: Vec<Vec<f64>>,
}

impl VecMatrix {
    fn sub_row(&self, i: usize, col_start: usize, col_end: usize) -> SubRowSlice {
         SubRowSlice { m: self, row: i, col_start: col_start, col_end: col_end }
    }

    fn sub_col(&self, i: usize, row_start: usize, row_end: usize) -> SubColSlice {
         SubColSlice { m: self, col: i, row_start: row_start, row_end: row_end }
    }
}

impl<'a> Matrix<'a, f64> for VecMatrix {
    type RowVector = SubRowSlice<'a>;
    type ColVector = SubColSlice<'a>;
    type MatrixSlice = SubMatrix<'a>;
    type RowIter = SubMatRowIter<'a>;
    type ColIter = SubMatColIter<'a>;
    
    ///行数
    fn row_len(&self) -> usize {
         self.v.len()
    }

    ///列数
    fn col_len(&self) -> usize {
         self.v[0].len()
    }
    
    ///读元素
    fn get(&self, i: usize, j: usize) -> f64 {
        self.v[i][j]
    }

    ///读一行
    fn row(&'a self, i: usize)-> SubRowSlice {
         SubRowSlice { m: self, row: i, col_start: 0, col_end: self.col_len() }
    }

    ///读一列
    fn col(&self, i: usize)-> SubColSlice {
         SubColSlice { m: self, col: i, row_start: 0, row_end: self.row_len() }
    }

    ///读子矩阵
    fn sub_matrix(&'a self, row: Range<usize>, col: Range<usize>) -> Self::MatrixSlice {
        SubMatrix { m: self, row_range: row, col_range: col }
    }

    ///遍历行
    fn row_iter(&self) ->SubMatRowIter {
        SubMatRowIter{
            m: self,
            row_at: 0,
            row_end: self.row_len(),
            
            col_start: 0,
            col_end: self.col_len(),
        }
    }

    ///遍历列
    fn col_iter(&self) ->SubMatColIter {
        SubMatColIter{
            m: self,
            col_at: 0,
            col_end: self.row_len(),
            
            row_start: 0,
            row_end: self.row_len(),
        }
    }
}

impl<'a> MatrixMut<'a, f64> for VecMatrix {
    ///写元素
    fn set(&mut self, i: usize, j: usize, x: &f64) {
        self.v[i][j] = *x;
    }

    fn set_row(&mut self, i: usize, r: &Self::RowVector) {
        self.v[i] = r.iter().cloned().collect::<Vec<f64>>();
    }
    
    fn set_col(&mut self, j: usize, c: &Self::ColVector) {
        let v = c.iter().cloned().collect::<Vec<f64>>();
        for i in 0..self.v.len() {
            self.v[i][j] = v[i];
        }
    }

    ///写子矩阵
    fn set_sub_matrix(&'a mut self, row: Range<usize>, col: Range<usize>, m: &Self) {
        for (i, row_i) in row.enumerate() {
            for (j, col_j) in col.clone().enumerate() {
                self.v[row_i][col_j] = m.v[i][j];
            }
        }
    }

}

impl LinearSpace for VecMatrix {
    fn add(&self, rhs: &Self) -> Self {
        let mut m = vec![ vec![0f64; self.v.col_len()]; self.v.row_len() ];
        for i in 0..self.row_len() {
            for j in 0..self.col_len() {
                m[i][j] = self.v[i][j] + rhs.v[i][j];
            }
        }
        VecMatrix { v: m }
    }
    
    fn sub(&self, rhs: &Self) -> Self {
        let mut m = vec![ vec![0f64; self.v.col_len()]; self.row_len() ];
        for i in 0..self.row_len() {
            for j in 0..self.col_len() {
                m[i][j] = self.v[i][j] - rhs.v[i][j];
            }
        }
        VecMatrix { v: m }
    }
    
    fn zero(&self) -> Self {
        VecMatrix { v:  vec![ vec![0f64; self.col_len()]; self.row_len() ] }
    }
    
    fn mul(&self, rhs: f64) -> Self {
        let mut m = vec![ vec![0f64; self.col_len()]; self.v.row_len() ];
        for i in 0..self.row_len() {
            for j in 0..self.col_len() {
                m[i][j] = self.v[i][j] * rhs;
            }
        }
        VecMatrix { v: m }  
    }
}

///以引用形式的只读的子矩阵
pub struct SubMatrix<'a> {
    m: &'a VecMatrix,
    row_range: Range<usize>,
    col_range: Range<usize>,
}

impl<'a> Matrix<'a,f64> for SubMatrix<'a> {
    type RowVector = SubRowSlice<'a>;
    type ColVector = SubColSlice<'a>;
    type MatrixSlice = SubMatrix<'a>;
    type RowIter = SubMatRowIter<'a>;
    type ColIter = SubMatColIter<'a>;
    
    ///行数
    fn row_len(&self) -> usize {
         self.row_range.len()
    }

    ///列数
    fn col_len(&self) -> usize {
        self.col_range.len()
    }

    ///读元素
    fn get(&self, i: usize, j: usize) -> f64 {
        self.m.v[self.row_range.start + i][self.col_range.start + j]
    }

    ///读一行
    fn row(&self, i: usize) -> SubRowSlice {
        SubRowSlice {
            m: self.m,
            row: self.row_range.start + i,
            col_start: self.col_range.start,
            col_end: self.col_range.end
        }
    }

    ///读一列
    fn col(&self, i: usize) -> SubColSlice {
        SubColSlice {
            m: self.m,
            col: self.col_range.start + i,
            row_start: self.row_range.start,
            row_end: self.row_range.end
        }
    }

    ///读子矩阵
    fn sub_matrix(&self, row: Range<usize>, col: Range<usize>) -> SubMatrix {
        SubMatrix {
            m: self.m,
            row_range: self.row_range.start + row.start..self.row_range.start + row.end,
            col_range: self.col_range.start + col.start..self.col_range.start + col.end,
        }
    }

    ///遍历行
    fn row_iter(&self) ->SubMatRowIter {
        SubMatRowIter{
            m: self.m,
            row_at: self.row_range.start,
            row_end: self.row_range.end,
            col_start: self.col_range.start,
            col_end: self.col_range.end
        }
    }

    ///遍历列
    fn col_iter(&self) ->SubMatColIter {
        SubMatColIter{
            m: self.m,
            col_at: self.col_range.start,
            col_end: self.col_range.end,
            row_start: self.row_range.start,
            row_end: self.row_range.end
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let mut m = vec![ vec![0f64; 10]; 10 ];
        m.set_sub_matrix(0..2,
                         0..2,
                         &vec![vec![11f64,12.],
                               vec![21f64,22.]]);
        println!("m is {:?}", m);
        let sub = m.sub_matrix(0..2,0..2);
        println!("sub matrix is {:?}", sub);

        let row = m.row(0);
        println!("row 0 is {:?}", row);

        let col = m.col(0);
        println!("col 0 is {:?}", col);
        
        assert_eq!(1,2);
    }


    #[test]
    fn it_works2() {
        let m = vec![ vec![0f64, 1.9, 2.0]; 10 ];
        let m = VecMatrix {v: m};
        let m = m.sub_matrix(0..2, 0..2);
        for row in m.col_iter() {
            for x in row.iter() {
                print!("x = {:?}\t", x);
            }
            println!("");
        }

        assert_eq!(1,2);
    }

}
