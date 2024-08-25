use std::{collections::HashMap, error::Error, fmt::Display, io::BufRead};

use anyhow::{Context, Result};

use crate::{linalg::Matrix, pca::pca};

#[derive(Debug)]
pub struct Table {
    headers: Vec<String>,
    col_types: Vec<ColType>,
    numerics: Vec<Vec<f64>>,
    strings: Vec<Vec<String>>,
    col_to_numeric: Vec<Option<usize>>,
    col_to_string: Vec<Option<usize>>,
    len: usize,
}

#[derive(Debug)]
pub enum ColType {
    Numeric,
    String,
}

impl Table {
    pub fn from_csv(reader: Box<dyn BufRead>, delimiter: &str) -> Result<Self> {
        let context = "Parsing CSV to Table";

        let mut lines = reader.lines();
        let mut headers = String::from("#");
        while headers.starts_with("#") {
            headers = match lines.next() {
                Some(l) => l.context(context)?,
                None => return Err(TableParserError::EmptyFile).context(context)?,
            };
        }
        let headers: Vec<String> = headers.split(delimiter)
            .map(|h| h.trim().to_string())
            .collect();

        let first_line = match lines.next() {
            Some(l) => l.context(context)?,
            None => return  Err(TableParserError::NoData).context(context)?,
        };
        let first_entries: Vec<&str> = first_line.split(delimiter)
            .map(|l| l.trim())
            .collect();
        if first_entries.len() != headers.len() {
            return Err(TableParserError::LineSizeConflict(1)).context(context);
        }
        let mut col_types = Vec::<ColType>::with_capacity(headers.len());
        let mut numerics = Vec::<Vec<f64>>::with_capacity(headers.len());
        let mut strings = Vec::<Vec<String>>::with_capacity(headers.len());
        let mut col_to_numeric = Vec::<Option<usize>>::with_capacity(headers.len());
        let mut col_to_string = Vec::<Option<usize>>::with_capacity(headers.len());
        for i in 0..first_entries.len() {
            let num = first_entries[i].parse::<f64>();
            if let Ok(n) = num {
                col_types.push(ColType::Numeric);
                let column = vec![n];
                let index = numerics.len();
                col_to_numeric.push(Some(index));
                col_to_string.push(None);
                numerics.push(column);
            } else {
                col_types.push(ColType::String);
                let value = first_entries[i].to_string();
                let column = vec![value];
                let index = strings.len();
                col_to_numeric.push(None);
                col_to_string.push(Some(index));
                strings.push(column);
            }
        }

        let mut line_num = 2;
        for line in lines {
            let line = line?;
            let entries: Vec<&str> = line.split(delimiter)
                .map(|e| e.trim())
                .collect();
            if entries.len() != headers.len() {
                return Err(TableParserError::LineSizeConflict(line_num)).context(context);
            }
            for i in 0..entries.len() {
                match col_types[i] {
                    // from col type look up we assume value exists in col_to_X
                    ColType::Numeric => {
                        let value: f64 = entries[i].parse()
                            .context(format!(
                                "Failed to parse numeric on line {}, col {}",
                                line_num,
                                i,
                            ))
                            .context(context)?;
                        let num_col: usize = col_to_numeric[i].unwrap();
                        numerics[num_col].push(value);
                    },
                    ColType::String => {
                        let string_col: usize = col_to_string[i].unwrap();
                        strings[string_col].push(entries[i].to_string());
                    }
                }
            }
            line_num += 1;
        }

        Ok(Table {
            headers,
            col_types,
            numerics,
            strings,
            col_to_numeric,
            col_to_string,
            len: line_num - 1,
        })
    }

    pub fn num_agg(&self, col_name: &str) -> Result<Self> {
        let col_index = self.headers.iter().position(|h| h == col_name);
        let num_index = if let Some(c) = col_index {
            self.col_to_numeric[c]
        } else {
            return Err(TableError::ColumnNotFound.into());
        };
        let col = if let Some(ni) = num_index {
            &self.numerics[ni]
        } else {
            return Err(TableError::ColumnNotNumeric.into());
        };

        let mut agg_builder = AggNumBuilder::new();

        for val in col.iter() {
            agg_builder.add_val(*val);
        }

        let agg = agg_builder.build()?;

        Ok(Table {
            headers: vec![
                "min".into(),
                "max".into(),
                "mean".into(),
                "sum".into(),
                "stddev".into()
            ],
            col_types: vec![
                ColType::Numeric,
                ColType::Numeric,
                ColType::Numeric,
                ColType::Numeric,
                ColType::Numeric,
            ],
            col_to_numeric: vec![
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
            ],
            col_to_string: vec![
                None,
                None,
                None,
                None,
                None,
            ],
            numerics: vec![
                vec![agg.min],
                vec![agg.max],
                vec![agg.mean],
                vec![agg.sum],
                vec![agg.stddev],
            ],
            strings: vec![],
            len: 1,
        })
    }

    pub fn group_num_agg(&self, col_name: &str, group_col_name: &str) -> Result<Table> {
        let col_index = self.headers.iter().position(|h| h == col_name);
        let num_index = if let Some(c) = col_index {
            self.col_to_numeric[c]
        } else {
            return Err(TableError::ColumnNotFound.into());
        };
        let col = if let Some(ni) = num_index {
            &self.numerics[ni]
        } else {
            return Err(TableError::ColumnNotNumeric.into());
        };

        let group_index = match self.headers.iter().position(|h| h == group_col_name) {
            Some(i) => i,
            None => return Err(TableError::GroupColumnNotFound.into()),
        };
        let group_string_index = match self.col_types[group_index] {
            ColType::Numeric => return Err(TableError::GroupOnNumericNotImplemented.into()),
            ColType::String => self.col_to_string[group_index],
        };
        let group_col = match group_string_index {
            Some(i) => &self.strings[i],
            None => return Err(TableError::GroupColumnNotString.into()),
        };

        let mut builder_map = HashMap::<String, AggNumBuilder>::new();

        for (i, val) in col.iter().enumerate() {
            let group_name = group_col[i].clone();
            match builder_map.get_mut(&group_name) {
                Some(agg_builder) => agg_builder.add_val(*val),
                None => {
                    let mut agg_builder = AggNumBuilder::new();
                    agg_builder.add_val(*val);
                    builder_map.insert(group_name, agg_builder);
                }
            }
        }

        let mut table = Table {
            headers: vec![
                group_col_name.into(),
                "min".into(),
                "max".into(),
                "mean".into(),
                "sum".into(),
                "stddev".into(),
            ],
            col_types: vec![
                ColType::String,
                ColType::Numeric,
                ColType::Numeric,
                ColType::Numeric,
                ColType::Numeric,
                ColType::Numeric,
            ],
            col_to_string: vec![
                Some(0),
                None,
                None,
                None,
                None,
                None,
            ],
            col_to_numeric: vec![
                None,
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
            ],
            strings: vec![vec![]],
            numerics: vec![
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            ],
            len: 0,
        };

        for (g, ab) in builder_map {
            let agg = ab.build().unwrap();
            table.strings[0].push(g);
            table.numerics[0].push(agg.min);
            table.numerics[1].push(agg.max);
            table.numerics[2].push(agg.mean);
            table.numerics[3].push(agg.sum);
            table.numerics[4].push(agg.stddev);
            table.len += 1;
        }

        Ok(table)
    }

    pub fn to_csv(&self, delimiter: &str) -> String {
        let mut lines = Vec::with_capacity(self.len + 1);
        lines.push(self.headers.join(delimiter));
        for i in 0..self.len {
            let mut line: Vec<String> = Vec::with_capacity(self.headers.len()); 
            for j in 0..self.headers.len() {
                let value = match self.col_types[j] {
                    // Assume table is correctly formatted
                    ColType::Numeric => {
                        let index = self.col_to_numeric[j].unwrap();
                        self.numerics[index][i].to_string()
                    }
                    ColType::String => {
                        let index = self.col_to_string[j].unwrap();
                        self.strings[index][i].clone()
                    }
                };
                line.push(value);
            }
            lines.push(line.join(delimiter));
        }
        lines.join("\n")
    }

    pub fn pca(&mut self) -> Result<()> {
        let data = self.get_matrix().context("Get matrix from table")?;
        let pca_data = pca(data)?;
        self.numerics_from_matrix(&pca_data)?;
        Ok(())
    }

    fn get_matrix(&self) -> Result<Matrix> {
        Matrix::new(
            self.numerics.concat(),
            self.len,
            self.numerics.len()
        ).context("Create new matrix")
    }

    pub fn numerics_from_matrix(&mut self, matrix: &Matrix) -> Result<()> {
        for i in 0..self.numerics.len() {
            self.numerics[i] = matrix.get_col(i)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
enum TableError {
    ColumnNotFound,
    ColumnNotNumeric,
    GroupColumnNotFound,
    GroupOnNumericNotImplemented,
    GroupColumnNotString,
}

impl Display for TableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for TableError {}

#[derive(Debug)]
enum TableParserError {
    EmptyFile,
    NoData,
    LineSizeConflict(usize),
}

impl Display for TableParserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TableParserError::LineSizeConflict(l) => write!(f, "{:?} on line {}", self, l),
            _ => write!(f, "{:?}", self),
        }
    }
}

impl Error for TableParserError {}

#[derive(Debug)]
pub struct AggNum {
    pub mean: f64,
    // index value pairs
    pub min: f64,
    pub max: f64,
    pub sum: f64,
    pub stddev: f64,
}

#[derive(Debug)]
struct AggNumBuilder {
    sum: f64,
    squared_sum: f64,
    len: usize,
    // index value pairs
    min: Option<f64>,
    max: Option<f64>,
}

impl AggNumBuilder {
    fn new() -> Self {
        Self {
            sum: 0.0,
            squared_sum: 0.0,
            len: 0,
            min: None,
            max: None,
        }
    }

    fn add_val(&mut self, val: f64) {
        self.sum += val;
        self.squared_sum += val * val;
        self.len += 1;

        if let Some(min) = self.min {
            if min > val {
                self.min = Some(val)
            }
        } else {
            self.min = Some(val)
        }

        if let Some(max) = self.max {
            if max < val {
                self.max = Some(val)
            }
        } else {
            self.max = Some(val)
        }
    }

    fn build(&self) -> Result<AggNum> {
        let max = match self.max {
            Some(m) => m,
            None => {
                return Err(AggNumBuilderError::NoMax.into());
            }
        };
        let min = match self.min {
            Some(m) => m,
            None => {
                return Err(AggNumBuilderError::NoMin.into());
            }
        };

        let mean = self.sum / self.len as f64;
        let stddev = ((self.squared_sum / self.len as f64) - mean * mean).sqrt();

        Ok(AggNum {
            mean,
            min,
            max,
            sum: self.sum,
            stddev,
        })
    }
}

#[derive(Debug)]
enum AggNumBuilderError {
    NoMin,
    NoMax,
}

impl Display for AggNumBuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for AggNumBuilderError {}