use std::{collections::HashMap, error::Error, fmt::{write, Display}, io::BufRead};

use anyhow::{Context, Result};

#[derive(Debug)]
pub struct Table {
    headers: Vec<String>,
    col_types: Vec<ColType>,
    numerics: Vec<Column<f64>>,
    strings: Vec<Column<String>>,
    col_to_numeric: Vec<Option<usize>>,
    col_to_string: Vec<Option<usize>>,
    len: usize,
}

#[derive(Debug)]
pub struct Column<T> {
    name: String,
    entries: Vec<T>,
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
        let mut numerics = Vec::<Column<f64>>::with_capacity(headers.len());
        let mut strings = Vec::<Column<String>>::with_capacity(headers.len());
        let mut col_to_numeric = Vec::<Option<usize>>::with_capacity(headers.len());
        let mut col_to_string = Vec::<Option<usize>>::with_capacity(headers.len());
        for i in 0..first_entries.len() {
            let num = first_entries[i].parse::<f64>();
            if let Ok(n) = num {
                col_types.push(ColType::Numeric);
                let column = Column::<f64> {
                    name: headers[i].clone(),
                    entries: vec![n],
                };
                let index = numerics.len();
                col_to_numeric.push(Some(index));
                col_to_string.push(None);
                numerics.push(column);
            } else {
                col_types.push(ColType::String);
                let value = first_entries[i].to_string();
                let column = Column::<String> {
                    name: headers[i].clone(),
                    entries: vec![value],
                };
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
                        numerics[num_col].entries.push(value);
                    },
                    ColType::String => {
                        let string_col: usize = col_to_string[i].unwrap();
                        strings[string_col].entries.push(entries[i].to_string());
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

    pub fn num_agg(&self, col_name: &str) -> Result<AggNum> {
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

        for (i, val) in col.entries.iter().enumerate() {
            agg_builder.add_val(*val, i);
        }

        let agg = agg_builder.build()?;

        Ok(agg)
    }

    pub fn group_num_agg(&self, col_name: &str, group_col: &str) -> Result<Vec<(String, AggNum)>> {
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

        let group_index = match self.headers.iter().position(|h| h == group_col) {
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

        for (i, val) in col.entries.iter().enumerate() {
            let group_name = group_col.entries[i].clone();
            match builder_map.get_mut(&group_name) {
                Some(agg_builder) => agg_builder.add_val(*val, i),
                None => {
                    let mut agg_builder = AggNumBuilder::new();
                    agg_builder.add_val(*val, i);
                    builder_map.insert(group_name, agg_builder);
                }
            }
        }

        let result = builder_map.into_iter()
            .map(|(g, ab)| (g, ab.build().unwrap()))
            .collect();

        Ok(result)
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
    pub min: (usize, f64),
    pub max: (usize, f64),
    pub sum: f64,
    pub stddev: f64,
}

#[derive(Debug)]
struct AggNumBuilder {
    sum: f64,
    squared_sum: f64,
    len: usize,
    // index value pairs
    min: Option<(usize, f64)>,
    max: Option<(usize, f64)>,
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

    fn add_val(&mut self, val: f64, index: usize) {
        self.sum += val;
        self.squared_sum += val * val;
        self.len += 1;

        if let Some((_, min)) = self.min {
            if min > val {
                self.min = Some((index, val))
            }
        } else {
            self.min = Some((index, val))
        }

        if let Some((_, max)) = self.max {
            if max < val {
                self.max = Some((index, val))
            }
        } else {
            self.max = Some((index, val))
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