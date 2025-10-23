use std::{
    collections::HashMap,
    error::Error,
    fmt::Display,
    fs::OpenOptions,
    io::{BufRead, BufReader, Lines, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::mpsc,
};

use anyhow::{Context, Result};
use polars::prelude::*;

use crate::{agg::AggNumBuilder, hist::Hist, linalg::Matrix, pca::pca, util::sorted_insert};

#[derive(Debug, PartialEq)]
enum ColType {
    Numeric,
    String,
}

pub trait Aggragate {
    fn group_num_agg(
        &mut self,
        col_name: &str,
        group_col_name: &str,
        sort: bool,
    ) -> Result<TableFull>;
    fn num_agg(&mut self, col_name: &str) -> Result<TableFull>;
}

#[derive(Debug)]
pub struct TableFull {
    headers: Vec<String>,
    col_types: Vec<ColType>,
    numerics: Vec<Vec<f64>>,
    strings: Vec<Vec<String>>,
    col_to_numeric: Vec<Option<usize>>,
    col_to_string: Vec<Option<usize>>,
    len: usize,
}

impl TableFull {
    pub fn from_csv(reader: Box<dyn BufRead>, delimiter: &str) -> Result<Self> {
        let context = "Parsing CSV to Table";

        let mut lines = reader.lines();
        // TODO: handle comments correctly throughout file
        let mut headers = String::from("#");
        while headers.starts_with("#") {
            headers = match lines.next() {
                Some(l) => l.context(context)?,
                None => return Err(TableParserError::EmptyFile)?,
            };
        }
        let headers: Vec<String> = headers
            .split(delimiter)
            .map(|h| h.trim().to_string())
            .collect();

        let first_line = match lines.next() {
            Some(l) => l.context(context)?,
            None => return Err(TableParserError::NoData).context(context)?,
        };
        let first_entries: Vec<&str> = first_line.split(delimiter).map(|l| l.trim()).collect();
        if first_entries.len() != headers.len() {
            return Err(TableParserError::LineSizeConflict(1)).context(context);
        }
        let mut col_types = Vec::<ColType>::with_capacity(headers.len());
        let mut numerics = Vec::<Vec<f64>>::with_capacity(headers.len());
        let mut strings = Vec::<Vec<String>>::with_capacity(headers.len());
        let mut col_to_numeric = Vec::<Option<usize>>::with_capacity(headers.len());
        let mut col_to_string = Vec::<Option<usize>>::with_capacity(headers.len());
        for entry in first_entries {
            let num = entry.parse::<f64>();
            if let Ok(n) = num {
                col_types.push(ColType::Numeric);
                let column = vec![n];
                let index = numerics.len();
                col_to_numeric.push(Some(index));
                col_to_string.push(None);
                numerics.push(column);
            } else {
                col_types.push(ColType::String);
                let value = entry.to_string();
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
            let entries: Vec<&str> = line.split(delimiter).map(|e| e.trim()).collect();
            if entries.len() != headers.len() {
                return Err(TableParserError::LineSizeConflict(line_num)).context(context);
            }
            for i in 0..entries.len() {
                match col_types[i] {
                    // from col type look up we assume value exists in col_to_X
                    ColType::Numeric => {
                        let value: f64 = entries[i]
                            .parse()
                            .context(format!("Parsing numeric line {line_num}, col {i}"))
                            .context(context)?;
                        let num_col: usize = col_to_numeric[i].unwrap();
                        numerics[num_col].push(value);
                    }
                    ColType::String => {
                        let string_col: usize = col_to_string[i].unwrap();
                        strings[string_col].push(entries[i].to_string());
                    }
                }
            }
            line_num += 1;
        }

        Ok(Self {
            headers,
            col_types,
            numerics,
            strings,
            col_to_numeric,
            col_to_string,
            len: line_num - 1,
        })
    }

    pub fn to_csv(&self, delimiter: &str, precision: usize) -> String {
        let mut lines = Vec::with_capacity(self.len + 1);
        lines.push(self.headers.join(delimiter));
        for i in 0..self.len {
            let mut line: Vec<String> = Vec::with_capacity(self.headers.len());
            for j in 0..self.headers.len() {
                let value = match self.col_types[j] {
                    // Assume table is correctly formatted
                    ColType::Numeric => {
                        let index = self.col_to_numeric[j].unwrap();
                        format!("{0:.1$}", self.numerics[index][i], precision)
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
        lines.join("\n") + "\n"
    }

    pub fn pca(&mut self, round_places: Option<i32>) -> Result<()> {
        let data = self.get_matrix().context("Get matrix from table")?;
        let pca_data = pca(data, round_places)?;
        self.numerics_from_matrix(&pca_data)?;
        let mut pca_i = 1;
        for i in 0..self.headers.len() {
            if self.col_types[i] == ColType::Numeric {
                self.headers[i] = format!("pca{pca_i}");
                pca_i += 1;
            }
        }
        Ok(())
    }

    fn get_matrix(&self) -> Result<Matrix> {
        let mut elements = vec![0.0; self.len * self.numerics.len()];
        for i in 0..self.numerics.len() {
            for j in 0..self.len {
                elements[j * self.numerics.len() + i] = self.numerics[i][j];
            }
        }
        Matrix::new(elements, self.len, self.numerics.len()).context("Create new matrix")
    }

    fn numerics_from_matrix(&mut self, matrix: &Matrix) -> Result<()> {
        for i in 0..self.numerics.len() {
            self.numerics[i] = matrix.get_col(i)?;
        }
        Ok(())
    }

    pub fn hist(
        &self,
        column: &str,
        width: f64,
        min: Option<f64>,
        max: Option<f64>,
    ) -> Result<Self> {
        let head_i = match self.headers.iter().position(|h| h == column) {
            Some(i) => i,
            None => return Err(TableError::ColumnNotFound.into()),
        };
        let col = match self.col_to_numeric[head_i] {
            Some(i) => &self.numerics[i],
            None => return Err(TableError::ColumnNotNumeric.into()),
        };

        let hist = Hist::new(col, width, min, max);
        Ok(Self {
            headers: vec![column.to_string(), "count".to_string()],
            col_types: vec![ColType::Numeric, ColType::Numeric],
            col_to_numeric: vec![Some(0), Some(1)],
            col_to_string: vec![None, None],
            numerics: vec![hist.buckets, hist.counts],
            strings: vec![],
            len: hist.len,
        })
    }

    fn new_agg_table(group: Option<String>) -> Self {
        match group {
            Some(g) => TableFull {
                headers: vec![
                    g,
                    "min".into(),
                    "max".into(),
                    "mean".into(),
                    "count".into(),
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
                col_to_string: vec![Some(0), None, None, None, None, None],
                col_to_numeric: vec![None, Some(0), Some(1), Some(2), Some(3), Some(4)],
                strings: vec![vec![]],
                numerics: vec![vec![], vec![], vec![], vec![], vec![]],
                len: 0,
            },
            None => TableFull {
                headers: vec![
                    "min".into(),
                    "max".into(),
                    "mean".into(),
                    "count".into(),
                    "stddev".into(),
                ],
                col_types: vec![
                    ColType::Numeric,
                    ColType::Numeric,
                    ColType::Numeric,
                    ColType::Numeric,
                    ColType::Numeric,
                ],
                col_to_string: vec![None, None, None, None, None],
                col_to_numeric: vec![Some(0), Some(1), Some(2), Some(3), Some(4)],
                strings: vec![],
                numerics: vec![vec![], vec![], vec![], vec![], vec![]],
                len: 0,
            },
        }
    }
}

impl Aggragate for TableFull {
    fn num_agg(&mut self, col_name: &str) -> Result<TableFull> {
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

        let mut table = TableFull::new_agg_table(None);
        table.numerics[0].push(agg.min);
        table.numerics[1].push(agg.max);
        table.numerics[2].push(agg.mean);
        table.numerics[3].push(agg.count as f64);
        table.numerics[4].push(agg.stddev);
        table.len = 1;

        Ok(table)
    }

    fn group_num_agg(
        &mut self,
        col_name: &str,
        group_col_name: &str,
        sort: bool,
    ) -> Result<TableFull> {
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
        let mut groups: Vec<&str> = Vec::new();

        for (i, val) in col.iter().enumerate() {
            let group_name = group_col[i].clone();
            match builder_map.get_mut(&group_name) {
                Some(agg_builder) => agg_builder.add_val(*val),
                None => {
                    let mut agg_builder = AggNumBuilder::new();
                    agg_builder.add_val(*val);
                    builder_map.insert(group_name, agg_builder);
                    if sort {
                        sorted_insert(&mut groups, &group_col[i]);
                    } else {
                        groups.push(&group_col[i]);
                    }
                }
            }
        }

        let mut table = TableFull::new_agg_table(Some(group_col_name.into()));

        for group in groups {
            let ab = &builder_map[group];
            let agg = ab.build().unwrap();
            table.strings[0].push(group.to_string());
            table.numerics[0].push(agg.min);
            table.numerics[1].push(agg.max);
            table.numerics[2].push(agg.mean);
            table.numerics[3].push(agg.count as f64);
            table.numerics[4].push(agg.stddev);
            table.len += 1;
        }

        Ok(table)
    }
}

pub struct TableStream {
    headers: Vec<String>,
    col_types: Vec<ColType>,
    first_numerics: Vec<f64>,
    first_strings: Vec<String>,
    col_to_numeric: Vec<Option<usize>>,
    col_to_string: Vec<Option<usize>>,
    lines: Lines<Box<dyn BufRead>>,
    delimiter: String,
}

impl TableStream {
    pub fn from_csv(reader: Box<dyn BufRead>, delimiter: &str) -> Result<Self> {
        let context = "Parsing CSV to Table";

        let mut lines = reader.lines();
        let mut headers = String::from("#");
        while headers.starts_with("#") {
            headers = match lines.next() {
                Some(l) => l.context(context)?,
                None => return Err(TableParserError::EmptyFile.into()),
            };
        }
        let headers: Vec<String> = headers
            .split(delimiter)
            .map(|h| h.trim().to_string())
            .collect();

        let first_line = match lines.next() {
            Some(l) => l.context(context)?,
            None => return Err(TableParserError::NoData).context(context)?,
        };
        let first_entries: Vec<&str> = first_line.split(delimiter).map(|l| l.trim()).collect();
        if first_entries.len() != headers.len() {
            return Err(TableParserError::LineSizeConflict(1)).context(context);
        }
        let mut col_types = Vec::<ColType>::with_capacity(headers.len());
        let mut first_numerics = Vec::<f64>::with_capacity(headers.len());
        let mut first_strings = Vec::<String>::with_capacity(headers.len());
        let mut col_to_numeric = Vec::<Option<usize>>::with_capacity(headers.len());
        let mut col_to_string = Vec::<Option<usize>>::with_capacity(headers.len());
        for entry in first_entries {
            let num = entry.parse::<f64>();
            if let Ok(n) = num {
                col_types.push(ColType::Numeric);
                let index = first_numerics.len();
                col_to_numeric.push(Some(index));
                col_to_string.push(None);
                first_numerics.push(n);
            } else {
                col_types.push(ColType::String);
                let value = entry.to_string();
                let index = first_strings.len();
                col_to_numeric.push(None);
                col_to_string.push(Some(index));
                first_strings.push(value);
            }
        }

        Ok(Self {
            headers,
            col_types,
            first_numerics,
            first_strings,
            col_to_numeric,
            col_to_string,
            lines,
            delimiter: delimiter.to_string(),
        })
    }
}

impl Aggragate for TableStream {
    fn num_agg(&mut self, col_name: &str) -> Result<TableFull> {
        let col_index = match self.headers.iter().position(|h| h == col_name) {
            Some(i) => i,
            None => return Err(TableError::ColumnNotFound.into()),
        };
        let num_index = self.col_to_numeric[col_index];
        let first_val = if let Some(ni) = num_index {
            &self.first_numerics[ni]
        } else {
            return Err(TableError::ColumnNotNumeric.into());
        };

        let mut agg_builder = AggNumBuilder::new();
        agg_builder.add_val(*first_val);

        for line in &mut self.lines {
            let line = line.context("Aggregating")?;
            let parts: Vec<&str> = line.split(&self.delimiter).collect();
            let val: f64 = parts[col_index].parse().context("Aggregating")?;
            agg_builder.add_val(val);
        }

        let mut table = TableFull::new_agg_table(None);

        let agg = agg_builder.build().unwrap();
        table.numerics[0].push(agg.min);
        table.numerics[1].push(agg.max);
        table.numerics[2].push(agg.mean);
        table.numerics[3].push(agg.count as f64);
        table.numerics[4].push(agg.stddev);
        table.len += 1;

        Ok(table)
    }

    fn group_num_agg(
        &mut self,
        col_name: &str,
        group_col_name: &str,
        sort: bool,
    ) -> Result<TableFull> {
        let col_index = match self.headers.iter().position(|h| h == col_name) {
            Some(i) => i,
            None => return Err(TableError::ColumnNotFound.into()),
        };
        let num_index = self.col_to_numeric[col_index];
        let first_val = if let Some(ni) = num_index {
            &self.first_numerics[ni]
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
        let first_group = match group_string_index {
            Some(i) => self.first_strings[i].clone(),
            None => return Err(TableError::GroupColumnNotString.into()),
        };

        let mut builder_map = HashMap::<String, AggNumBuilder>::new();
        let mut groups = Vec::new();
        groups.push(first_group.to_string());
        let mut first_group_agg = AggNumBuilder::new();
        first_group_agg.add_val(*first_val);
        builder_map.insert(first_group, first_group_agg);

        for line in &mut self.lines {
            let line = line.context("Aggregating by group")?;
            let parts: Vec<&str> = line.split(&self.delimiter).collect();
            let group_name = parts[group_index];
            let val: f64 = parts[col_index].parse().context("Aggregating by group")?;
            match builder_map.get_mut(group_name) {
                Some(agg_builder) => agg_builder.add_val(val),
                None => {
                    let mut agg_builder = AggNumBuilder::new();
                    agg_builder.add_val(val);
                    builder_map.insert(group_name.to_string(), agg_builder);
                    if sort {
                        sorted_insert(&mut groups, group_name.to_string());
                    } else {
                        groups.push(group_name.to_string());
                    }
                }
            }
        }

        let mut table = TableFull::new_agg_table(Some(group_col_name.into()));

        for group in groups {
            let ab = &builder_map[&group];
            let agg = ab.build().unwrap();
            table.strings[0].push(group.to_string());
            table.numerics[0].push(agg.min);
            table.numerics[1].push(agg.max);
            table.numerics[2].push(agg.mean);
            table.numerics[3].push(agg.count as f64);
            table.numerics[4].push(agg.stddev);
            table.len += 1;
        }

        Ok(table)
    }
}

pub struct TableParallelStream {
    headers: Vec<String>,
    col_types: Vec<ColType>,
    first_numerics: Vec<f64>,
    first_strings: Vec<String>,
    col_to_numeric: Vec<Option<usize>>,
    col_to_string: Vec<Option<usize>>,
    filesize: u64,
    filename: PathBuf,
    threads: u64,
    start_pos: u64,
    delimiter: String,
}

impl TableParallelStream {
    pub fn from_csv(filename: PathBuf, delimiter: &str, threads: u64) -> Result<Self> {
        let context = "Parsing CSV to Table";
        let file = OpenOptions::new().read(true).open(&filename)?;
        let filesize = file.metadata()?.len();

        let reader = BufReader::new(file);
        let mut start_pos = 0;
        let mut lines = reader.lines();
        let mut headers = String::from("#");
        while headers.starts_with("#") {
            headers = match lines.next() {
                Some(l) => l.context(context)?,
                None => return Err(TableParserError::EmptyFile.into()),
            };
            start_pos += headers.len() + 1;
        }
        let headers: Vec<String> = headers
            .split(delimiter)
            .map(|h| h.trim().to_string())
            .collect();

        let first_line = match lines.next() {
            Some(l) => l.context(context)?,
            None => return Err(TableParserError::NoData).context(context)?,
        };
        start_pos += first_line.len() + 1;
        let first_entries: Vec<&str> = first_line.split(delimiter).map(|l| l.trim()).collect();
        if first_entries.len() != headers.len() {
            return Err(TableParserError::LineSizeConflict(1)).context(context);
        }
        let mut col_types = Vec::<ColType>::with_capacity(headers.len());
        let mut first_numerics = Vec::<f64>::with_capacity(headers.len());
        let mut first_strings = Vec::<String>::with_capacity(headers.len());
        let mut col_to_numeric = Vec::<Option<usize>>::with_capacity(headers.len());
        let mut col_to_string = Vec::<Option<usize>>::with_capacity(headers.len());
        for entry in first_entries {
            let num = entry.parse::<f64>();
            if let Ok(n) = num {
                col_types.push(ColType::Numeric);
                let index = first_numerics.len();
                col_to_numeric.push(Some(index));
                col_to_string.push(None);
                first_numerics.push(n);
            } else {
                col_types.push(ColType::String);
                let value = entry.to_string();
                let index = first_strings.len();
                col_to_numeric.push(None);
                col_to_string.push(Some(index));
                first_strings.push(value);
            }
        }

        Ok(Self {
            headers,
            col_types,
            first_numerics,
            first_strings,
            col_to_numeric,
            col_to_string,
            start_pos: start_pos as u64,
            filesize,
            threads,
            filename,
            delimiter: delimiter.to_string(),
        })
    }
}

impl Aggragate for TableParallelStream {
    fn num_agg(&mut self, col_name: &str) -> Result<TableFull> {
        let col_index = match self.headers.iter().position(|h| h == col_name) {
            Some(i) => i,
            None => return Err(TableError::ColumnNotFound.into()),
        };
        let num_index = self.col_to_numeric[col_index];
        let first_val = if let Some(ni) = num_index {
            &self.first_numerics[ni]
        } else {
            return Err(TableError::ColumnNotNumeric.into());
        };
        let chunksize = (self.filesize - self.start_pos) / self.threads;
        let mut threads = Vec::with_capacity(self.threads as usize);
        let (result_sender, result_receiver) = mpsc::channel::<AggNumBuilder>();
        for i in 0..self.threads {
            let sender = result_sender.clone();
            let mut file = OpenOptions::new().read(true).open(&self.filename)?;
            let pos = chunksize * i + self.start_pos;
            let delimiter = self.delimiter.clone();
            let last_thread = i == self.threads - 1;
            let thread = std::thread::spawn(move || {
                file.seek(SeekFrom::Start(pos - 1)).unwrap();
                let reader = BufReader::new(file);
                let mut lines = reader.lines();
                // Start from the first full line
                let skipped_line = match lines.next() {
                    Some(l) => l.unwrap(),
                    None => return,
                };
                let mut agg_builder = AggNumBuilder::new();
                let mut bytes_read = (skipped_line.len() + 1) as u64;
                // If we're the last thread keep going to the end of the file.
                // Since the remainder is less than self.threads bytes, which
                // is small, it should be fine to just have the last thread
                // handle it.
                while bytes_read <= chunksize || last_thread {
                    let line = match lines.next() {
                        Some(l) => l.unwrap(),
                        None => break,
                    };
                    bytes_read += (line.len() + 1) as u64;
                    let parts: Vec<&str> = line.split(&delimiter).collect();
                    let val = parts[col_index].parse().unwrap();
                    agg_builder.add_val(val);
                }
                sender.send(agg_builder).unwrap();
            });
            threads.push(thread);
        }

        let mut agg_builder = AggNumBuilder::new();
        agg_builder.add_val(*first_val);
        let mut received = 0;
        // If 1 thread panics this gets stuck
        // TODO: Create error channel to abort if an error is found
        // in any thread
        while received < self.threads {
            let agg = result_receiver.recv().unwrap();
            agg_builder.add(&agg);
            received += 1;
        }

        for thread in threads {
            thread.join().unwrap();
        }

        let mut table = TableFull::new_agg_table(None);

        let agg = agg_builder.build().unwrap();
        table.numerics[0].push(agg.min);
        table.numerics[1].push(agg.max);
        table.numerics[2].push(agg.mean);
        table.numerics[3].push(agg.count as f64);
        table.numerics[4].push(agg.stddev);
        table.len += 1;

        Ok(table)
    }

    fn group_num_agg(
        &mut self,
        col_name: &str,
        group_col_name: &str,
        sort: bool,
    ) -> Result<TableFull> {
        let col_index = match self.headers.iter().position(|h| h == col_name) {
            Some(i) => i,
            None => return Err(TableError::ColumnNotFound.into()),
        };
        let num_index = self.col_to_numeric[col_index];
        let first_val = if let Some(ni) = num_index {
            &self.first_numerics[ni]
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
        let first_group = match group_string_index {
            Some(i) => self.first_strings[i].clone(),
            None => return Err(TableError::GroupColumnNotString.into()),
        };

        let chunksize = (self.filesize - self.start_pos) / self.threads;
        let mut threads = Vec::with_capacity(self.threads as usize);
        let (result_sender, result_receiver) = mpsc::channel::<HashMap<String, AggNumBuilder>>();
        for i in 0..self.threads {
            let sender = result_sender.clone();
            let mut file = OpenOptions::new().read(true).open(&self.filename)?;
            let pos = chunksize * i + self.start_pos;
            let delimiter = self.delimiter.clone();
            let last_thread = i == self.threads - 1;
            let thread = std::thread::spawn(move || {
                file.seek(SeekFrom::Start(pos - 1)).unwrap();
                let reader = BufReader::new(file);
                let mut lines = reader.lines();
                // Start from the first full line
                let skipped_line = match lines.next() {
                    Some(l) => l.unwrap(),
                    None => return,
                };
                let mut group_agg_builder: HashMap<String, AggNumBuilder> = HashMap::new();
                let mut bytes_read = (skipped_line.len() + 1) as u64;
                // If we're the last thread keep going to the end of the file.
                // Since the remainder is less than self.threads bytes, which
                // is small, it should be fine to just have the last thread
                // handle it.
                while bytes_read <= chunksize || last_thread {
                    let line = match lines.next() {
                        Some(l) => l.unwrap(),
                        None => break,
                    };
                    bytes_read += (line.len() + 1) as u64;
                    let parts: Vec<&str> = line.split(&delimiter).collect();
                    let val = parts[col_index].parse().unwrap();
                    let group = parts[group_index];
                    if let Some(agg_builder) = group_agg_builder.get_mut(group) {
                        agg_builder.add_val(val);
                    } else {
                        let mut agg_builder = AggNumBuilder::new();
                        agg_builder.add_val(val);
                        group_agg_builder.insert(group.to_string(), agg_builder);
                    }
                }
                sender.send(group_agg_builder).unwrap();
            });
            threads.push(thread);
        }
        let mut group_agg_builder = HashMap::new();
        let mut agg_builder = AggNumBuilder::new();
        agg_builder.add_val(*first_val);
        group_agg_builder.insert(first_group, agg_builder);
        let mut received = 0;
        // If 1 thread panics this gets stuck
        // TODO: Create error channel to abort if an error is found
        // in any thread
        while received < self.threads {
            let groups_res = result_receiver.recv().unwrap();
            for (group_res, builder_res) in groups_res {
                if let Some(builder) = group_agg_builder.get_mut(&group_res) {
                    builder.add(&builder_res);
                } else {
                    group_agg_builder.insert(group_res, builder_res);
                }
            }
            received += 1;
        }

        for thread in threads {
            thread.join().unwrap();
        }

        let mut table = TableFull::new_agg_table(Some(group_col_name.into()));

        let groups: Vec<&String> = if sort {
            let groups = group_agg_builder.keys();
            let mut res = Vec::with_capacity(groups.len());
            for g in groups {
                res.push(g);
            }
            res.sort();
            res
        } else {
            let groups = group_agg_builder.keys();
            let mut res = Vec::with_capacity(groups.len());
            for g in groups {
                res.push(g);
            }
            res
        };

        for group in groups {
            let builder = group_agg_builder.get(group).unwrap();
            let agg = builder.build().unwrap();
            table.strings[0].push(group.to_string());
            table.numerics[0].push(agg.min);
            table.numerics[1].push(agg.max);
            table.numerics[2].push(agg.mean);
            table.numerics[3].push(agg.count as f64);
            table.numerics[4].push(agg.stddev);
            table.len += 1;
        }

        Ok(table)
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
        write!(f, "{self:?}")
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
            TableParserError::LineSizeConflict(l) => write!(f, "{self:?} on line {l}"),
            _ => write!(f, "{self:?}"),
        }
    }
}

impl Error for TableParserError {}

pub fn agg_csv(
    path: &Path,
    column: &str,
    delimiter: u8,
    group: Option<&str>,
    sort: bool,
    precision: usize,
) -> Result<Vec<u8>> {
    let lf = LazyCsvReader::new(path)
        .with_has_header(true)
        .with_separator(delimiter)
        .finish()
        .unwrap();
    let aggs = [
        col(column).min().alias("min"),
        col(column).max().alias("max"),
        col(column).mean().alias("mean"),
        col(column).count().alias("count"),
        col(column).std(1).alias("stddev"),
    ];
    let mut aggregated = if let Some(g) = group {
        let mut df = lf.group_by([col(g)]).agg(aggs).collect().unwrap();
        if sort {
            df = df.sort([g], Default::default()).unwrap();
        }
        df
    } else {
        lf.select(aggs).collect().unwrap()
    };
    let mut result: Vec<u8> = Vec::new();
    let cursor = std::io::Cursor::new(&mut result);
    let mut writer = CsvWriter::new(cursor)
        .include_header(true)
        .with_separator(delimiter)
        .with_float_precision(Some(precision))
        .with_null_value(String::from("0"));
    writer.finish(&mut aggregated).unwrap();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;

    #[test]
    fn test_full_table_agg() {
        let test_file = OpenOptions::new()
            .read(true)
            .open("../datasets/weather_stations.csv")
            .unwrap();
        let reader = Box::new(BufReader::new(test_file));
        let mut table = TableFull::from_csv(reader, ";").unwrap();
        let result = table
            .group_num_agg("measurement", "location", true)
            .unwrap();

        let mut expected_file = OpenOptions::new()
            .read(true)
            .open("../datasets/weather_stations_agg.csv")
            .unwrap();
        let mut expected_result: Vec<u8> = Vec::new();
        expected_file.read_to_end(&mut expected_result).unwrap();

        assert_eq!(
            result.to_csv(";", 0),
            String::from_utf8(expected_result).unwrap()
        );
    }

    #[test]
    fn test_streamed_table_agg() {
        let test_file = OpenOptions::new()
            .read(true)
            .open("../datasets/weather_stations.csv")
            .unwrap();
        let reader = Box::new(BufReader::new(test_file));
        let mut table = TableStream::from_csv(reader, ";").unwrap();
        let result = table
            .group_num_agg("measurement", "location", true)
            .unwrap();

        let mut expected_file = OpenOptions::new()
            .read(true)
            .open("../datasets/weather_stations_agg.csv")
            .unwrap();
        let mut expected_result: Vec<u8> = Vec::new();
        expected_file.read_to_end(&mut expected_result).unwrap();

        assert_eq!(
            result.to_csv(";", 0),
            String::from_utf8(expected_result).unwrap()
        );
    }

    #[test]
    fn test_threaded_stream_table_agg() {
        let test_file = "../datasets/weather_stations.csv";
        let mut table = TableParallelStream::from_csv(test_file.into(), ";", 3).unwrap();
        let result = table
            .group_num_agg("measurement", "location", true)
            .unwrap();

        let mut expected_file = OpenOptions::new()
            .read(true)
            .open("../datasets/weather_stations_agg.csv")
            .unwrap();
        let mut expected_result: Vec<u8> = Vec::new();
        expected_file.read_to_end(&mut expected_result).unwrap();

        assert_eq!(
            result.to_csv(";", 0),
            String::from_utf8(expected_result).unwrap()
        );
    }

    #[test]
    fn test_polars_agg() {
        let test_file = Path::new("../datasets/weather_stations.csv");
        let result = agg_csv(
            &test_file,
            "measurement",
            b';',
            Some("location"),
            true,
            0,
        ).unwrap();

        let mut expected_file = OpenOptions::new()
            .read(true)
            .open("../datasets/weather_stations_agg.csv")
            .unwrap();
        let mut expected_result: Vec<u8> = Vec::new();
        expected_file.read_to_end(&mut expected_result).unwrap();

        assert_eq!(
            String::from_utf8(result).unwrap(),
            String::from_utf8(expected_result).unwrap()
        );
    }
}
