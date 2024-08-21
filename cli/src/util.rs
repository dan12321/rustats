use std::{
    error::Error, fmt::Display, fs::OpenOptions, io::{self, BufRead, BufReader}, path::PathBuf
};

use anyhow::Result;
use clap::ValueEnum;

pub fn get_buff_reader(filename: &Option<PathBuf>) -> Result<Box<dyn BufRead>> {
    let reader: Box<dyn BufRead> = if let Some(filename) = filename {
        let file = OpenOptions::new()
            .read(true)
            .open(filename)?;

        Box::new(BufReader::new(file))
    } else {
        let stdin = io::stdin();
        Box::new(BufReader::new(stdin))
    };
    Ok(reader)
}

#[derive(Debug, ValueEnum, Clone, Copy)]
pub enum DataType {
    /// Comma separated values
    CSV,
}

impl DataType {
    pub fn from_filename(filename: &PathBuf) -> Result<Self, DataTypeError> {
        let dt = filename.extension()
            .map(|e| e.to_str())
            .flatten()
            .map(|e| DataType::from_str(e, true).ok())
            .flatten();


        match dt {
            Some(dt) => Ok(dt),
            None => Err(DataTypeError::CouldNotGetFromFileExt(
                filename.to_string_lossy().to_string())),
        }
    }
}

#[derive(Debug)]
pub enum DataTypeError {
    CouldNotGetFromFileExt(String),
}

impl Display for DataTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataTypeError::CouldNotGetFromFileExt(file) => {
                write!(f, "File extension couldn't be identified on {}", file)
            }
        }
    }
}

impl Error for DataTypeError {}