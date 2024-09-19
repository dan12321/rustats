use std::{error::Error, fmt::Display};

use anyhow::Result;

#[derive(Debug)]
pub struct AggNum {
    pub mean: f64,
    // index value pairs
    pub min: f64,
    pub max: f64,
    pub count: usize,
    pub stddev: f64,
}

#[derive(Debug)]
pub struct AggNumBuilder {
    sum: f64,
    squared_sum: f64,
    len: usize,
    // index value pairs
    min: Option<f64>,
    max: Option<f64>,
}

impl AggNumBuilder {
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            squared_sum: 0.0,
            len: 0,
            min: None,
            max: None,
        }
    }

    pub fn add_val(&mut self, val: f64) {
        self.sum += val;
        self.squared_sum += val * val;
        self.len += 1;

        if let Some(min) = self.min {
            self.min = Some(min.min(val))
        } else {
            self.min = Some(val)
        }

        if let Some(max) = self.max {
            self.max = Some(max.max(val))
        } else {
            self.max = Some(val)
        }
    }

    pub fn add(&mut self, agg: &AggNumBuilder) {
        self.sum += agg.sum;
        self.squared_sum += agg.squared_sum;
        self.len += agg.len;

        if let Some(val) = agg.min {
            if let Some(min) = self.min {
                self.min = Some(min.min(val))
            } else {
                self.min = Some(val);
            }
        }

        if let Some(val) = agg.max {
            if let Some(max) = self.max {
                self.max = Some(max.max(val))
            } else {
                self.max = Some(val)
            }
        }
    }

    pub fn build(&self) -> Result<AggNum> {
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
            count: self.len,
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
