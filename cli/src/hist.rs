use std::{io::BufRead, path::PathBuf};

use clap::Args;
use stats::table::Table;

use crate::util::{self, DataType};

/// Principal component analysis
#[derive(Debug, Args)]
#[command(version, about, long_about = None)]
pub struct HistArgs {
    /// The format of the file
    #[arg(value_enum, short, long)]
    datatype: Option<DataType>,
    /// CSV delimiter
    #[arg(long, default_value_t = String::from(","))]
    csv_delim: String,
    /// Min value
    #[arg(long)]
    min: Option<f64>,
    /// Max value
    #[arg(long)]
    max: Option<f64>,
    /// The width of each histogram bucket
    #[arg(short, long)]
    width: f64,
    /// The width of each histogram bucket
    #[arg(short, long)]
    column: String,
    /// File containing data
    filename: Option<PathBuf>,
}

pub fn hist_main(args: HistArgs) {
    let reader: Box<dyn BufRead> = match util::get_buff_reader(&args.filename) {
        Ok(br) => br,
        Err(e) => {
            eprintln!(
                "Could not read file due to error: {}",
                e.to_string()
            );
            return;
        }
    };

    let datatype = if let Some(d) = args.datatype {
        d
    } else {
        if let Some(f) = args.filename {
            match util::DataType::from_filename(&f) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{}", e);
                    return;
                },
            }
        } else {
            eprintln!("No file provided. --datatype must be specified");
            return;
        }
    };

    let table = match datatype {
        DataType::CSV => Table::from_csv(reader, &args.csv_delim),
    };
    let mut table = match table {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error parsing data: {}", e);
            return;
        }
    };
    let result = match table.hist(&args.column, args.width, args.min, args.max) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error calculating pca: {}", e);
            return;
        }
    };
    println!("{}", result.to_csv(&args.csv_delim));
}

