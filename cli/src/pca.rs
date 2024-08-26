use std::{io::BufRead, path::PathBuf};

use clap::Args;
use stats::table::Table;

use crate::util::{self, DataType};

/// Principal component analysis
#[derive(Debug, Args)]
#[command(version, about, long_about = None)]
pub struct PcaArgs {
    /// The format of the file
    #[arg(value_enum, short, long)]
    datatype: Option<DataType>,
    /// File containing data
    filename: Option<PathBuf>,
    /// CSV delimiter
    #[arg(short, long, default_value_t = String::from(","))]
    csv_delim: String,
}

pub fn pca_main(args: PcaArgs) {
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
    match table.pca() {
        Ok(()) => (),
        Err(e) => {
            eprintln!("Error calculating pca: {}", e);
            return;
        }
    }
    println!("{}", table.to_csv(&args.csv_delim));
}

