use std::{io::BufRead, path::PathBuf};

use anyhow::Result;
use clap::Args;
use stats::table::{TableFull, TableStream, Aggragate};

use crate::util::{self, DataType};

/// Aggregate
#[derive(Debug, Args)]
#[command(version, about, long_about = None)]
pub struct AggArgs {
    /// A column name to group by
    #[arg(value_enum, short, long)]
    group_by: Option<String>,
    /// The format of the file
    #[arg(value_enum, short, long)]
    datatype: Option<DataType>,
    /// Whether to sort by the group name
    #[arg(long)]
    sort: bool,
    /// CSV delimiter
    #[arg(short, long, default_value_t = String::from(","))]
    csv_delim: String,
    /// Aggregate as parsing line by line
    #[arg(short, long, default_value_t = false)]
    stream: bool,
    /// The name of the column to aggregate
    column: String,
    /// File containing data
    filename: Option<PathBuf>,
}

pub fn agg_main(args: AggArgs) {
    let reader: Box<dyn BufRead> = match util::get_buff_reader(&args.filename) {
        Ok(br) => br,
        Err(e) => {
            eprintln!("Could not read file due to error: {}", e.to_string());
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
                }
            }
        } else {
            eprintln!("No file provided. --datatype must be specified");
            return;
        }
    };

    let table: Result<Box<dyn Aggragate>> = if args.stream {
        match datatype {
            DataType::CSV => TableStream::from_csv(reader, &args.csv_delim)
                .map(|t| Box::from(t) as Box<dyn Aggragate>),
        }
    } else {
        match datatype {
            DataType::CSV => TableFull::from_csv(reader, &args.csv_delim)
                .map(|t| Box::from(t) as Box<dyn Aggragate>),
        }
    };
    let mut table: Box<dyn Aggragate> = match table {
        Ok(t) => Box::from(t),
        Err(e) => {
            eprintln!("Error parsing data: {}", e);
            return;
        }
    };

    let agg = match args.group_by {
        Some(group_by) => table.group_num_agg(&args.column, &group_by, args.sort),
        None => table.num_agg(&args.column),
    };
    let agg = match agg {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error calculating aggregate data: {}", e);
            return;
        }
    };
    println!("{}", agg.to_csv(&args.csv_delim));
}

