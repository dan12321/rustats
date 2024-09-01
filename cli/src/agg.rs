use std::{io::BufRead, path::PathBuf};

use clap::Args;
use stats::table::Table;

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
    /// CSV delimiter
    #[arg(short, long, default_value_t = String::from(","))]
    csv_delim: String,
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

    let table = match datatype {
        DataType::CSV => Table::from_csv(reader, &args.csv_delim),
    };
    let table = match table {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error parsing data: {}", e);
            return;
        }
    };

    let agg = match args.group_by {
        Some(group_by) => {
            let aggs = table.group_num_agg(&args.column, &group_by);
            match aggs {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("Error calculating aggregate data: {}", e);
                    return;
                }
            }
        }
        None => {
            let agg = table.num_agg(&args.column);
            match agg {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("Error calculating aggregate data: {}", e);
                    return;
                }
            }
        }
    };
    println!("{}", agg.to_csv(&args.csv_delim));
}

