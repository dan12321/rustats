use std::{io::BufRead, path::PathBuf};

use clap::Args;

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

    match datatype {
        DataType::CSV => {},
    }
}

