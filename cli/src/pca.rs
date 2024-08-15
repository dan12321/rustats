use std::{
    fs::{read, OpenOptions},
    io::{self, BufRead, BufReader}};

use crate::DataType;

use super::PcaArgs;

pub fn pca_main(args: PcaArgs) {
    let reader: Box<dyn BufRead> = if let Some(filename) = &args.filename {
        let mut file = OpenOptions::new()
            .read(true)
            .open(filename);

        let file = match file {
            Ok(f) => f,
            Err(e) => {
                println!(
                    "Could not read file {} due to error: {}",
                    filename.to_string_lossy(),
                    e.to_string()
                );
                return;
            },
        };
        Box::new(BufReader::new(file))
    } else {
        let stdin = io::stdin();
        Box::new(BufReader::new(stdin))
    };

    match args.datatype {
        DataType::Discover => {
            if let Some(filename) = &args.filename {
                if let Some(ext) = filename.extension() {
                    println!("{:?}", ext);
                } else {
                    println!(
                        "No file extension on {} --datatype must be provided",
                        filename.to_string_lossy(),
                    );
                    return;
                }
            } else {
                println!("No file extension --datatype must be provided");
                return;
            }
        },
        DataType::CSV => {},
    }
}