mod pca;

use std::path::PathBuf;

use clap::{Parser, Subcommand, Args, ValueEnum};

/// Tools for Stats
#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    tool: Tool,
}

/// Tools for Stats
#[derive(Subcommand, Debug)]
#[command(version, about, long_about = None)]
enum Tool {
    /// Principal component analysis
    #[command(version, about, long_about = None)]
    PCA(PcaArgs),
}

/// Principal component analysis
#[derive(Debug, Args)]
#[command(version, about, long_about = None)]
struct PcaArgs {
    /// The format of the file
    #[arg(value_enum, short, long, default_value_t = DataType::Discover)]
    datatype: DataType,
    /// File containing data
    filename: Option<PathBuf>,
}

#[derive(Debug, ValueEnum, Clone, Copy)]
enum DataType {
    /// Comma separated values
    CSV,
    /// Discover based on file extension
    /// Not valid for stdin
    Discover,
}

fn main() {
    let cli = Cli::parse();
    println!("{:?}", cli);
    match cli.tool {
        Tool::PCA(args) => pca::pca_main(args),
    }
}
