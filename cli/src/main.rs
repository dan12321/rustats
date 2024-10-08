mod agg;
mod hist;
mod pca;
mod util;

use agg::AggArgs;
use clap::{Parser, Subcommand};
use hist::HistArgs;
use pca::PcaArgs;

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

    /// Aggregate data
    #[command(version, about, long_about = None)]
    Agg(AggArgs),

    /// Histogram
    #[command(version, about, long_about = None)]
    Hist(HistArgs),
}

fn main() {
    let cli = Cli::parse();
    match cli.tool {
        Tool::PCA(args) => pca::pca_main(args),
        Tool::Agg(args) => agg::agg_main(args),
        Tool::Hist(args) => hist::hist_main(args),
    }
}
