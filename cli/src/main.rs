mod agg;
mod hist;
mod pca;
mod util;
mod plot;
mod tui;

use agg::AggArgs;
use clap::{Parser, Subcommand};
use hist::HistArgs;
use pca::PcaArgs;
use plot::PlotArgs;

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
    Pca(PcaArgs),

    /// Aggregate data
    #[command(version, about, long_about = None)]
    Agg(AggArgs),

    /// Histogram
    #[command(version, about, long_about = None)]
    Hist(HistArgs),

    /// Plot
    #[command(version, about, long_about = None)]
    Plot(PlotArgs),
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    match cli.tool {
        Tool::Pca(args) => pca::pca_main(args).await,
        Tool::Agg(args) => agg::agg_main(args).await,
        Tool::Hist(args) => hist::hist_main(args).await,
        Tool::Plot(args) => plot::plot_main(args).await,
    }
}
