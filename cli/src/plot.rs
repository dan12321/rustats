use std::{io::BufRead, path::PathBuf};

use anyhow::Result;
use clap::{Args, ValueEnum};

use crate::util::DataType;
use crate::tui;

/// Plot
#[derive(Debug, Args)]
#[command(version, about, long_about = None)]
pub struct PlotArgs {
    /// The name of the column to put on the x axis
    x: String,
    /// The name of the column to put on the y axis
    y: String,
    /// The format of the file
    #[arg(value_enum, short, long)]
    datatype: Option<DataType>,
    /// File containing data
    filename: Option<PathBuf>,
}

#[derive(Debug, ValueEnum, Clone, Copy)]
pub enum PlotType {
    /// A scatter plot
    Scatter,
}

pub async fn plot_main(args: PlotArgs) {
    // UI should run on it's own thread to avoid becoming unresponsive
    let ui_thread = tokio::task::spawn_blocking(|| {
        tui::app::run()
    });

    // Should close on sigterm instead
    ui_thread.await.unwrap().unwrap();
}
