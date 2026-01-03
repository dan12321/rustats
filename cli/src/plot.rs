use std::{io::BufRead, path::PathBuf};

use anyhow::Result;
use clap::{Args, ValueEnum};
use ndarray::Axis;
use polars::frame::row::Row;
use polars::io::SerReader;
use polars::prelude::*;

use crate::tui::{self, events::PlotChartData};
use crate::util::DataType;
use tui::events::{TermEvent, UiEvent};

/// Plot
#[derive(Debug, Args)]
#[command(version, about, long_about = None)]
pub struct PlotArgs {
    /// The name of the column to put on the x axis
    pub x: String,
    /// The name of the column to put on the y axis
    pub y: String,
    /// The format of the file
    #[arg(value_enum, short, long)]
    pub datatype: Option<DataType>,
    /// File containing data
    pub filename: Option<PathBuf>,
}

#[derive(Debug, ValueEnum, Clone, Copy)]
pub enum PlotType {
    /// A scatter plot
    Scatter,
}

pub async fn plot_main(args: PlotArgs) {
    let (ui_sender, ui_receiver) = tokio::sync::mpsc::channel::<UiEvent>(64);
    // UI should run on it's own thread to avoid becoming unresponsive
    let ui_thread = tokio::task::spawn_blocking(|| tui::ui::run(ui_receiver));

    let (term_event_sender, mut term_event_receiver) = tokio::sync::mpsc::channel::<TermEvent>(64);
    let term_event_thread =
        tokio::task::spawn_blocking(|| tui::events::start_listening(term_event_sender));
    if let Some(filename) = &args.filename {
        let csv_result = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(filename.clone()))
            .and_then(|c| c.finish());
        let data = match csv_result {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to open csv: {e}");
                return;
            }
        };
        let data = data
            .lazy()
            .select([
                col(&args.x),
                col(&args.y),
                col(&args.x).min().alias("x_min"),
                col(&args.x).max().alias("x_max"),
                col(&args.y).min().alias("y_min"),
                col(&args.y).max().alias("y_max"),
            ])
            .collect();
        let data = match data {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to parse csv: {e}");
                return;
            }
        };
        let x_min = data.column("x_min")
            .and_then(|v| v.f64())
            .map(|v| v.first()).unwrap().unwrap();
        let x_max = data.column("x_max")
            .and_then(|v| v.f64())
            .map(|v| v.first()).unwrap().unwrap();
        let y_min = data.column("y_min")
            .and_then(|v| v.f64())
            .map(|v| v.first()).unwrap().unwrap();
        let y_max = data.column("y_max")
            .and_then(|v| v.f64())
            .map(|v| v.first()).unwrap().unwrap();
        let arr = data.to_ndarray::<Float64Type>(IndexOrder::C).unwrap();
        let arr = arr.rows().into_iter()
            .map(|r| (r[0], r[1]))
            .collect();
        ui_sender.send(UiEvent::ChartData(PlotChartData {
            args,
            x_bounds: (x_min.floor(), x_max.ceil()),
            y_bounds: (y_min.floor(), y_max.ceil()),
            points: arr,
        })).await.unwrap();
    }

    let mut exit = false;
    while !exit && !ui_sender.is_closed() {
        match term_event_receiver.recv().await {
            Some(TermEvent::Quit) => {
                ui_sender.send(UiEvent::Exit).await.unwrap();
                term_event_receiver.close();
                exit = true;
            }
            None => {
                ui_sender.send(UiEvent::Exit).await.unwrap();
                term_event_receiver.close();
                exit = true
            }
        }
    }

    let _ = tokio::join!(ui_thread, term_event_thread);
}
