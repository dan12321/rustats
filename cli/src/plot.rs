use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, ValueEnum};
use polars::io::SerReader;
use polars::prelude::*;
use tokio::sync::mpsc::Sender;

use crate::tui::ui::{AsyncReqBody, AsyncRes, AsyncResBody, GetChartDataReq, UiEvent, UiState};
use crate::tui::{self, ui::AsyncReq};
use crate::util::DataType;
use tui::ui::{PlotChartData, UiCommand};

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
    let (ui_event_sender, mut ui_event_receiver) = tokio::sync::mpsc::channel::<UiEvent>(64);
    let (ui_sender, ui_receiver) = tokio::sync::mpsc::channel::<UiCommand>(64);
    // UI is really the main thread and should lead the process, tokio_main is
    // the io and async task handler.
    let ui_thread =
        tokio::task::spawn_blocking(|| tui::ui::run(args, ui_receiver, ui_event_sender));

    let term_ui_sender = ui_sender.clone();
    let term_event_thread =
        tokio::task::spawn_blocking(|| tui::term::start_listening(term_ui_sender));

    loop {
        match ui_event_receiver.recv().await.unwrap() {
            UiEvent::StateChange(state) => match state {
                UiState::Exiting => break,
                _ => continue,
            },
            UiEvent::AsyncReq(req) => handle_req(req, ui_sender.clone()),
        }
    }

    let _ = tokio::join!(ui_thread, term_event_thread);
}

fn handle_req(req: AsyncReq, res_sender: Sender<UiCommand>) {
    match req.body {
        AsyncReqBody::GetChartData(_) => {
            tokio::task::spawn_blocking(|| handle_blocking_req(req, res_sender));
        }
    };
}

fn handle_blocking_req(req: AsyncReq, res_sender: Sender<UiCommand>) {
    let body: AsyncResBody = match req.body {
        AsyncReqBody::GetChartData(rb) => AsyncResBody::PlotChartData(get_chart_data(rb)),
    };
    res_sender
        .blocking_send(UiCommand::AsyncRes(AsyncRes { id: req.id, body }))
        .unwrap();
}

fn get_chart_data(rb: GetChartDataReq) -> Result<PlotChartData> {
    let data = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(rb.filename))
        .and_then(|c| c.finish())?;
    let data = data
        .lazy()
        .select([
            col(&rb.x),
            col(&rb.y),
            col(&rb.x).min().alias("x_min"),
            col(&rb.x).max().alias("x_max"),
            col(&rb.y).min().alias("y_min"),
            col(&rb.y).max().alias("y_max"),
        ])
        .collect()?;
    let x_min = get_f64_scalar(&data, "x_min");
    let x_max = get_f64_scalar(&data, "x_max");
    let y_min = get_f64_scalar(&data, "y_min");
    let y_max = get_f64_scalar(&data, "y_max");
    let arr = data.to_ndarray::<Float64Type>(IndexOrder::C).unwrap();
    let arr = arr.rows().into_iter().map(|r| (r[0], r[1])).collect();
    Ok(PlotChartData {
        x_bounds: (x_min.floor(), x_max.ceil()),
        y_bounds: (y_min.floor(), y_max.ceil()),
        points: arr,
    })
}

fn get_f64_scalar(data: &DataFrame, col: &str) -> f64 {
    data.column(col)
        .and_then(|v| v.f64())
        .map(|v| v.first())
        .unwrap()
        .unwrap()
}
