use std::io;
use std::path::PathBuf;

use anyhow::Result;
use ratatui;
use ratatui::prelude::{Buffer, Rect};
use ratatui::widgets::{Block, Widget};
use ratatui::{DefaultTerminal, Frame};
use tokio::sync::mpsc::{Receiver, Sender};

use crate::plot::PlotArgs;
use crate::tui::charts::{ScatterPlot, UiChart};
use crate::tui::term::TermEvent;

#[derive(Debug)]
pub struct Ui<'a> {
    rx: Receiver<UiCommand>,
    event_sender: Sender<UiEvent>,
    state: UiState,
    chart: Option<UiChart<'a>>,
    args: PlotArgs,
    max_req_id: usize,
    last_chart_data_id: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub enum UiState {
    New,
    Run,
    Exiting,
    Exit,
}

/// Commands to send to the UI
#[derive(Debug)]
pub enum UiCommand {
    AsyncRes(AsyncRes),
    TermEvent(TermEvent),
}

pub enum UiEvent {
    StateChange(UiState),
    AsyncReq(AsyncReq),
}

/// Requests from the UI to tokio thread
pub struct AsyncReq {
    pub id: usize,
    pub body: AsyncReqBody,
}

pub enum AsyncReqBody {
    GetChartData(GetChartDataReq),
}

pub struct GetChartDataReq {
    pub filename: PathBuf,
    pub x: String,
    pub y: String,
}

#[derive(Debug)]
pub struct AsyncRes {
    pub id: usize,
    pub body: AsyncResBody,
}

#[derive(Debug)]
pub enum AsyncResBody {
    PlotChartData(Result<PlotChartData>),
}

#[derive(Debug)]
pub struct PlotChartData {
    pub x_bounds: (f64, f64),
    pub y_bounds: (f64, f64),
    pub points: Vec<(f64, f64)>,
}

pub fn run(
    args: PlotArgs,
    rx: Receiver<UiCommand>,
    event_sender: Sender<UiEvent>,
) -> io::Result<()> {
    let mut terminal = ratatui::init();
    let app_result = Ui::new(args, rx, event_sender).run(&mut terminal);
    ratatui::restore();
    app_result
}

impl Ui<'_> {
    pub fn new(args: PlotArgs, rx: Receiver<UiCommand>, event_sender: Sender<UiEvent>) -> Self {
        Ui {
            rx,
            event_sender,
            state: UiState::New,
            chart: None,
            args,
            max_req_id: 0,
            last_chart_data_id: None,
        }
    }

    /// runs the application's main loop until the user quits
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        self.change_state(UiState::Run);
        if let Some(filename) = self.args.filename.clone() {
            self.send_async(AsyncReqBody::GetChartData(GetChartDataReq {
                filename,
                x: self.args.x.clone(),
                y: self.args.y.clone(),
            }));
        }
        while !matches!(self.state, UiState::Exit) {
            terminal.draw(|frame| self.draw(frame))?;
            self.handle_events();
        }
        Ok(())
    }

    fn draw(&self, frame: &mut Frame) {
        frame.render_widget(self, frame.area());
    }

    fn handle_events(&mut self) {
        let events_cap = 64;
        let mut events = Vec::with_capacity(events_cap);
        let _ = self.rx.blocking_recv_many(&mut events, events_cap);
        for event in events {
            match event {
                UiCommand::TermEvent(te) => self.handle_term_event(te),
                UiCommand::AsyncRes(res) => self.handle_async_res(res),
            }
        }
        if matches!(self.state, UiState::Exiting) {
            // TODO!: Give other threads a chance to finish before closing
            // then close on finish/force exit
            self.change_state(UiState::Exit);
            self.rx.close();
        }
    }

    fn handle_term_event(&mut self, te: TermEvent) {
        match te {
            TermEvent::Quit => self.exit(),
        }
    }

    fn exit(&mut self) {
        self.change_state(UiState::Exiting);
    }

    fn send_async(&mut self, body: AsyncReqBody) {
        let id = self.max_req_id;
        self.max_req_id += 1;
        self.event_sender
            .blocking_send(UiEvent::AsyncReq(AsyncReq { id, body }))
            .unwrap();
    }

    fn handle_async_res(&mut self, res: AsyncRes) {
        match res.body {
            AsyncResBody::PlotChartData(d) => {
                if let Some(id) = self.last_chart_data_id {
                    if id >= res.id {
                        return;
                    }
                }
                self.last_chart_data_id = Some(res.id);
                self.handle_chart_data(d.unwrap());
            }
        }
    }

    fn handle_chart_data(&mut self, data: PlotChartData) {
        let plot = ScatterPlot::default()
            // TODO!: axis should probably be shared. Maybe a Pin<String>
            .y_axis(self.args.y.clone())
            .x_axis(self.args.x.clone())
            .x_bounds(data.x_bounds)
            .y_bounds(data.y_bounds)
            .points(data.points);
        self.chart = Some(UiChart::ScatterPlot(plot));
    }

    fn change_state(&mut self, state: UiState) {
        self.state = state;
        self.event_sender
            .blocking_send(UiEvent::StateChange(state))
            .unwrap();
    }
}

impl Widget for &Ui<'_> {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        let block = Block::bordered().title_top("rustats");
        if let Some(chart) = self.chart.as_ref() {
            match chart {
                UiChart::ScatterPlot(p) => {
                    let inner = block.inner(area);
                    p.render(inner, buf);
                }
            }
        } else {
            block.render(area, buf);
        }
    }
}
