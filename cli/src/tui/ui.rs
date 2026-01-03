use std::io;

use ratatui;
use ratatui::prelude::{Buffer, Rect};
use ratatui::widgets::{Block, Widget};
use ratatui::{DefaultTerminal, Frame};
use tokio::sync::mpsc::Receiver;

use crate::tui::charts::{ScatterPlot, UiChart};
use crate::tui::events::PlotChartData;

use super::events::UiEvent;

#[derive(Debug)]
pub struct Ui<'a> {
    rx: Receiver<UiEvent>,
    state: UiState,
    chart: Option<UiChart<'a>>,
}

#[derive(Debug)]
enum UiState {
    New,
    Run,
    Exiting,
    Exit,
}

pub fn run(rx: Receiver<UiEvent>) -> io::Result<()> {
    let mut terminal = ratatui::init();
    let app_result = Ui::new(rx).run(&mut terminal);
    ratatui::restore();
    app_result
}

impl Ui<'_> {
    pub fn new(rx: Receiver<UiEvent>) -> Self {
        Ui {
            rx,
            state: UiState::New,
            chart: None,
        }
    }

    /// runs the application's main loop until the user quits
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        self.state = UiState::Run;
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
        let n = self.rx.blocking_recv_many(&mut events, events_cap);
        for event in events {
            match event {
                UiEvent::Exit => self.exit(),
                UiEvent::ChartData(data) => self.handle_chart_data(data),
            }
        }
        if n == 0 && matches!(self.state, UiState::Exiting) {
            self.state = UiState::Exit;
        }
    }

    fn exit(&mut self) {
        self.rx.close();
        self.state = UiState::Exiting;
    }

    fn handle_chart_data(&mut self, data: PlotChartData) {
        let plot = ScatterPlot::default()
            .y_axis(data.args.y)
            .x_axis(data.args.x)
            .x_bounds(data.x_bounds)
            .y_bounds(data.y_bounds)
            .points(data.points);
        self.chart = Some(UiChart::ScatterPlot(plot));
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
