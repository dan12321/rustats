use std::{fmt::Debug, time::Duration};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use tokio::sync::mpsc::Sender;

use crate::plot::PlotArgs;

/// Events to send to the UI
#[derive(Debug)]
pub enum UiEvent {
    ChartData(PlotChartData),
    Exit,
}

#[derive(Debug)]
pub struct PlotChartData {
    pub args: PlotArgs,
    pub x_bounds: (f64, f64),
    pub y_bounds: (f64, f64),
    pub points: Vec<(f64, f64)>,
}

/// Event from the terminal
#[derive(Debug)]
pub enum TermEvent {
    Quit,
}

pub fn start_listening(sender: Sender<TermEvent>) {
    // Automatically drop if last receiver is closed/dropped.
    while !sender.is_closed() {
        match read_event() {
            Ok(e) => {
                if let Some(ev) = e {
                    if let Some(err) = sender.blocking_send(ev).err() {
                        eprintln!("Failed to send term event: {err}");
                    };
                }
            }
            Err(e) => eprintln!("Failed to read event: {e}"),
        }
    }
}

fn read_event() -> std::io::Result<Option<TermEvent>> {
    // If we're shutting down we don't want to get stuck on read
    if !event::poll(Duration::from_millis(10))? {
        return Ok(None);
    }
    match event::read()? {
        Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
            Ok(get_key_press(key_event))
        }
        _ => Ok(None),
    }
}

fn get_key_press(key_event: KeyEvent) -> Option<TermEvent> {
    match key_event.code {
        KeyCode::Char('q') => Some(TermEvent::Quit),
        _ => None,
    }
}
