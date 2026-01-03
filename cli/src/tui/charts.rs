use ratatui::prelude::{Buffer, Rect};
use ratatui::style::{Style, Stylize};
use ratatui::widgets::{Axis, Block, Chart, Dataset, GraphType, Widget};
use ratatui::{self, symbols};

#[derive(Debug)]
pub enum UiChart<'a> {
    ScatterPlot(ScatterPlot<'a>),
}

#[derive(Debug, Default)]
pub struct ScatterPlot<'a> {
    name: String,
    x_axis: String,
    y_axis: String,
    points: Vec<(f64, f64)>,
    x_bounds: (f64, f64),
    y_bounds: (f64, f64),
    style: Style,
    block: Option<Block<'a>>,
}

impl<'a> ScatterPlot<'a> {
    pub fn name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn x_axis(mut self, x_axis: String) -> Self {
        self.x_axis = x_axis;
        self
    }

    pub fn y_axis(mut self, y_axis: String) -> Self {
        self.y_axis = y_axis;
        self
    }

    pub fn points(mut self, points: Vec<(f64, f64)>) -> Self {
        self.points = points;
        self
    }

    pub fn x_bounds(mut self, x_bounds: (f64, f64)) -> Self {
        self.x_bounds = x_bounds;
        self
    }

    pub fn y_bounds(mut self, y_bounds: (f64, f64)) -> Self {
        self.y_bounds = y_bounds;
        self
    }

    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }

    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }
}

impl Widget for &ScatterPlot<'_> {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        let datasets = vec![Dataset::default()
            .name(self.name.as_str())
            .marker(symbols::Marker::Dot)
            .graph_type(GraphType::Scatter)
            .style(Style::default().red())
            .data(&self.points)];

        let x_axis = Axis::default()
            .title(self.x_axis.as_str().red())
            .style(self.style.white())
            .bounds([self.x_bounds.0, self.x_bounds.1])
            .labels(get_labels(&self.x_bounds));

        let y_axis = Axis::default()
            .title(self.y_axis.as_str().red())
            .style(self.style.white())
            .bounds([self.y_bounds.0, self.y_bounds.1])
            .labels(get_labels(&self.y_bounds));

        self.block.render(area, buf);
        Chart::new(datasets)
            .style(self.style)
            .x_axis(x_axis)
            .y_axis(y_axis)
            .render(area, buf);
    }
}

fn get_labels(bounds: &(f64, f64)) -> [String; 3] {
    let lower_bound = bounds.0;
    let upper_bound = bounds.1;
    let mid = (lower_bound + upper_bound) / 2.0;
    [
        lower_bound.to_string(),
        mid.to_string(),
        upper_bound.to_string(),
    ]
}
