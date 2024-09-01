#[derive(Debug, PartialEq)]
pub struct Hist {
    pub buckets: Vec<f64>,
    pub counts: Vec<f64>,
    pub len: usize,
}

impl Hist {
    pub fn hist(column: &Vec<f64>, width: f64, min: Option<f64>, max: Option<f64>) -> Self {
        if column.len() == 0 {
            return Hist {
                buckets: Vec::new(),
                counts: Vec::new(),
                len: 0,
            }
        }

        let mut auto_min = f64::MAX;
        let mut auto_max = f64::MIN;
        if min.is_none() || max.is_none() {
            for n in column {
                auto_min = auto_min.min(*n);
                auto_max = auto_max.max(*n);
            }
        }
        let min = min.unwrap_or(auto_min);
        let max = max.unwrap_or(auto_max);
        let steps = ((max - min) / width).ceil() as usize;
        let mut buckets = Vec::with_capacity(steps + 1);
        let mut counts = vec![0.0; steps + 1];
        for i in 0..steps + 1 {
            buckets.push(i as f64 * width + min);
        }

        for n in column {
            let n = n.min(max);
            let n = n.max(min);
            let bucket = ((n - min) / width + 0.5).floor() as usize;
            counts[bucket] += 1.0;
        }

        Self {
            buckets,
            counts,
            len: steps + 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hist_given_min_max() {
        let column = vec![
            -1.0,
            2.0,
            2.1,
            4.0,
        ];
        let width = 1.0;
        let min = Some(-2.0);
        let max = Some(2.0);

        let hist = Hist::hist(&column, width, min, max);

        let expected = Hist {
            buckets: vec![
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
            ],
            counts: vec![
                0.0,
                1.0,
                0.0,
                0.0,
                3.0,
            ],
            len: 5,
        };

        assert_eq!(hist, expected);
    }


    #[test]
    fn test_hist_no_min_max() {
        let column = vec![
            -1.0,
            2.0,
            2.1,
            4.0,
        ];
        let width = 2.0;
        let min = None;
        let max = None;

        let hist = Hist::hist(&column, width, min, max);

        let expected = Hist {
            buckets: vec![
                -1.0,
                1.0,
                3.0,
                5.0,
            ],
            counts: vec![
                1.0,
                0.0,
                2.0,
                1.0,
            ],
            len: 4,
        };

        assert_eq!(hist, expected);
    }
}
