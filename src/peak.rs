use crate::csv::parse_csv_line;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;

pub const CHANNELS: usize = 9;
pub const ACTIVE_SENSORS: usize = 8;
pub const PAIRS: usize = ACTIVE_SENSORS * (ACTIVE_SENSORS - 1) / 2;
pub const FEATURES: usize = ACTIVE_SENSORS * 2 + PAIRS * 8;
pub const OUTPUTS: usize = 14;
pub const MAX_ADC: f32 = 4095.0;

pub const LABELS: [&str; OUTPUTS] = [
    "Floral",
    "Soft Floral",
    "Floral Amber",
    "Amber",
    "Soft Amber",
    "Woody Amber",
    "Woods",
    "Mossy Woods",
    "Dry Woods",
    "Aromatic",
    "Citrus",
    "Water",
    "Green",
    "Fruity",
];

#[derive(Clone)]
pub struct PeakModel {
    pub weights: [[f32; FEATURES]; OUTPUTS],
    pub bias: [f32; OUTPUTS],
    pub hold_secs: f32,
}

#[derive(Clone)]
pub struct LiveFrame {
    pub row_index: usize,
    pub segment: String,
    pub source_sample_id: String,
    pub labels: [String; 3],
    pub target: [bool; OUTPUTS],
    pub elapsed_ms: u64,
    pub adc: [f32; CHANNELS],
}

#[derive(Clone)]
pub struct PeakStepOutput {
    pub frame: LiveFrame,
    pub held_peaks: [f32; ACTIVE_SENSORS],
    pub bins: [u8; ACTIVE_SENSORS],
    pub features: [f32; FEATURES],
    pub logits: [f32; OUTPUTS],
}

pub struct PeakRuntime {
    model: PeakModel,
    windows: Vec<VecDeque<f32>>,
    hold_rows: usize,
}

impl PeakRuntime {
    pub fn new(model: PeakModel, period_ms: u64) -> Self {
        let hold_rows = ((model.hold_secs * 1000.0) / period_ms.max(1) as f32)
            .round()
            .max(1.0) as usize;
        Self::with_hold_rows(model, hold_rows)
    }

    pub fn with_hold_rows(model: PeakModel, hold_rows: usize) -> Self {
        Self {
            model,
            windows: (0..ACTIVE_SENSORS)
                .map(|_| VecDeque::<f32>::with_capacity(hold_rows.max(1)))
                .collect(),
            hold_rows: hold_rows.max(1),
        }
    }

    pub fn hold_rows(&self) -> usize {
        self.hold_rows
    }

    pub fn hold_secs(&self) -> f32 {
        self.model.hold_secs
    }

    pub fn step(&mut self, frame: LiveFrame) -> PeakStepOutput {
        let mut held_peaks = [0.0_f32; ACTIVE_SENSORS];
        let mut bins = [0_u8; ACTIVE_SENSORS];

        for sensor in 0..ACTIVE_SENSORS {
            let window = &mut self.windows[sensor];
            window.push_back(frame.adc[sensor]);
            while window.len() > self.hold_rows {
                window.pop_front();
            }
            held_peaks[sensor] = window.iter().fold(0.0_f32, |acc, value| acc.max(*value));
            bins[sensor] = quantize_8(held_peaks[sensor]);
        }

        let features = pairwise_features(&bins);
        let logits = self.model.predict(&features);
        PeakStepOutput {
            frame,
            held_peaks,
            bins,
            features,
            logits,
        }
    }
}

impl PeakModel {
    pub fn predict(&self, features: &[f32; FEATURES]) -> [f32; OUTPUTS] {
        let mut logits = self.bias;
        for (label, logit) in logits.iter_mut().enumerate() {
            for (feature, value) in features.iter().enumerate() {
                *logit += self.weights[label][feature] * value;
            }
        }
        logits
    }
}

pub fn load_model(path: &Path) -> Result<PeakModel, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    if lines.next() != Some("NOSEKNOWS_PEAK_PAIR_READOUT_V1") {
        return Err(format!("{} is not a peak-pair readout model", path.display()).into());
    }

    let mut model = PeakModel {
        weights: [[0.0; FEATURES]; OUTPUTS],
        bias: [0.0; OUTPUTS],
        hold_secs: 8.0,
    };

    for line in lines {
        if let Some(value) = line.strip_prefix("hold_secs=") {
            model.hold_secs = value.parse()?;
        } else if let Some((label, value)) = line.strip_prefix("bias.").and_then(split_once_eq) {
            if let Some(index) = label_index(label) {
                model.bias[index] = value.parse()?;
            }
        } else if let Some((label, value)) = line.strip_prefix("weights.").and_then(split_once_eq) {
            if let Some(index) = label_index(label) {
                let weights = value
                    .split(',')
                    .map(str::parse::<f32>)
                    .collect::<Result<Vec<_>, _>>()?;
                if weights.len() != FEATURES {
                    return Err(format!(
                        "{} has {} weights for {label}, expected {FEATURES}",
                        path.display(),
                        weights.len()
                    )
                    .into());
                }
                model.weights[index].copy_from_slice(&weights);
            }
        }
    }

    Ok(model)
}

pub fn read_live_frames(path: &Path) -> Result<Vec<LiveFrame>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = lines.next().ok_or("live input CSV is empty")?;
    let header_fields = parse_csv_line(header);
    let index = |name: &str| -> Result<usize, Box<dyn std::error::Error>> {
        header_fields
            .iter()
            .position(|field| field == name)
            .ok_or_else(|| format!("{} missing column {name}", path.display()).into())
    };

    let label_indexes = [index("label_1")?, index("label_2")?, index("label_3")?];
    let elapsed_index = index("host_elapsed_ms")?;
    let segment_index = header_fields
        .iter()
        .position(|field| field == "stream_segment")
        .or_else(|| header_fields.iter().position(|field| field == "sample_id"));
    let source_sample_index = header_fields
        .iter()
        .position(|field| field == "source_sample_id")
        .or_else(|| header_fields.iter().position(|field| field == "sample_id"));
    let adc_indexes = [
        index("adc0")?,
        index("adc1")?,
        index("adc2")?,
        index("adc3")?,
        index("adc4")?,
        index("adc5")?,
        index("adc6")?,
        index("adc7")?,
        index("adc8")?,
    ];

    let mut rows = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_line(line);
        if fields.len() <= *adc_indexes.iter().max().expect("adc indexes") {
            continue;
        }

        let labels = [
            fields[label_indexes[0]].clone(),
            fields[label_indexes[1]].clone(),
            fields[label_indexes[2]].clone(),
        ];
        let mut target = [false; OUTPUTS];
        for label in &labels {
            if let Some(index) = label_index(label) {
                target[index] = true;
            }
        }

        let mut adc = [0.0_f32; CHANNELS];
        for (channel, field_index) in adc_indexes.iter().enumerate() {
            adc[channel] = fields[*field_index].parse::<f32>().unwrap_or(0.0);
        }

        rows.push(LiveFrame {
            row_index: rows.len(),
            segment: segment_index
                .and_then(|index| fields.get(index))
                .cloned()
                .unwrap_or_else(|| format!("row_{:010}", rows.len())),
            source_sample_id: source_sample_index
                .and_then(|index| fields.get(index))
                .cloned()
                .unwrap_or_else(|| fields.first().cloned().unwrap_or_default()),
            labels,
            target,
            elapsed_ms: fields[elapsed_index].parse::<u64>().unwrap_or(0),
            adc,
        });
    }

    Ok(rows)
}

pub fn median_period_ms(rows: &[LiveFrame]) -> u64 {
    let mut deltas = rows
        .windows(2)
        .filter_map(|pair| pair[1].elapsed_ms.checked_sub(pair[0].elapsed_ms))
        .filter(|delta| *delta > 0)
        .collect::<Vec<_>>();
    deltas.sort_unstable();
    deltas.get(deltas.len() / 2).copied().unwrap_or(100).max(1)
}

pub fn quantize_8(value: f32) -> u8 {
    ((value / MAX_ADC).clamp(0.0, 1.0) * 8.0).floor().min(7.0) as u8
}

pub fn pairwise_features(bins: &[u8; ACTIVE_SENSORS]) -> [f32; FEATURES] {
    let mut features = [0.0_f32; FEATURES];
    let mut cursor = 0;

    for bin in bins {
        features[cursor] = *bin as f32 / 7.0;
        cursor += 1;
    }
    for bin in bins {
        features[cursor] = if *bin >= 6 { 1.0 } else { 0.0 };
        cursor += 1;
    }

    for left in 0..ACTIVE_SENSORS {
        for right in (left + 1)..ACTIVE_SENSORS {
            let a = bins[left] as f32 / 7.0;
            let b = bins[right] as f32 / 7.0;
            features[cursor] = a.min(b);
            features[cursor + 1] = a.max(b);
            features[cursor + 2] = (a - b).abs();
            features[cursor + 3] = (a - b).max(0.0);
            features[cursor + 4] = (b - a).max(0.0);
            features[cursor + 5] = if bins[left] >= 5 && bins[right] >= 5 {
                1.0
            } else {
                0.0
            };
            features[cursor + 6] = if bins[left] >= 5 && bins[right] <= 1 {
                1.0
            } else {
                0.0
            };
            features[cursor + 7] = if bins[right] >= 5 && bins[left] <= 1 {
                1.0
            } else {
                0.0
            };
            cursor += 8;
        }
    }

    features
}

pub fn label_index(label: &str) -> Option<usize> {
    LABELS
        .iter()
        .position(|candidate| candidate.eq_ignore_ascii_case(label))
}

pub fn top_k(values: &[f32; OUTPUTS], k: usize) -> Vec<(usize, f32)> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| LABELS[a.0].cmp(LABELS[b.0]))
    });
    indexed.truncate(k);
    indexed
}

pub fn predicted_labels(logits: &[f32; OUTPUTS], gate_threshold: f32) -> Vec<usize> {
    top_k(logits, 3)
        .into_iter()
        .filter_map(|(label, score)| (score > gate_threshold).then_some(label))
        .collect()
}

pub fn expected_names(target: &[bool; OUTPUTS]) -> Vec<&'static str> {
    let labels = target
        .iter()
        .enumerate()
        .filter_map(|(index, active)| active.then_some(LABELS[index]))
        .collect::<Vec<_>>();
    if labels.is_empty() {
        vec!["No Scent"]
    } else {
        labels
    }
}

pub fn is_no_scent_target(target: &[bool; OUTPUTS]) -> bool {
    !target.iter().any(|value| *value)
}

fn split_once_eq(line: &str) -> Option<(&str, &str)> {
    line.split_once('=')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_8_caps_to_seven() {
        assert_eq!(quantize_8(0.0), 0);
        assert_eq!(quantize_8(MAX_ADC), 7);
        assert_eq!(quantize_8(MAX_ADC * 2.0), 7);
    }

    #[test]
    fn pairwise_feature_count_is_exact() {
        let bins = [0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(pairwise_features(&bins).len(), FEATURES);
    }
}
