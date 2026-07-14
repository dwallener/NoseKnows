use crate::csv::parse_csv_line;
use crate::peak::{label_index, top_k, CHANNELS, LABELS, OUTPUTS};
use std::collections::VecDeque;
use std::fs;
use std::path::Path;

pub const ACTIVE_SENSORS: usize = 8;
pub const LOOKBACK_SECS: usize = 8;
pub const FEATURES: usize = ACTIVE_SENSORS * LOOKBACK_SECS;
pub const MAX_ADC: f32 = 4095.0;

#[derive(Clone)]
pub struct GridModel {
    pub weights: [[f32; FEATURES]; OUTPUTS],
    pub bias: [f32; OUTPUTS],
    pub lookback_secs: usize,
}

#[derive(Clone)]
pub struct RawSample {
    pub id: String,
    pub labels: [String; 3],
    pub target: [bool; OUTPUTS],
    pub elapsed_ms: Vec<u64>,
    pub rows: Vec<[f32; CHANNELS]>,
}

pub struct GridRuntime {
    model: GridModel,
    buckets: VecDeque<[u8; ACTIVE_SENSORS]>,
    current_second: Option<u64>,
    current_max: [f32; ACTIVE_SENSORS],
}

impl GridRuntime {
    pub fn new(model: GridModel) -> Self {
        let lookback_secs = model.lookback_secs.max(1);
        Self {
            model: GridModel {
                lookback_secs,
                ..model
            },
            buckets: VecDeque::with_capacity(lookback_secs),
            current_second: None,
            current_max: [0.0; ACTIVE_SENSORS],
        }
    }

    pub fn step(
        &mut self,
        elapsed_ms: u64,
        adc: &[f32; CHANNELS],
    ) -> ([u8; FEATURES], [f32; OUTPUTS]) {
        let second = elapsed_ms / 1000;
        match self.current_second {
            None => self.current_second = Some(second),
            Some(current) if second != current => {
                self.push_current_bucket();
                self.current_second = Some(second);
                self.current_max = [0.0; ACTIVE_SENSORS];
            }
            _ => {}
        }
        for sensor in 0..ACTIVE_SENSORS {
            self.current_max[sensor] = self.current_max[sensor].max(adc[sensor]);
        }
        let features = self.features_with_current();
        let logits = self.model.predict_bins(&features);
        (features, logits)
    }

    fn push_current_bucket(&mut self) {
        let mut bucket = [0_u8; ACTIVE_SENSORS];
        for sensor in 0..ACTIVE_SENSORS {
            bucket[sensor] = quantize_8(self.current_max[sensor]);
        }
        self.buckets.push_back(bucket);
        while self.buckets.len() > self.model.lookback_secs {
            self.buckets.pop_front();
        }
    }

    fn features_with_current(&self) -> [u8; FEATURES] {
        let mut features = [0_u8; FEATURES];
        let mut buckets = self.buckets.iter().copied().collect::<Vec<_>>();
        let mut current = [0_u8; ACTIVE_SENSORS];
        for sensor in 0..ACTIVE_SENSORS {
            current[sensor] = quantize_8(self.current_max[sensor]);
        }
        buckets.push(current);
        if buckets.len() > self.model.lookback_secs {
            buckets.remove(0);
        }

        let start = self.model.lookback_secs.saturating_sub(buckets.len());
        for (offset, bucket) in buckets.iter().enumerate() {
            for sensor in 0..ACTIVE_SENSORS {
                features[sensor * LOOKBACK_SECS + start + offset] = bucket[sensor];
            }
        }
        features
    }
}

impl GridModel {
    pub fn new(lookback_secs: usize) -> Self {
        Self {
            weights: [[0.0; FEATURES]; OUTPUTS],
            bias: [0.0; OUTPUTS],
            lookback_secs: lookback_secs.max(1),
        }
    }

    pub fn predict(&self, features: &[f32; FEATURES]) -> [f32; OUTPUTS] {
        let mut logits = self.bias;
        for label in 0..OUTPUTS {
            for feature in 0..FEATURES {
                logits[label] += self.weights[label][feature] * features[feature];
            }
        }
        logits
    }

    pub fn predict_bins(&self, features: &[u8; FEATURES]) -> [f32; OUTPUTS] {
        let normalized = normalize_bins(features);
        self.predict(&normalized)
    }
}

pub fn encode_sample(sample: &RawSample, lookback_secs: usize) -> [u8; FEATURES] {
    let second_bins = sample_second_bins(sample);
    encode_from_second_bins(&second_bins, second_bins.len(), lookback_secs)
}

pub fn encode_from_second_bins(
    second_bins: &[[u8; ACTIVE_SENSORS]],
    end_exclusive: usize,
    lookback_secs: usize,
) -> [u8; FEATURES] {
    let mut features = [0_u8; FEATURES];
    let end = end_exclusive.min(second_bins.len());
    let start_index = end.saturating_sub(lookback_secs);
    let selected = if start_index < end {
        &second_bins[start_index..end]
    } else {
        &[]
    };
    let start = lookback_secs.saturating_sub(selected.len());
    for (offset, bucket) in selected.iter().enumerate() {
        for sensor in 0..ACTIVE_SENSORS {
            features[sensor * LOOKBACK_SECS + start + offset] = bucket[sensor];
        }
    }
    features
}

pub fn sample_second_grid_windows(
    sample: &RawSample,
    lookback_secs: usize,
) -> Vec<(usize, [u8; FEATURES])> {
    let second_bins = sample_second_bins(sample);
    (1..=second_bins.len())
        .map(|end| {
            (
                end - 1,
                encode_from_second_bins(&second_bins, end, lookback_secs),
            )
        })
        .collect()
}

pub fn normalize_bins(features: &[u8; FEATURES]) -> [f32; FEATURES] {
    let mut normalized = [0.0_f32; FEATURES];
    for feature in 0..FEATURES {
        normalized[feature] = features[feature] as f32 / 7.0;
    }
    normalized
}

pub fn is_one_note_or_no_scent(sample: &RawSample) -> bool {
    sample.target.iter().filter(|value| **value).count() <= 1
}

pub fn load_samples(data_dir: &Path) -> Result<Vec<RawSample>, Box<dyn std::error::Error>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(data_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) == Some("csv") {
            paths.push(path);
        }
    }
    paths.sort();

    let mut samples = Vec::new();
    for path in paths {
        if let Some(sample) = load_sample(&path)? {
            samples.push(sample);
        }
    }
    Ok(samples)
}

pub fn save_model(path: &Path, model: &GridModel) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut text = String::new();
    text.push_str("NOSEKNOWS_GRID8_READOUT_V1\n");
    text.push_str(&format!("active_sensors={ACTIVE_SENSORS}\n"));
    text.push_str(&format!("lookback_secs={}\n", model.lookback_secs));
    text.push_str(&format!("features={FEATURES}\n"));
    text.push_str(&format!("outputs={OUTPUTS}\n"));
    text.push_str(&format!("labels={}\n", LABELS.join(",")));
    for label in 0..OUTPUTS {
        let weights = model.weights[label]
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(",");
        text.push_str(&format!(
            "bias.{}={:.6}\n",
            LABELS[label], model.bias[label]
        ));
        text.push_str(&format!("weights.{}={weights}\n", LABELS[label]));
    }
    fs::write(path, text)?;
    Ok(())
}

pub fn load_model(path: &Path) -> Result<GridModel, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    if lines.next() != Some("NOSEKNOWS_GRID8_READOUT_V1") {
        return Err(format!("{} is not a grid8 model", path.display()).into());
    }
    let mut model = GridModel::new(LOOKBACK_SECS);
    for line in lines {
        if let Some(value) = line.strip_prefix("lookback_secs=") {
            model.lookback_secs = value.parse::<usize>()?.max(1);
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

pub fn quantize_8(value: f32) -> u8 {
    ((value / MAX_ADC).clamp(0.0, 1.0) * 8.0).floor().min(7.0) as u8
}

pub fn top_label_names(logits: &[f32; OUTPUTS]) -> String {
    top_k(logits, 3)
        .into_iter()
        .map(|(label, score)| format!("{} {score:.3}", LABELS[label]))
        .collect::<Vec<_>>()
        .join(", ")
}

fn sample_second_bins(sample: &RawSample) -> Vec<[u8; ACTIVE_SENSORS]> {
    let mut buckets = Vec::new();
    let mut current_second = None;
    let mut current_max = [0.0_f32; ACTIVE_SENSORS];

    for (row_index, row) in sample.rows.iter().enumerate() {
        let second = sample.elapsed_ms.get(row_index).copied().unwrap_or(0) / 1000;
        match current_second {
            None => current_second = Some(second),
            Some(current) if current != second => {
                buckets.push(quantized_bucket(&current_max));
                current_second = Some(second);
                current_max = [0.0; ACTIVE_SENSORS];
            }
            _ => {}
        }
        for sensor in 0..ACTIVE_SENSORS {
            current_max[sensor] = current_max[sensor].max(row[sensor]);
        }
    }

    if current_second.is_some() {
        buckets.push(quantized_bucket(&current_max));
    }
    buckets
}

fn quantized_bucket(values: &[f32; ACTIVE_SENSORS]) -> [u8; ACTIVE_SENSORS] {
    let mut bucket = [0_u8; ACTIVE_SENSORS];
    for sensor in 0..ACTIVE_SENSORS {
        bucket[sensor] = quantize_8(values[sensor]);
    }
    bucket
}

fn load_sample(path: &Path) -> Result<Option<RawSample>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return Ok(None);
    };
    let header_fields = parse_csv_line(header);
    let index = |name: &str| -> Result<usize, Box<dyn std::error::Error>> {
        header_fields
            .iter()
            .position(|field| field == name)
            .ok_or_else(|| format!("{} missing column {name}", path.display()).into())
    };

    let sample_id_index = index("sample_id")?;
    let elapsed_index = index("host_elapsed_ms")?;
    let label_indexes = [index("label_1")?, index("label_2")?, index("label_3")?];
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

    let mut id = String::new();
    let mut labels = [String::new(), String::new(), String::new()];
    let mut elapsed_ms = Vec::new();
    let mut rows = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_line(line);
        if fields.len() <= *adc_indexes.iter().max().expect("adc indexes") {
            continue;
        }
        if id.is_empty() {
            id = fields[sample_id_index].clone();
            labels = [
                fields[label_indexes[0]].clone(),
                fields[label_indexes[1]].clone(),
                fields[label_indexes[2]].clone(),
            ];
        }
        elapsed_ms.push(fields[elapsed_index].parse::<u64>().unwrap_or(0));
        let mut row = [0.0_f32; CHANNELS];
        for (channel, field_index) in adc_indexes.iter().enumerate() {
            row[channel] = fields[*field_index].parse::<f32>().unwrap_or(0.0);
        }
        rows.push(row);
    }

    if id.is_empty() {
        return Ok(None);
    }

    let mut target = [false; OUTPUTS];
    for label in &labels {
        if let Some(index) = label_index(label) {
            target[index] = true;
        }
    }
    Ok(Some(RawSample {
        id,
        labels,
        target,
        elapsed_ms,
        rows,
    }))
}

fn split_once_eq(line: &str) -> Option<(&str, &str)> {
    line.split_once('=')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_shape_is_8_by_8() {
        assert_eq!(FEATURES, 64);
    }

    #[test]
    fn normalization_maps_seven_to_one() {
        let bins = [7_u8; FEATURES];
        let normalized = normalize_bins(&bins);
        assert!(normalized.iter().all(|value| (*value - 1.0).abs() < 0.0001));
    }
}
