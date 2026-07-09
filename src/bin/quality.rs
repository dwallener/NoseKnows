use std::cmp::Ordering;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const CHANNELS: usize = 9;
const FEATURE_GROUPS: usize = 5;
const FEATURES: usize = CHANNELS * FEATURE_GROUPS;
const LABELS: [&str; 14] = [
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
const NO_SCENT_LABEL: &str = "No Scent";

struct Config {
    data_dir: PathBuf,
    epochs: usize,
    learning_rate: f32,
    validation_fraction: f32,
    seed: u64,
}

#[derive(Clone)]
struct Sample {
    id: String,
    labels: [String; 3],
    target: [f32; LABELS.len()],
    rows: Vec<[f32; CHANNELS]>,
}

struct Baseline {
    weights: Vec<f32>,
    bias: [f32; LABELS.len()],
}

#[derive(Default)]
struct Metrics {
    loss: f32,
    primary_at_1: f32,
    any_at_3: f32,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("quality error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let samples = load_samples(&config.data_dir)?;
    if samples.len() < 2 {
        return Err("quality run needs at least two usable samples".into());
    }

    let features = samples
        .iter()
        .map(|sample| extract_features(&sample.rows))
        .collect::<Vec<_>>();
    let (train_indexes, validation_indexes) =
        split_indexes(samples.len(), config.validation_fraction, config.seed);

    println!(
        "Loaded {} sample(s): {} train, {} validation",
        samples.len(),
        train_indexes.len(),
        validation_indexes.len()
    );

    let mut model = Baseline::new();
    for epoch in 1..=config.epochs {
        for sample_index in &train_indexes {
            model.train_one(
                &features[*sample_index],
                &samples[*sample_index].target,
                config.learning_rate,
            );
        }

        if epoch == 1 || epoch == config.epochs || epoch % 25 == 0 {
            let train = evaluate(&model, &samples, &features, &train_indexes);
            let validation = evaluate(&model, &samples, &features, &validation_indexes);
            println!(
                "epoch {epoch:>4} train loss {:.5} p@1 {:.2}% any@3 {:.2}% | val loss {:.5} p@1 {:.2}% any@3 {:.2}%",
                train.loss,
                train.primary_at_1 * 100.0,
                train.any_at_3 * 100.0,
                validation.loss,
                validation.primary_at_1 * 100.0,
                validation.any_at_3 * 100.0
            );
        }
    }

    println!();
    println!("Validation examples:");
    for sample_index in validation_indexes.iter().take(10) {
        let logits = model.predict(&features[*sample_index]);
        let top = top_k(&logits, 3);
        let sample = &samples[*sample_index];
        println!(
            "{} labels=[{}, {}, {}] predicted=[{}, {}, {}]",
            sample.id,
            sample.labels[0],
            sample.labels[1],
            sample.labels[2],
            LABELS[top[0].0],
            LABELS[top[1].0],
            LABELS[top[2].0]
        );
    }

    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut data_dir = PathBuf::from("data/raw");
    let mut epochs = 250;
    let mut learning_rate = 0.25;
    let mut validation_fraction = 0.2;
    let mut seed = 0x7157_2026_u64;

    let args: Vec<String> = env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--data" => {
                index += 1;
                data_dir = PathBuf::from(args.get(index).ok_or("--data requires a path")?);
            }
            "--epochs" => {
                index += 1;
                epochs = args
                    .get(index)
                    .ok_or("--epochs requires a value")?
                    .parse()?;
            }
            "--lr" => {
                index += 1;
                learning_rate = args.get(index).ok_or("--lr requires a value")?.parse()?;
            }
            "--validation" => {
                index += 1;
                validation_fraction = args
                    .get(index)
                    .ok_or("--validation requires a fraction")?
                    .parse()?;
            }
            "--seed" => {
                index += 1;
                seed = args.get(index).ok_or("--seed requires a value")?.parse()?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin quality -- [--data data/raw] [--epochs 250] [--validation 0.2]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        index += 1;
    }

    if !(0.0..0.9).contains(&validation_fraction) {
        return Err("--validation must be >= 0.0 and < 0.9".into());
    }

    Ok(Config {
        data_dir,
        epochs,
        learning_rate,
        validation_fraction,
        seed,
    })
}

fn load_samples(data_dir: &Path) -> Result<Vec<Sample>, Box<dyn std::error::Error>> {
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

fn load_sample(path: &Path) -> Result<Option<Sample>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = match lines.next() {
        Some(header) => header,
        None => return Ok(None),
    };
    let header_fields = parse_csv_line(header);
    let index = |name: &str| -> Result<usize, Box<dyn std::error::Error>> {
        header_fields
            .iter()
            .position(|field| field == name)
            .ok_or_else(|| format!("{} missing column {name}", path.display()).into())
    };

    let sample_id_index = index("sample_id")?;
    let label_1_index = index("label_1")?;
    let label_2_index = index("label_2")?;
    let label_3_index = index("label_3")?;
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
                fields[label_1_index].clone(),
                fields[label_2_index].clone(),
                fields[label_3_index].clone(),
            ];
        }

        let mut row = [0.0_f32; CHANNELS];
        for (channel, field_index) in adc_indexes.iter().enumerate() {
            row[channel] = fields[*field_index].parse::<f32>()? / 4095.0;
        }
        rows.push(row);
    }

    if rows.is_empty() {
        return Ok(None);
    }

    let target = target_from_labels(&labels)?;
    Ok(Some(Sample {
        id,
        labels,
        target,
        rows,
    }))
}

fn extract_features(rows: &[[f32; CHANNELS]]) -> [f32; FEATURES] {
    let mut features = [0.0_f32; FEATURES];
    let mut min = [f32::INFINITY; CHANNELS];
    let mut max = [f32::NEG_INFINITY; CHANNELS];
    let mut mean = [0.0_f32; CHANNELS];

    for row in rows {
        for channel in 0..CHANNELS {
            mean[channel] += row[channel];
            min[channel] = min[channel].min(row[channel]);
            max[channel] = max[channel].max(row[channel]);
        }
    }

    for channel in 0..CHANNELS {
        mean[channel] /= rows.len() as f32;
        let first = rows.first().expect("rows")[channel];
        let last = rows.last().expect("rows")[channel];
        let range = max[channel] - min[channel];
        let delta = last - first;

        features[channel] = mean[channel];
        features[CHANNELS + channel] = max[channel];
        features[CHANNELS * 2 + channel] = range;
        features[CHANNELS * 3 + channel] = delta;
        features[CHANNELS * 4 + channel] = last;
    }

    features
}

impl Baseline {
    fn new() -> Self {
        Self {
            weights: vec![0.0; FEATURES * LABELS.len()],
            bias: [0.0; LABELS.len()],
        }
    }

    fn predict(&self, features: &[f32; FEATURES]) -> [f32; LABELS.len()] {
        let mut logits = self.bias;
        for (class, logit) in logits.iter_mut().enumerate() {
            for (feature_index, feature) in features.iter().enumerate() {
                *logit += feature * self.weights[feature_index * LABELS.len() + class];
            }
        }
        logits
    }

    fn train_one(&mut self, features: &[f32; FEATURES], target: &[f32; LABELS.len()], lr: f32) {
        let logits = self.predict(features);
        for class in 0..LABELS.len() {
            let gradient = sigmoid(logits[class]) - target[class];
            self.bias[class] -= lr * gradient;
            for (feature_index, feature) in features.iter().enumerate() {
                let weight = &mut self.weights[feature_index * LABELS.len() + class];
                *weight -= lr * gradient * feature;
                *weight = weight.clamp(-8.0, 8.0);
            }
        }
    }
}

fn evaluate(
    model: &Baseline,
    samples: &[Sample],
    features: &[[f32; FEATURES]],
    indexes: &[usize],
) -> Metrics {
    if indexes.is_empty() {
        return Metrics::default();
    }

    let mut metrics = Metrics::default();
    for sample_index in indexes {
        let sample = &samples[*sample_index];
        let logits = model.predict(&features[*sample_index]);
        metrics.loss += binary_cross_entropy_with_logits(&logits, &sample.target);

        let label_indexes = sample
            .labels
            .iter()
            .filter_map(|label| label_index(label))
            .collect::<Vec<_>>();
        let top = top_k(&logits, 3);
        if let Some(primary) = label_indexes.first() {
            if top[0].0 == *primary {
                metrics.primary_at_1 += 1.0;
            }
        } else if top[0].1 <= 0.0 {
            metrics.primary_at_1 += 1.0;
        }

        if !label_indexes.is_empty()
            && top
                .iter()
                .any(|(predicted, _)| label_indexes.contains(predicted))
        {
            metrics.any_at_3 += 1.0;
        } else if label_indexes.is_empty() && top[0].1 <= 0.0 {
            metrics.any_at_3 += 1.0;
        }
    }

    let count = indexes.len() as f32;
    metrics.loss /= count;
    metrics.primary_at_1 /= count;
    metrics.any_at_3 /= count;
    metrics
}

fn split_indexes(count: usize, validation_fraction: f32, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut indexes = (0..count).collect::<Vec<_>>();
    let mut rng = Lcg::new(seed);
    for index in (1..indexes.len()).rev() {
        let other = rng.range_usize(0, index + 1);
        indexes.swap(index, other);
    }

    let validation_count =
        ((count as f32 * validation_fraction).round() as usize).clamp(1, count.saturating_sub(1));
    let validation = indexes[..validation_count].to_vec();
    let train = indexes[validation_count..].to_vec();
    (train, validation)
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;

    while let Some(character) = chars.next() {
        match character {
            '"' if in_quotes && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                fields.push(field);
                field = String::new();
            }
            _ => field.push(character),
        }
    }

    fields.push(field);
    fields
}

fn target_from_labels(
    labels: &[String; 3],
) -> Result<[f32; LABELS.len()], Box<dyn std::error::Error>> {
    let mut target = [0.0_f32; LABELS.len()];
    let weights = [1.0_f32, 0.66, 0.33];
    for (label, weight) in labels.iter().zip(weights) {
        if is_no_scent_label(label) {
            continue;
        }
        let index = label_index(label).ok_or_else(|| format!("unknown label: {label}"))?;
        target[index] = weight;
    }
    Ok(target)
}

fn is_no_scent_label(label: &str) -> bool {
    normalize_label(label) == normalize_label(NO_SCENT_LABEL)
}

fn label_index(label: &str) -> Option<usize> {
    let normalized = normalize_label(label);
    LABELS
        .iter()
        .position(|candidate| normalize_label(candidate) == normalized)
}

fn normalize_label(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace() && *character != '-' && *character != '_')
        .flat_map(char::to_lowercase)
        .collect()
}

fn binary_cross_entropy_with_logits(logits: &[f32], target: &[f32; LABELS.len()]) -> f32 {
    let mut loss = 0.0;
    for (logit, target) in logits.iter().zip(target.iter()) {
        let max = logit.max(0.0);
        loss += max - logit * target + (1.0 + (-logit.abs()).exp()).ln();
    }
    loss / logits.len() as f32
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn top_k(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    indexed.truncate(k);
    indexed
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn range_usize(&mut self, min: usize, max: usize) -> usize {
        min + (self.next_u32() as usize % (max - min))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_summary_features() {
        let rows = vec![[0.1; CHANNELS], [0.3; CHANNELS]];
        let features = extract_features(&rows);

        assert_close(features[0], 0.2);
        assert_close(features[CHANNELS], 0.3);
        assert_close(features[CHANNELS * 2], 0.2);
        assert_close(features[CHANNELS * 3], 0.2);
        assert_close(features[CHANNELS * 4], 0.3);
    }

    #[test]
    fn splits_with_validation_rows() {
        let (train, validation) = split_indexes(10, 0.2, 1);
        assert_eq!(train.len(), 8);
        assert_eq!(validation.len(), 2);
    }

    #[test]
    fn no_scent_labels_map_to_empty_target() {
        let target = target_from_labels(&[
            NO_SCENT_LABEL.to_string(),
            NO_SCENT_LABEL.to_string(),
            NO_SCENT_LABEL.to_string(),
        ])
        .expect("target");

        assert!(target.iter().all(|value| *value == 0.0));
    }

    fn assert_close(left: f32, right: f32) {
        assert!((left - right).abs() < 1e-5, "{left} != {right}");
    }
}
