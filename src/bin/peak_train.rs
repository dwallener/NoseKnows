use std::cmp::Ordering;
use std::collections::VecDeque;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const CHANNELS: usize = 9;
const ACTIVE_SENSORS: usize = 8;
const PAIRS: usize = ACTIVE_SENSORS * (ACTIVE_SENSORS - 1) / 2;
const FEATURES: usize = ACTIVE_SENSORS * 2 + PAIRS * 8;
const OUTPUTS: usize = 14;
const MAX_ADC: f32 = 4095.0;
const DEFAULT_DATA: &str = "data/training/snn_comprehensive";
const DEFAULT_OUT: &str = "data/models/peak_pair_readout.npm";

const LABELS: [&str; OUTPUTS] = [
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

struct Config {
    data_dir: PathBuf,
    output_path: PathBuf,
    epochs: usize,
    learning_rate: f32,
    validation_fraction: f32,
    seed: u64,
    hold_secs: f32,
}

struct RawSample {
    id: String,
    labels: [String; 3],
    target: [bool; OUTPUTS],
    elapsed_ms: Vec<u64>,
    rows: Vec<[f32; CHANNELS]>,
}

struct EncodedSample {
    id: String,
    labels: [String; 3],
    target: [bool; OUTPUTS],
    bins: [u8; ACTIVE_SENSORS],
    features: [f32; FEATURES],
}

#[derive(Clone)]
struct PeakModel {
    weights: [[f32; FEATURES]; OUTPUTS],
    bias: [f32; OUTPUTS],
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    active: usize,
    no_scent: usize,
    p_at_1: usize,
    any_at_3: usize,
    silent_active: usize,
    silent_no_scent: usize,
    false_positive: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("peak_train error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let samples = load_samples(&config.data_dir)?;
    let samples = samples
        .into_iter()
        .filter(is_one_note_or_no_scent)
        .collect::<Vec<_>>();
    if samples.len() < 2 {
        return Err("peak training needs at least two one-note/no-scent samples".into());
    }

    let encoded = samples
        .iter()
        .map(|sample| encode_sample(sample, config.hold_secs))
        .collect::<Vec<_>>();
    let (train_indexes, validation_indexes) =
        split_indexes(encoded.len(), config.validation_fraction, config.seed);

    println!(
        "Loaded {} peak sample(s): {} train, {} validation",
        encoded.len(),
        train_indexes.len(),
        validation_indexes.len()
    );
    println!(
        "Peak accordion: {} active sensors -> 8-bin peak hold over {:.1}s -> {} pairwise features -> {} labels",
        ACTIVE_SENSORS, config.hold_secs, FEATURES, OUTPUTS
    );

    let mut model = PeakModel::new();
    let mut best_model = model.clone();
    let mut best_validation = Metrics::default();
    for epoch in 1..=config.epochs {
        for index in &train_indexes {
            model.train_one(&encoded[*index], config.learning_rate);
        }

        if epoch == 1 || epoch == config.epochs || epoch % 25 == 0 {
            let train = evaluate(&model, &encoded, &train_indexes);
            let validation = evaluate(&model, &encoded, &validation_indexes);
            if validation.score() > best_validation.score() {
                best_validation = validation;
                best_model = model.clone();
            }
            println!(
                "peak epoch {epoch:>4} train p@1 {:.2}% any@3 {:.2}% active-silence {:.2}% no-scent silence {:.2}% fp {:.2}% | val p@1 {:.2}% any@3 {:.2}% active-silence {:.2}% no-scent silence {:.2}% fp {:.2}%",
                train.p_at_1_pct(),
                train.any_at_3_pct(),
                train.active_silence_pct(),
                train.no_scent_silence_pct(),
                train.false_positive_pct(),
                validation.p_at_1_pct(),
                validation.any_at_3_pct(),
                validation.active_silence_pct(),
                validation.no_scent_silence_pct(),
                validation.false_positive_pct()
            );
        }
    }

    println!(
        "best peak validation p@1 {:.2}% any@3 {:.2}% active-silence {:.2}% no-scent silence {:.2}% fp {:.2}%",
        best_validation.p_at_1_pct(),
        best_validation.any_at_3_pct(),
        best_validation.active_silence_pct(),
        best_validation.no_scent_silence_pct(),
        best_validation.false_positive_pct()
    );
    print_validation_examples(&best_model, &encoded, &validation_indexes);
    save_model(&config.output_path, &config, &best_model)?;
    println!("Saved peak model to {}", config.output_path.display());
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut data_dir = PathBuf::from(DEFAULT_DATA);
    let mut output_path = PathBuf::from(DEFAULT_OUT);
    let mut epochs = 250;
    let mut learning_rate = 0.18;
    let mut validation_fraction = 0.2;
    let mut seed = 0x5eed_2026_u64;
    let mut hold_secs: f32 = 8.0;

    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--data" => {
                index += 1;
                data_dir = PathBuf::from(args.get(index).ok_or("--data requires a path")?);
            }
            "--out" => {
                index += 1;
                output_path = PathBuf::from(args.get(index).ok_or("--out requires a path")?);
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
            "--hold-secs" => {
                index += 1;
                hold_secs = args
                    .get(index)
                    .ok_or("--hold-secs requires a value")?
                    .parse()?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin peak_train -- [--data data/training/snn_comprehensive] [--out data/models/peak_pair_readout.npm] [--epochs 250] [--hold-secs 8]"
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
        output_path,
        epochs,
        learning_rate,
        validation_fraction,
        seed,
        hold_secs: hold_secs.max(0.1),
    })
}

fn is_one_note_or_no_scent(sample: &RawSample) -> bool {
    let active = sample.target.iter().filter(|value| **value).count();
    active <= 1
}

fn encode_sample(sample: &RawSample, hold_secs: f32) -> EncodedSample {
    let bins = peak_bins(sample, hold_secs);
    EncodedSample {
        id: sample.id.clone(),
        labels: sample.labels.clone(),
        target: sample.target,
        bins,
        features: pairwise_features(&bins),
    }
}

fn peak_bins(sample: &RawSample, hold_secs: f32) -> [u8; ACTIVE_SENSORS] {
    let window_rows = hold_window_rows(sample, hold_secs);
    let mut windows = (0..ACTIVE_SENSORS)
        .map(|_| VecDeque::<f32>::with_capacity(window_rows))
        .collect::<Vec<_>>();
    let mut held = [0.0_f32; ACTIVE_SENSORS];

    for row in &sample.rows {
        for sensor in 0..ACTIVE_SENSORS {
            let window = &mut windows[sensor];
            window.push_back(row[sensor]);
            while window.len() > window_rows {
                window.pop_front();
            }
            let peak = window.iter().fold(0.0_f32, |acc, value| acc.max(*value));
            held[sensor] = held[sensor].max(peak);
        }
    }

    let mut bins = [0_u8; ACTIVE_SENSORS];
    for sensor in 0..ACTIVE_SENSORS {
        bins[sensor] = quantize_8(held[sensor]);
    }
    bins
}

fn hold_window_rows(sample: &RawSample, hold_secs: f32) -> usize {
    let mut deltas = sample
        .elapsed_ms
        .windows(2)
        .filter_map(|pair| pair[1].checked_sub(pair[0]))
        .filter(|delta| *delta > 0)
        .collect::<Vec<_>>();
    deltas.sort_unstable();
    let period_ms = deltas.get(deltas.len() / 2).copied().unwrap_or(100).max(1);
    ((hold_secs * 1000.0) / period_ms as f32).round().max(1.0) as usize
}

fn quantize_8(value: f32) -> u8 {
    ((value / MAX_ADC).clamp(0.0, 1.0) * 8.0).floor().min(7.0) as u8
}

fn pairwise_features(bins: &[u8; ACTIVE_SENSORS]) -> [f32; FEATURES] {
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

    debug_assert_eq!(cursor, FEATURES);
    features
}

impl PeakModel {
    fn new() -> Self {
        Self {
            weights: [[0.0; FEATURES]; OUTPUTS],
            bias: [0.0; OUTPUTS],
        }
    }

    fn train_one(&mut self, sample: &EncodedSample, learning_rate: f32) {
        let logits = self.predict(&sample.features);
        let no_scent = is_no_scent_target(&sample.target);
        for label in 0..OUTPUTS {
            let target = if sample.target[label] { 1.0 } else { 0.0 };
            let weight = if sample.target[label] {
                4.0
            } else if no_scent {
                2.0
            } else {
                1.0
            };
            let error = target - sigmoid(logits[label]);
            let update = learning_rate * weight * error;
            self.bias[label] = (self.bias[label] + update * 0.02).clamp(-16.0, 16.0);
            for feature in 0..FEATURES {
                self.weights[label][feature] = (self.weights[label][feature]
                    + update * sample.features[feature])
                    .clamp(-32.0, 32.0);
            }
        }
    }

    fn predict(&self, features: &[f32; FEATURES]) -> [f32; OUTPUTS] {
        let mut logits = self.bias;
        for (label, logit) in logits.iter_mut().enumerate() {
            for (feature, value) in features.iter().enumerate() {
                *logit += self.weights[label][feature] * value;
            }
        }
        logits
    }
}

fn evaluate(model: &PeakModel, samples: &[EncodedSample], indexes: &[usize]) -> Metrics {
    let mut metrics = Metrics::default();
    for index in indexes {
        let sample = &samples[*index];
        let logits = model.predict(&sample.features);
        if is_no_scent_target(&sample.target) {
            metrics.no_scent += 1;
            if logits.iter().all(|value| *value <= 0.0) {
                metrics.silent_no_scent += 1;
            } else {
                metrics.false_positive += 1;
            }
            continue;
        }

        metrics.active += 1;
        if logits.iter().all(|value| *value <= 0.0) {
            metrics.silent_active += 1;
        }
        let top = top_k(&logits, 3);
        if sample.target[top[0].0] {
            metrics.p_at_1 += 1;
        }
        if top.iter().any(|(label, _)| sample.target[*label]) {
            metrics.any_at_3 += 1;
        }
    }
    metrics
}

impl Metrics {
    fn score(self) -> f32 {
        self.p_at_1_pct() + self.any_at_3_pct() + self.no_scent_silence_pct()
            - self.false_positive_pct()
            - self.active_silence_pct()
    }

    fn p_at_1_pct(self) -> f32 {
        percentage(self.p_at_1, self.active)
    }

    fn any_at_3_pct(self) -> f32 {
        percentage(self.any_at_3, self.active)
    }

    fn active_silence_pct(self) -> f32 {
        percentage(self.silent_active, self.active)
    }

    fn no_scent_silence_pct(self) -> f32 {
        percentage(self.silent_no_scent, self.no_scent)
    }

    fn false_positive_pct(self) -> f32 {
        percentage(self.false_positive, self.no_scent)
    }
}

fn print_validation_examples(model: &PeakModel, samples: &[EncodedSample], indexes: &[usize]) {
    println!();
    println!("Validation examples:");
    for index in indexes.iter().take(16) {
        let sample = &samples[*index];
        let logits = model.predict(&sample.features);
        let top = top_k(&logits, 3);
        println!(
            "{} labels=[{}, {}, {}] bins={:?} predicted=[{} {:.3}, {} {:.3}, {} {:.3}]",
            sample.id,
            sample.labels[0],
            sample.labels[1],
            sample.labels[2],
            sample.bins,
            LABELS[top[0].0],
            top[0].1,
            LABELS[top[1].0],
            top[1].1,
            LABELS[top[2].0],
            top[2].1
        );
    }
}

fn load_samples(data_dir: &Path) -> Result<Vec<RawSample>, Box<dyn std::error::Error>> {
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

fn load_sample(path: &Path) -> Result<Option<RawSample>, Box<dyn std::error::Error>> {
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

fn save_model(
    output_path: &Path,
    config: &Config,
    model: &PeakModel,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = fs::File::create(output_path)?;
    writeln!(file, "NOSEKNOWS_PEAK_PAIR_READOUT_V1")?;
    writeln!(file, "active_sensors={ACTIVE_SENSORS}")?;
    writeln!(file, "peak_bins=8")?;
    writeln!(file, "pairwise_features={FEATURES}")?;
    writeln!(file, "outputs={OUTPUTS}")?;
    writeln!(file, "hold_secs={:.3}", config.hold_secs)?;
    writeln!(file, "labels={}", LABELS.join(","))?;
    for label in 0..OUTPUTS {
        let weights = model.weights[label]
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(file, "bias.{}={:.6}", LABELS[label], model.bias[label])?;
        writeln!(file, "weights.{}={weights}", LABELS[label])?;
    }
    Ok(())
}

fn is_no_scent_target(target: &[bool; OUTPUTS]) -> bool {
    !target.iter().any(|value| *value)
}

fn label_index(label: &str) -> Option<usize> {
    LABELS
        .iter()
        .position(|candidate| candidate.eq_ignore_ascii_case(label))
}

fn top_k(values: &[f32; OUTPUTS], k: usize) -> Vec<(usize, f32)> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| LABELS[a.0].cmp(LABELS[b.0]))
    });
    indexed.truncate(k);
    indexed
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

fn percentage(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 * 100.0 / denominator as f32
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value.clamp(-40.0, 40.0)).exp())
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;

    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                fields.push(field);
                field = String::new();
            }
            _ => field.push(ch),
        }
    }
    fields.push(field);
    fields
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
    fn quantize_8_caps_to_seven() {
        assert_eq!(quantize_8(0.0), 0);
        assert_eq!(quantize_8(MAX_ADC), 7);
        assert_eq!(quantize_8(MAX_ADC * 2.0), 7);
    }

    #[test]
    fn pairwise_feature_count_is_exact() {
        let bins = [0, 1, 2, 3, 4, 5, 6, 7];
        let features = pairwise_features(&bins);
        assert_eq!(features.len(), FEATURES);
        assert!(features.iter().any(|value| *value > 0.0));
    }

    #[test]
    fn filters_only_no_scent_or_single_note() {
        let mut sample = RawSample {
            id: String::new(),
            labels: [
                "No Scent".to_string(),
                "No Scent".to_string(),
                "No Scent".to_string(),
            ],
            target: [false; OUTPUTS],
            elapsed_ms: Vec::new(),
            rows: Vec::new(),
        };
        assert!(is_one_note_or_no_scent(&sample));
        sample.target[0] = true;
        assert!(is_one_note_or_no_scent(&sample));
        sample.target[1] = true;
        assert!(!is_one_note_or_no_scent(&sample));
    }
}
