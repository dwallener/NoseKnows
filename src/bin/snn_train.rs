use std::cmp::Ordering;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const ADC_CHANNELS: usize = 9;
const ACTIVE_SENSORS: usize = 8;
const SNN_INPUTS: usize = ACTIVE_SENSORS * 2;
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
const DEFAULT_BINS: usize = 180;
const DEFAULT_SUBSLOTS: usize = 5;
const DEFAULT_RATE_BUDGET: usize = 5;
const DEFAULT_LATENCY_BUDGET: usize = 5;
const MAX_ADC: f32 = 4095.0;
const THRESHOLD: i32 = 1000;
const DECAY_ALPHA_Q8: i32 = 235;
const LIF_WEIGHT_SCALE: f32 = 260.0;
const LIF_FINE_TUNE_MULTIPLIER: f32 = 6.0;
const MIN_MEMBRANE: i32 = -3000;

#[derive(Clone)]
struct Sample {
    id: String,
    labels: [String; 3],
    target: [bool; LABELS.len()],
    rows: Vec<[f32; ADC_CHANNELS]>,
}

struct EncodedSample {
    id: String,
    labels: [String; 3],
    target: [bool; LABELS.len()],
    masks: Vec<u16>,
    features: [f32; SNN_INPUTS],
}

struct Config {
    data_dir: PathBuf,
    output_path: PathBuf,
    epochs: usize,
    learning_rate: f32,
    validation_fraction: f32,
    seed: u64,
    bins: usize,
    subslots: usize,
    rate_budget: usize,
    latency_budget: usize,
    include_designer: bool,
}

struct LifBank {
    weights: [[i16; SNN_INPUTS]; LABELS.len()],
}

struct LinearModel {
    weights: [[f32; SNN_INPUTS]; LABELS.len()],
    bias: [f32; LABELS.len()],
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    primary_at_1: f32,
    any_at_3: f32,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("snn_train error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let samples = load_samples(&config.data_dir)?;
    if samples.len() < 2 {
        return Err("SNN training needs at least two usable CSV samples".into());
    }
    let samples = samples
        .into_iter()
        .filter(|sample| config.include_designer || !sample.id.starts_with("designer_"))
        .collect::<Vec<_>>();
    if samples.len() < 2 {
        return Err(
            "SNN training needs at least two simple samples after filtering designer_* captures"
                .into(),
        );
    }

    let encoded = samples
        .iter()
        .map(|sample| encode_sample(sample, &config))
        .collect::<Vec<_>>();
    let (train_indexes, validation_indexes) =
        split_indexes(encoded.len(), config.validation_fraction, config.seed);

    println!(
        "Loaded {} sample(s): {} train, {} validation",
        encoded.len(),
        train_indexes.len(),
        validation_indexes.len()
    );
    println!(
        "SNN encoder: {} active sensors -> {} input streams, bins={}, subslots={}, rate_budget={}, latency_budget={}",
        ACTIVE_SENSORS,
        SNN_INPUTS,
        config.bins,
        config.subslots,
        config.rate_budget,
        config.latency_budget
    );
    if !config.include_designer {
        println!("Dataset filter: excluding designer_* complex fragrance captures");
    }

    let mut linear = LinearModel::new();
    for epoch in 1..=config.epochs {
        linear.train_epoch(&encoded, &train_indexes, config.learning_rate);

        if epoch == 1 || epoch == config.epochs || epoch % 25 == 0 {
            let bank = LifBank::from_linear(&linear);
            let linear_train = evaluate_linear(&linear, &encoded, &train_indexes);
            let linear_validation = evaluate_linear(&linear, &encoded, &validation_indexes);
            let lif_validation = evaluate_lif(&bank, &encoded, &validation_indexes);
            println!(
                "epoch {epoch:>4} linear train p@1 {:.2}% any@3 {:.2}% | val p@1 {:.2}% any@3 {:.2}% || lif val p@1 {:.2}% any@3 {:.2}%",
                linear_train.primary_at_1 * 100.0,
                linear_train.any_at_3 * 100.0,
                linear_validation.primary_at_1 * 100.0,
                linear_validation.any_at_3 * 100.0,
                lif_validation.primary_at_1 * 100.0,
                lif_validation.any_at_3 * 100.0
            );
        }
    }

    let mut lif_weights = linear.to_lif_float_weights();
    let mut best_lif_weights = lif_weights;
    let mut best_validation = Metrics::default();
    println!();
    println!("Fine-tuning integer LIF bank initialized from linear weights");
    for epoch in 1..=config.epochs {
        train_lif_epoch(
            &mut lif_weights,
            &encoded,
            &train_indexes,
            config.learning_rate * LIF_FINE_TUNE_MULTIPLIER,
        );

        if epoch == 1 || epoch == config.epochs || epoch % 25 == 0 {
            let bank = LifBank::from_lif_weights(&lif_weights);
            let train = evaluate_lif(&bank, &encoded, &train_indexes);
            let validation = evaluate_lif(&bank, &encoded, &validation_indexes);
            if better_metrics(&validation, &best_validation) {
                best_validation = validation;
                best_lif_weights = lif_weights;
            }
            println!(
                "lif epoch {epoch:>4} train p@1 {:.2}% any@3 {:.2}% | val p@1 {:.2}% any@3 {:.2}%",
                train.primary_at_1 * 100.0,
                train.any_at_3 * 100.0,
                validation.primary_at_1 * 100.0,
                validation.any_at_3 * 100.0
            );
        }
    }

    println!(
        "best lif validation p@1 {:.2}% any@3 {:.2}%",
        best_validation.primary_at_1 * 100.0,
        best_validation.any_at_3 * 100.0
    );

    let bank = LifBank::from_lif_weights(&best_lif_weights);
    println!();
    println!("Validation examples:");
    for index in validation_indexes.iter().take(10) {
        let activations = bank.forward(&encoded[*index].masks);
        let top = top_k(&activations, 3);
        let sample = &encoded[*index];
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

    save_model(&config.output_path, &config, &bank)?;
    println!();
    println!("Saved SNN LIF weights to {}", config.output_path.display());
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut data_dir = PathBuf::from("data/raw");
    let mut output_path = PathBuf::from("data/models/snn_lif.nsm");
    let mut epochs = 250;
    let mut learning_rate = 2.0;
    let mut validation_fraction = 0.2;
    let mut seed = 0x5eed_5d5_u64;
    let mut bins = DEFAULT_BINS;
    let mut subslots = DEFAULT_SUBSLOTS;
    let mut rate_budget = DEFAULT_RATE_BUDGET;
    let mut latency_budget = DEFAULT_LATENCY_BUDGET;
    let mut include_designer = false;

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
            "--bins" => {
                index += 1;
                bins = args.get(index).ok_or("--bins requires a value")?.parse()?;
            }
            "--subslots" => {
                index += 1;
                subslots = args
                    .get(index)
                    .ok_or("--subslots requires a value")?
                    .parse()?;
            }
            "--rate-budget" => {
                index += 1;
                rate_budget = args
                    .get(index)
                    .ok_or("--rate-budget requires a value")?
                    .parse()?;
            }
            "--latency-budget" => {
                index += 1;
                latency_budget = args
                    .get(index)
                    .ok_or("--latency-budget requires a value")?
                    .parse()?;
            }
            "--include-designer" => {
                include_designer = true;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin snn_train -- [--data data/raw] [--out data/models/snn_lif.nsm] [--epochs 250] [--lr 2.0] [--bins 180] [--subslots 5] [--rate-budget 5] [--latency-budget 5] [--include-designer]"
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
        bins,
        subslots,
        rate_budget,
        latency_budget,
        include_designer,
    })
}

fn evaluate_lif(bank: &LifBank, samples: &[EncodedSample], indexes: &[usize]) -> Metrics {
    if indexes.is_empty() {
        return Metrics::default();
    }

    let mut metrics = Metrics::default();
    for sample_index in indexes {
        let sample = &samples[*sample_index];
        let activations = bank.forward(&sample.masks);
        let top = top_k(&activations, 3);
        if sample.labels[0] == LABELS[top[0].0] {
            metrics.primary_at_1 += 1.0;
        }
        if top
            .iter()
            .any(|(label_index, _)| sample.target[*label_index])
        {
            metrics.any_at_3 += 1.0;
        }
    }

    metrics.primary_at_1 /= indexes.len() as f32;
    metrics.any_at_3 /= indexes.len() as f32;
    metrics
}

fn evaluate_linear(model: &LinearModel, samples: &[EncodedSample], indexes: &[usize]) -> Metrics {
    if indexes.is_empty() {
        return Metrics::default();
    }

    let mut metrics = Metrics::default();
    for sample_index in indexes {
        let sample = &samples[*sample_index];
        let logits = model.predict(&sample.features);
        let top = top_k_f32(&logits, 3);
        if sample.labels[0] == LABELS[top[0].0] {
            metrics.primary_at_1 += 1.0;
        }
        if top
            .iter()
            .any(|(label_index, _)| sample.target[*label_index])
        {
            metrics.any_at_3 += 1.0;
        }
    }

    metrics.primary_at_1 /= indexes.len() as f32;
    metrics.any_at_3 /= indexes.len() as f32;
    metrics
}

fn better_metrics(candidate: &Metrics, current: &Metrics) -> bool {
    candidate
        .any_at_3
        .partial_cmp(&current.any_at_3)
        .unwrap_or(Ordering::Equal)
        .then_with(|| {
            candidate
                .primary_at_1
                .partial_cmp(&current.primary_at_1)
                .unwrap_or(Ordering::Equal)
        })
        == Ordering::Greater
}

fn train_lif_epoch(
    weights: &mut [[f32; SNN_INPUTS]; LABELS.len()],
    samples: &[EncodedSample],
    train_indexes: &[usize],
    learning_rate: f32,
) {
    let bank = LifBank::from_lif_weights(weights);
    for sample_index in train_indexes {
        let sample = &samples[*sample_index];
        let activations = bank.forward(&sample.masks);
        let top = top_k(&activations, 3);

        for label_index in 0..LABELS.len() {
            if sample.target[label_index] && !top.iter().any(|(index, _)| *index == label_index) {
                update_lif_label(weights, label_index, &sample.features, learning_rate);
            }
        }

        for (label_index, _) in top {
            if !sample.target[label_index] {
                update_lif_label(weights, label_index, &sample.features, -learning_rate);
            }
        }
    }
}

fn update_lif_label(
    weights: &mut [[f32; SNN_INPUTS]; LABELS.len()],
    label_index: usize,
    features: &[f32; SNN_INPUTS],
    amount: f32,
) {
    for (input_index, feature) in features.iter().enumerate() {
        weights[label_index][input_index] =
            (weights[label_index][input_index] + amount * feature).clamp(-2048.0, 2048.0);
    }
}

impl LinearModel {
    fn new() -> Self {
        Self {
            weights: [[0.0; SNN_INPUTS]; LABELS.len()],
            bias: [0.0; LABELS.len()],
        }
    }

    fn train_epoch(
        &mut self,
        samples: &[EncodedSample],
        train_indexes: &[usize],
        learning_rate: f32,
    ) {
        for sample_index in train_indexes {
            let sample = &samples[*sample_index];
            let logits = self.predict(&sample.features);
            for (label_index, logit) in logits.iter().enumerate() {
                let target = if sample.target[label_index] { 1.0 } else { 0.0 };
                let positive_weight = if sample.target[label_index] { 3.0 } else { 1.0 };
                let error = target - sigmoid(*logit);
                let update = learning_rate * positive_weight * error;
                self.bias[label_index] = (self.bias[label_index] + update * 0.02).clamp(-8.0, 8.0);
                for input in 0..SNN_INPUTS {
                    self.weights[label_index][input] = (self.weights[label_index][input]
                        + update * sample.features[input])
                        .clamp(-16.0, 16.0);
                }
            }
        }
    }

    fn predict(&self, features: &[f32; SNN_INPUTS]) -> [f32; LABELS.len()] {
        let mut logits = self.bias;
        for (label_index, logit) in logits.iter_mut().enumerate() {
            for (input, feature) in features.iter().enumerate() {
                *logit += self.weights[label_index][input] * feature;
            }
        }
        logits
    }

    fn to_lif_float_weights(&self) -> [[f32; SNN_INPUTS]; LABELS.len()] {
        let mut weights = [[0.0_f32; SNN_INPUTS]; LABELS.len()];
        for label in 0..LABELS.len() {
            for input in 0..SNN_INPUTS {
                weights[label][input] =
                    (self.weights[label][input] * LIF_WEIGHT_SCALE).clamp(-2048.0, 2048.0);
            }
        }
        weights
    }
}

impl LifBank {
    fn from_linear(model: &LinearModel) -> Self {
        Self::from_lif_weights(&model.to_lif_float_weights())
    }

    fn from_lif_weights(weights: &[[f32; SNN_INPUTS]; LABELS.len()]) -> Self {
        let mut quantized = [[0_i16; SNN_INPUTS]; LABELS.len()];
        for label in 0..LABELS.len() {
            for input in 0..SNN_INPUTS {
                quantized[label][input] =
                    weights[label][input].round().clamp(-2048.0, 2048.0) as i16;
            }
        }
        Self { weights: quantized }
    }

    fn forward(&self, masks: &[u16]) -> [i32; LABELS.len()] {
        let mut membrane = [0_i32; LABELS.len()];
        let mut activations = [0_i32; LABELS.len()];

        for mask in masks {
            for label in 0..LABELS.len() {
                let mut next = (membrane[label] * DECAY_ALPHA_Q8) >> 8;
                for input in 0..SNN_INPUTS {
                    if ((mask >> input) & 1) != 0 {
                        next += self.weights[label][input] as i32;
                    }
                }

                if next >= THRESHOLD {
                    activations[label] += THRESHOLD;
                    membrane[label] = 0;
                } else {
                    membrane[label] = next.clamp(MIN_MEMBRANE, THRESHOLD - 1);
                }
            }
        }

        for label in 0..LABELS.len() {
            activations[label] += membrane[label] / 4;
        }
        activations
    }
}

fn encode_sample(sample: &Sample, config: &Config) -> EncodedSample {
    let bins = config.bins.min(sample.rows.len()).max(8);
    let subslots = config.subslots.max(1);
    let binned = bin_rows(&sample.rows, bins);
    let mut masks = vec![0_u16; binned.len() * subslots];
    let mut features = [0.0_f32; SNN_INPUTS];

    for channel in 0..ACTIVE_SENSORS {
        let baseline = binned
            .iter()
            .take(binned.len().min(10))
            .map(|row| row[channel])
            .sum::<f32>()
            / binned.len().min(10) as f32;
        let peak = binned
            .iter()
            .map(|row| row[channel])
            .fold(baseline, f32::max)
            .max(baseline + 1.0);
        let mut previous = binned[0][channel];

        for (bin_index, row) in binned.iter().enumerate() {
            let amplitude = ((row[channel] - baseline) / (peak - baseline)).clamp(0.0, 1.0);
            let rate_count = rate_spike_count(amplitude, config.rate_budget);
            for slot in 0..rate_count {
                let subslot = rate_subslot(slot, config.rate_budget, subslots);
                let mask_index = bin_index * subslots + subslot;
                masks[mask_index] |= 1 << channel;
                features[channel] += 1.0;
            }

            let delta = if bin_index == 0 {
                0.0
            } else {
                (row[channel] - previous).max(0.0) / MAX_ADC
            };
            if let Some(subslot) = latency_subslot(delta, subslots) {
                for slot in 0..config.latency_budget {
                    let event_subslot = (subslot + slot).min(subslots - 1);
                    let mask_index = bin_index * subslots + event_subslot;
                    masks[mask_index] |= 1 << (ACTIVE_SENSORS + channel);
                    features[ACTIVE_SENSORS + channel] += 1.0;
                }
            }
            previous = row[channel];
        }
    }

    let steps = masks.len().max(1) as f32;
    for feature in &mut features {
        *feature /= steps;
    }

    EncodedSample {
        id: sample.id.clone(),
        labels: sample.labels.clone(),
        target: sample.target,
        masks,
        features,
    }
}

fn bin_rows(rows: &[[f32; ADC_CHANNELS]], bins: usize) -> Vec<[f32; ADC_CHANNELS]> {
    let mut binned = Vec::with_capacity(bins);
    for bin in 0..bins {
        let start = bin * rows.len() / bins;
        let end = ((bin + 1) * rows.len() / bins).max(start + 1);
        let mut row = [0.0_f32; ADC_CHANNELS];
        for source in &rows[start..end.min(rows.len())] {
            for channel in 0..ADC_CHANNELS {
                row[channel] += source[channel];
            }
        }
        let count = (end.min(rows.len()) - start) as f32;
        for value in &mut row {
            *value /= count;
        }
        binned.push(row);
    }
    binned
}

fn rate_spike_count(amplitude: f32, budget: usize) -> usize {
    if amplitude <= 0.02 || budget == 0 {
        return 0;
    }
    let log_scaled = (1.0 + 31.0 * amplitude).ln() / 32.0_f32.ln();
    (log_scaled * budget as f32)
        .ceil()
        .clamp(0.0, budget as f32) as usize
}

fn rate_subslot(slot: usize, budget: usize, subslots: usize) -> usize {
    if budget <= 1 || subslots <= 1 {
        return 0;
    }
    let max_slot = budget - 1;
    ((slot * (subslots - 1)) + (max_slot / 2)) / max_slot
}

fn latency_subslot(delta: f32, subslots: usize) -> Option<usize> {
    if delta <= 0.0008 {
        return None;
    }
    let steepness = (delta / 0.08).clamp(0.0, 1.0);
    let max_subslot = subslots.saturating_sub(1);
    Some(((1.0 - steepness) * max_subslot as f32).round() as usize)
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

        let mut row = [0.0_f32; ADC_CHANNELS];
        for (channel, field_index) in adc_indexes.iter().enumerate() {
            row[channel] = fields[*field_index].parse::<f32>().unwrap_or(0.0);
        }
        rows.push(row);
    }

    if id.is_empty() {
        return Ok(None);
    }

    let mut target = [false; LABELS.len()];
    for label in &labels {
        if let Some(index) = label_index(label) {
            target[index] = true;
        }
    }

    Ok(Some(Sample {
        id,
        labels,
        target,
        rows,
    }))
}

fn save_model(
    output_path: &Path,
    config: &Config,
    bank: &LifBank,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = fs::File::create(output_path)?;
    writeln!(file, "NOSEKNOWS_SNN_LIF_V1")?;
    writeln!(file, "active_sensors={ACTIVE_SENSORS}")?;
    writeln!(file, "inputs={SNN_INPUTS}")?;
    writeln!(file, "outputs={}", LABELS.len())?;
    writeln!(file, "bins={}", config.bins)?;
    writeln!(file, "subslots={}", config.subslots)?;
    writeln!(file, "rate_budget={}", config.rate_budget)?;
    writeln!(file, "latency_budget={}", config.latency_budget)?;
    writeln!(file, "include_designer={}", config.include_designer)?;
    writeln!(file, "threshold={THRESHOLD}")?;
    writeln!(file, "decay_alpha_q8={DECAY_ALPHA_Q8}")?;
    writeln!(file, "labels={}", LABELS.join(","))?;
    for (label_index, row) in bank.weights.iter().enumerate() {
        let weights = row
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        writeln!(file, "weights.{}={weights}", LABELS[label_index])?;
    }
    Ok(())
}

fn split_indexes(count: usize, validation_fraction: f32, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut indexes = (0..count).collect::<Vec<_>>();
    let mut rng = Lcg::new(seed);
    for index in (1..indexes.len()).rev() {
        let other = rng.range_usize(0, index + 1);
        indexes.swap(index, other);
    }

    let validation_count = ((count as f32 * validation_fraction).round() as usize)
        .max(1)
        .min(count - 1);
    let validation = indexes[..validation_count].to_vec();
    let train = indexes[validation_count..].to_vec();
    (train, validation)
}

fn top_k(values: &[i32; LABELS.len()], k: usize) -> Vec<(usize, i32)> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| LABELS[a.0].cmp(LABELS[b.0]))
            .then(Ordering::Equal)
    });
    indexed.truncate(k);
    indexed
}

fn top_k_f32(values: &[f32; LABELS.len()], k: usize) -> Vec<(usize, f32)> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| LABELS[a.0].cmp(LABELS[b.0]))
    });
    indexed.truncate(k);
    indexed
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value.clamp(-40.0, 40.0)).exp())
}

fn label_index(label: &str) -> Option<usize> {
    LABELS
        .iter()
        .position(|candidate| candidate.eq_ignore_ascii_case(label))
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
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn range_usize(&mut self, min: usize, max: usize) -> usize {
        min + (self.next_u32() as usize % (max - min).max(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_count_respects_budget() {
        assert_eq!(rate_spike_count(0.0, 5), 0);
        assert!(rate_spike_count(0.9, 5) <= 5);
        assert!(rate_spike_count(0.9, 5) > rate_spike_count(0.1, 5));
    }

    #[test]
    fn latency_subslot_moves_earlier_for_steeper_delta() {
        let shallow = latency_subslot(0.01, 5).expect("shallow");
        let steep = latency_subslot(0.08, 5).expect("steep");

        assert!(steep < shallow);
    }

    #[test]
    fn lif_bank_accumulates_spikes() {
        let mut weights = [[0_i16; SNN_INPUTS]; LABELS.len()];
        weights[0][0] = 1200;
        let bank = LifBank { weights };

        let activations = bank.forward(&[1]);

        assert!(activations[0] >= THRESHOLD);
        assert_eq!(activations[1], 0);
    }

    #[test]
    fn parses_quoted_csv_fields() {
        let fields = parse_csv_line(r#"a,"b,c","d""e""#);

        assert_eq!(fields, vec!["a", "b,c", "d\"e"]);
    }
}
