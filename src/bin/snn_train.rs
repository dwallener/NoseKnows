use std::cmp::Ordering;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const ADC_CHANNELS: usize = 9;
const ACTIVE_SENSORS: usize = 8;
const SNN_INPUTS: usize = ACTIVE_SENSORS * 2;
const PATTERN_NEURONS: usize = 64;
const PATTERN_WINNERS_PER_STEP: usize = 3;
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
const MIN_SIGNAL_RANGE_ADC: f32 = 80.0;
const MIN_DELTA_ADC: f32 = 25.0;
const THRESHOLD: i32 = 1000;
const DECAY_ALPHA_Q8: i32 = 235;
const ADAPT_SENSOR: usize = 4;
const ADAPT_INCREMENT: i32 = 450;
const ADAPT_DECAY_Q8: i32 = 224;
const ADAPT_MAX: i32 = 2400;
const SILENCE_MAX_ACTIVATION: i32 = 0;
const SILENCE_MAX_LOGIT: f32 = 0.0;
const NO_SCENT_SUPPRESSION_MULTIPLIER: f32 = 0.25;
const NO_SCENT_LABEL: &str = "No Scent";
const LIF_WEIGHT_SCALE: f32 = 260.0;
const LIF_BIAS_SCALE: f32 = 40.0;
const PATTERN_LABEL_WEIGHT_SCALE: f32 = 220.0;
const PATTERN_LABEL_BIAS_SCALE: f32 = 40.0;
const LIF_FINE_TUNE_MULTIPLIER: f32 = 6.0;
const MIN_MEMBRANE: i32 = -3000;
const LABEL_FLORAL: usize = 0;
const LABEL_WATER: usize = 11;
const LABEL_GREEN: usize = 12;
const TOP_NOTE_INHIBITION: i32 = 1000;

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
    accordion: bool,
}

struct LifBank {
    weights: [[i16; SNN_INPUTS]; LABELS.len()],
    bias: [i16; LABELS.len()],
}

struct PatternBank {
    weights: [[i16; SNN_INPUTS]; PATTERN_NEURONS],
}

struct PatternEncodedSample {
    id: String,
    labels: [String; 3],
    target: [bool; LABELS.len()],
    masks: Vec<u64>,
    features: [f32; PATTERN_NEURONS],
}

struct PatternLinearModel {
    weights: [[f32; PATTERN_NEURONS]; LABELS.len()],
    bias: [f32; LABELS.len()],
}

struct PatternLabelBank {
    weights: [[i16; PATTERN_NEURONS]; LABELS.len()],
    bias: [i16; LABELS.len()],
}

struct AccordionModel {
    patterns: PatternBank,
    labels: PatternLabelBank,
}

struct LinearModel {
    weights: [[f32; SNN_INPUTS]; LABELS.len()],
    bias: [f32; LABELS.len()],
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    primary_at_1: f32,
    any_at_3: f32,
    no_scent_silence: f32,
    false_positive_rate: f32,
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
    if config.accordion {
        run_accordion_training(&config, &encoded, &train_indexes, &validation_indexes)?;
        return Ok(());
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
                "epoch {epoch:>4} linear train p@1 {:.2}% any@3 {:.2}% silence {:.2}% fp {:.2}% | val p@1 {:.2}% any@3 {:.2}% silence {:.2}% fp {:.2}% || lif val p@1 {:.2}% any@3 {:.2}% silence {:.2}% fp {:.2}%",
                linear_train.primary_at_1 * 100.0,
                linear_train.any_at_3 * 100.0,
                linear_train.no_scent_silence * 100.0,
                linear_train.false_positive_rate * 100.0,
                linear_validation.primary_at_1 * 100.0,
                linear_validation.any_at_3 * 100.0,
                linear_validation.no_scent_silence * 100.0,
                linear_validation.false_positive_rate * 100.0,
                lif_validation.primary_at_1 * 100.0,
                lif_validation.any_at_3 * 100.0,
                lif_validation.no_scent_silence * 100.0,
                lif_validation.false_positive_rate * 100.0
            );
        }
    }

    let mut lif_weights = linear.to_lif_float_weights();
    let mut lif_bias = linear.to_lif_float_bias();
    let mut best_lif_weights = lif_weights;
    let mut best_lif_bias = lif_bias;
    let mut best_validation = Metrics::default();
    println!();
    println!("Fine-tuning integer LIF bank initialized from linear weights");
    for epoch in 1..=config.epochs {
        train_lif_epoch(
            &mut lif_weights,
            &mut lif_bias,
            &encoded,
            &train_indexes,
            config.learning_rate * LIF_FINE_TUNE_MULTIPLIER,
        );

        if epoch == 1 || epoch == config.epochs || epoch % 25 == 0 {
            let bank = LifBank::from_lif_params(&lif_weights, &lif_bias);
            let train = evaluate_lif(&bank, &encoded, &train_indexes);
            let validation = evaluate_lif(&bank, &encoded, &validation_indexes);
            if better_metrics(&validation, &best_validation) {
                best_validation = validation;
                best_lif_weights = lif_weights;
                best_lif_bias = lif_bias;
            }
            println!(
                "lif epoch {epoch:>4} train p@1 {:.2}% any@3 {:.2}% silence {:.2}% fp {:.2}% | val p@1 {:.2}% any@3 {:.2}% silence {:.2}% fp {:.2}%",
                train.primary_at_1 * 100.0,
                train.any_at_3 * 100.0,
                train.no_scent_silence * 100.0,
                train.false_positive_rate * 100.0,
                validation.primary_at_1 * 100.0,
                validation.any_at_3 * 100.0,
                validation.no_scent_silence * 100.0,
                validation.false_positive_rate * 100.0
            );
        }
    }

    println!(
        "best lif validation p@1 {:.2}% any@3 {:.2}% silence {:.2}% fp {:.2}%",
        best_validation.primary_at_1 * 100.0,
        best_validation.any_at_3 * 100.0,
        best_validation.no_scent_silence * 100.0,
        best_validation.false_positive_rate * 100.0
    );

    let bank = LifBank::from_lif_params(&best_lif_weights, &best_lif_bias);
    println!();
    println!("Validation examples:");
    for index in validation_indexes.iter().take(10) {
        let activations = bank.forward(&encoded[*index].masks);
        let top = top_k(&activations, 3);
        let sample = &encoded[*index];
        if is_no_scent_target(&sample.target) && is_silent_i32(&activations) {
            println!(
                "{} labels=[{}, {}, {}] predicted=[{}]",
                sample.id, sample.labels[0], sample.labels[1], sample.labels[2], NO_SCENT_LABEL
            );
        } else {
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
    let mut accordion = false;

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
            "--accordion" => {
                accordion = true;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin snn_train -- [--data data/raw] [--out data/models/snn_lif.nsm] [--epochs 250] [--lr 2.0] [--bins 180] [--subslots 5] [--rate-budget 5] [--latency-budget 5] [--include-designer] [--accordion]"
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
        accordion,
    })
}

fn run_accordion_training(
    config: &Config,
    encoded: &[EncodedSample],
    train_indexes: &[usize],
    validation_indexes: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let patterns = PatternBank::seeded();
    let pattern_encoded = encoded
        .iter()
        .map(|sample| encode_patterns(sample, &patterns))
        .collect::<Vec<_>>();

    println!(
        "Accordion SNN: {} input streams -> {} seeded emergent-pattern neurons -> {} labels",
        SNN_INPUTS,
        PATTERN_NEURONS,
        LABELS.len()
    );
    println!(
        "Pattern layer: fixed LIF motifs, winner-take-few lateral inhibition, max {} winners/step",
        PATTERN_WINNERS_PER_STEP
    );
    println!(
        "Pattern adaptation: adc{ADAPT_SENSOR}-linked motifs threshold += {ADAPT_INCREMENT}, decay_q8={ADAPT_DECAY_Q8}, max={ADAPT_MAX}"
    );

    let mut linear = PatternLinearModel::new();
    for epoch in 1..=config.epochs {
        linear.train_epoch(&pattern_encoded, train_indexes, config.learning_rate);

        if epoch == 1 || epoch == config.epochs || epoch % 25 == 0 {
            let bank = PatternLabelBank::from_linear(&linear);
            let linear_train = evaluate_pattern_linear(&linear, &pattern_encoded, train_indexes);
            let linear_validation =
                evaluate_pattern_linear(&linear, &pattern_encoded, validation_indexes);
            let lif_validation = evaluate_pattern_lif(&bank, &pattern_encoded, validation_indexes);
            println!(
                "accordion epoch {epoch:>4} linear train p@1 {:.2}% any@3 {:.2}% silence {:.2}% fp {:.2}% | val p@1 {:.2}% any@3 {:.2}% silence {:.2}% fp {:.2}% || lif val p@1 {:.2}% any@3 {:.2}% silence {:.2}% fp {:.2}%",
                linear_train.primary_at_1 * 100.0,
                linear_train.any_at_3 * 100.0,
                linear_train.no_scent_silence * 100.0,
                linear_train.false_positive_rate * 100.0,
                linear_validation.primary_at_1 * 100.0,
                linear_validation.any_at_3 * 100.0,
                linear_validation.no_scent_silence * 100.0,
                linear_validation.false_positive_rate * 100.0,
                lif_validation.primary_at_1 * 100.0,
                lif_validation.any_at_3 * 100.0,
                lif_validation.no_scent_silence * 100.0,
                lif_validation.false_positive_rate * 100.0
            );
        }
    }

    let mut label_weights = linear.to_lif_float_weights();
    let mut label_bias = linear.to_lif_float_bias();
    let mut best_label_weights = label_weights;
    let mut best_label_bias = label_bias;
    let mut best_validation = Metrics::default();
    println!();
    println!("Fine-tuning accordion label LIF bank initialized from pattern-count linear weights");
    for epoch in 1..=config.epochs {
        train_pattern_lif_epoch(
            &mut label_weights,
            &mut label_bias,
            &pattern_encoded,
            train_indexes,
            config.learning_rate * LIF_FINE_TUNE_MULTIPLIER,
        );

        if epoch == 1 || epoch == config.epochs || epoch % 25 == 0 {
            let bank = PatternLabelBank::from_lif_params(&label_weights, &label_bias);
            let train = evaluate_pattern_lif(&bank, &pattern_encoded, train_indexes);
            let validation = evaluate_pattern_lif(&bank, &pattern_encoded, validation_indexes);
            if better_metrics(&validation, &best_validation) {
                best_validation = validation;
                best_label_weights = label_weights;
                best_label_bias = label_bias;
            }
            println!(
                "accordion lif epoch {epoch:>4} train p@1 {:.2}% any@3 {:.2}% silence {:.2}% fp {:.2}% | val p@1 {:.2}% any@3 {:.2}% silence {:.2}% fp {:.2}%",
                train.primary_at_1 * 100.0,
                train.any_at_3 * 100.0,
                train.no_scent_silence * 100.0,
                train.false_positive_rate * 100.0,
                validation.primary_at_1 * 100.0,
                validation.any_at_3 * 100.0,
                validation.no_scent_silence * 100.0,
                validation.false_positive_rate * 100.0
            );
        }
    }

    println!(
        "best accordion validation p@1 {:.2}% any@3 {:.2}% silence {:.2}% fp {:.2}%",
        best_validation.primary_at_1 * 100.0,
        best_validation.any_at_3 * 100.0,
        best_validation.no_scent_silence * 100.0,
        best_validation.false_positive_rate * 100.0
    );

    let label_bank = PatternLabelBank::from_lif_params(&best_label_weights, &best_label_bias);
    println!();
    println!("Validation examples:");
    for index in validation_indexes.iter().take(10) {
        let activations = label_bank.forward(&pattern_encoded[*index].masks);
        let top = top_k(&activations, 3);
        let sample = &pattern_encoded[*index];
        if is_no_scent_target(&sample.target) && is_silent_i32(&activations) {
            println!(
                "{} labels=[{}, {}, {}] predicted=[{}]",
                sample.id, sample.labels[0], sample.labels[1], sample.labels[2], NO_SCENT_LABEL
            );
        } else {
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
    }

    let model = AccordionModel {
        patterns,
        labels: label_bank,
    };
    save_accordion_model(&config.output_path, config, &model)?;
    println!();
    println!(
        "Saved SNN accordion weights to {}",
        config.output_path.display()
    );
    Ok(())
}

fn encode_patterns(sample: &EncodedSample, patterns: &PatternBank) -> PatternEncodedSample {
    let masks = patterns.forward_masks(&sample.masks);
    let mut features = [0.0_f32; PATTERN_NEURONS];
    for mask in &masks {
        for pattern in 0..PATTERN_NEURONS {
            if ((mask >> pattern) & 1) != 0 {
                features[pattern] += 1.0;
            }
        }
    }
    let steps = masks.len().max(1) as f32;
    for feature in &mut features {
        *feature /= steps;
    }

    PatternEncodedSample {
        id: sample.id.clone(),
        labels: sample.labels.clone(),
        target: sample.target,
        masks,
        features,
    }
}

fn evaluate_lif(bank: &LifBank, samples: &[EncodedSample], indexes: &[usize]) -> Metrics {
    if indexes.is_empty() {
        return Metrics::default();
    }

    let mut metrics = Metrics::default();
    let mut no_scent_count = 0.0_f32;
    for sample_index in indexes {
        let sample = &samples[*sample_index];
        let activations = bank.forward(&sample.masks);
        let top = top_k(&activations, 3);
        if is_no_scent_target(&sample.target) {
            no_scent_count += 1.0;
            if is_silent_i32(&activations) {
                metrics.primary_at_1 += 1.0;
                metrics.any_at_3 += 1.0;
                metrics.no_scent_silence += 1.0;
            } else {
                metrics.false_positive_rate += 1.0;
            }
        } else {
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
    }

    metrics.primary_at_1 /= indexes.len() as f32;
    metrics.any_at_3 /= indexes.len() as f32;
    if no_scent_count > 0.0 {
        metrics.no_scent_silence /= no_scent_count;
        metrics.false_positive_rate /= no_scent_count;
    }
    metrics
}

fn evaluate_linear(model: &LinearModel, samples: &[EncodedSample], indexes: &[usize]) -> Metrics {
    if indexes.is_empty() {
        return Metrics::default();
    }

    let mut metrics = Metrics::default();
    let mut no_scent_count = 0.0_f32;
    for sample_index in indexes {
        let sample = &samples[*sample_index];
        let logits = model.predict(&sample.features);
        let top = top_k_f32(&logits, 3);
        if is_no_scent_target(&sample.target) {
            no_scent_count += 1.0;
            if is_silent_f32(&logits) {
                metrics.primary_at_1 += 1.0;
                metrics.any_at_3 += 1.0;
                metrics.no_scent_silence += 1.0;
            } else {
                metrics.false_positive_rate += 1.0;
            }
        } else {
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
    }

    metrics.primary_at_1 /= indexes.len() as f32;
    metrics.any_at_3 /= indexes.len() as f32;
    if no_scent_count > 0.0 {
        metrics.no_scent_silence /= no_scent_count;
        metrics.false_positive_rate /= no_scent_count;
    }
    metrics
}

fn evaluate_pattern_linear(
    model: &PatternLinearModel,
    samples: &[PatternEncodedSample],
    indexes: &[usize],
) -> Metrics {
    if indexes.is_empty() {
        return Metrics::default();
    }

    let mut metrics = Metrics::default();
    let mut no_scent_count = 0.0_f32;
    for sample_index in indexes {
        let sample = &samples[*sample_index];
        let logits = model.predict(&sample.features);
        let top = top_k_f32(&logits, 3);
        if is_no_scent_target(&sample.target) {
            no_scent_count += 1.0;
            if is_silent_f32(&logits) {
                metrics.primary_at_1 += 1.0;
                metrics.any_at_3 += 1.0;
                metrics.no_scent_silence += 1.0;
            } else {
                metrics.false_positive_rate += 1.0;
            }
        } else {
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
    }

    metrics.primary_at_1 /= indexes.len() as f32;
    metrics.any_at_3 /= indexes.len() as f32;
    if no_scent_count > 0.0 {
        metrics.no_scent_silence /= no_scent_count;
        metrics.false_positive_rate /= no_scent_count;
    }
    metrics
}

fn evaluate_pattern_lif(
    bank: &PatternLabelBank,
    samples: &[PatternEncodedSample],
    indexes: &[usize],
) -> Metrics {
    if indexes.is_empty() {
        return Metrics::default();
    }

    let mut metrics = Metrics::default();
    let mut no_scent_count = 0.0_f32;
    for sample_index in indexes {
        let sample = &samples[*sample_index];
        let activations = bank.forward(&sample.masks);
        let top = top_k(&activations, 3);
        if is_no_scent_target(&sample.target) {
            no_scent_count += 1.0;
            if is_silent_i32(&activations) {
                metrics.primary_at_1 += 1.0;
                metrics.any_at_3 += 1.0;
                metrics.no_scent_silence += 1.0;
            } else {
                metrics.false_positive_rate += 1.0;
            }
        } else {
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
    }

    metrics.primary_at_1 /= indexes.len() as f32;
    metrics.any_at_3 /= indexes.len() as f32;
    if no_scent_count > 0.0 {
        metrics.no_scent_silence /= no_scent_count;
        metrics.false_positive_rate /= no_scent_count;
    }
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
        .then_with(|| {
            candidate
                .no_scent_silence
                .partial_cmp(&current.no_scent_silence)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| {
            current
                .false_positive_rate
                .partial_cmp(&candidate.false_positive_rate)
                .unwrap_or(Ordering::Equal)
        })
        == Ordering::Greater
}

fn is_no_scent_target(target: &[bool; LABELS.len()]) -> bool {
    target.iter().all(|value| !*value)
}

fn is_silent_i32(values: &[i32; LABELS.len()]) -> bool {
    values.iter().all(|value| *value <= SILENCE_MAX_ACTIVATION)
}

fn is_silent_f32(values: &[f32; LABELS.len()]) -> bool {
    values.iter().all(|value| *value <= SILENCE_MAX_LOGIT)
}

fn train_pattern_lif_epoch(
    weights: &mut [[f32; PATTERN_NEURONS]; LABELS.len()],
    bias: &mut [f32; LABELS.len()],
    samples: &[PatternEncodedSample],
    train_indexes: &[usize],
    learning_rate: f32,
) {
    let bank = PatternLabelBank::from_lif_params(weights, bias);
    for sample_index in train_indexes {
        let sample = &samples[*sample_index];
        let activations = bank.forward(&sample.masks);
        let top = top_k(&activations, 3);

        if is_no_scent_target(&sample.target) {
            for (label_index, activation) in activations.iter().enumerate() {
                if *activation > SILENCE_MAX_ACTIVATION {
                    update_pattern_lif_label(
                        weights,
                        bias,
                        label_index,
                        &sample.features,
                        -learning_rate * NO_SCENT_SUPPRESSION_MULTIPLIER,
                    );
                }
            }
            continue;
        }

        for label_index in 0..LABELS.len() {
            if sample.target[label_index] && !top.iter().any(|(index, _)| *index == label_index) {
                update_pattern_lif_label(
                    weights,
                    bias,
                    label_index,
                    &sample.features,
                    learning_rate,
                );
            }
        }

        for (label_index, _) in top {
            if !sample.target[label_index] {
                update_pattern_lif_label(
                    weights,
                    bias,
                    label_index,
                    &sample.features,
                    -learning_rate,
                );
            }
        }
    }
}

fn update_pattern_lif_label(
    weights: &mut [[f32; PATTERN_NEURONS]; LABELS.len()],
    bias: &mut [f32; LABELS.len()],
    label_index: usize,
    features: &[f32; PATTERN_NEURONS],
    amount: f32,
) {
    bias[label_index] = (bias[label_index] + amount * 0.2).clamp(-2048.0, 2048.0);
    for (pattern_index, feature) in features.iter().enumerate() {
        weights[label_index][pattern_index] =
            (weights[label_index][pattern_index] + amount * feature).clamp(-2048.0, 2048.0);
    }
}

fn train_lif_epoch(
    weights: &mut [[f32; SNN_INPUTS]; LABELS.len()],
    bias: &mut [f32; LABELS.len()],
    samples: &[EncodedSample],
    train_indexes: &[usize],
    learning_rate: f32,
) {
    let bank = LifBank::from_lif_params(weights, bias);
    for sample_index in train_indexes {
        let sample = &samples[*sample_index];
        let activations = bank.forward(&sample.masks);
        let top = top_k(&activations, 3);

        if is_no_scent_target(&sample.target) {
            for (label_index, activation) in activations.iter().enumerate() {
                if *activation > SILENCE_MAX_ACTIVATION {
                    update_lif_label(
                        weights,
                        bias,
                        label_index,
                        &sample.features,
                        -learning_rate * NO_SCENT_SUPPRESSION_MULTIPLIER,
                    );
                }
            }
            continue;
        }

        for label_index in 0..LABELS.len() {
            if sample.target[label_index] && !top.iter().any(|(index, _)| *index == label_index) {
                update_lif_label(weights, bias, label_index, &sample.features, learning_rate);
            }
        }

        for (label_index, _) in top {
            if !sample.target[label_index] {
                update_lif_label(weights, bias, label_index, &sample.features, -learning_rate);
            }
        }
    }
}

fn update_lif_label(
    weights: &mut [[f32; SNN_INPUTS]; LABELS.len()],
    bias: &mut [f32; LABELS.len()],
    label_index: usize,
    features: &[f32; SNN_INPUTS],
    amount: f32,
) {
    bias[label_index] = (bias[label_index] + amount * 0.2).clamp(-2048.0, 2048.0);
    for (input_index, feature) in features.iter().enumerate() {
        weights[label_index][input_index] =
            (weights[label_index][input_index] + amount * feature).clamp(-2048.0, 2048.0);
    }
}

impl PatternBank {
    fn seeded() -> Self {
        let mut weights = [[-60_i16; SNN_INPUTS]; PATTERN_NEURONS];

        for input in 0..SNN_INPUTS {
            weights[input] = [-90; SNN_INPUTS];
            weights[input][input] = 980;
            weights[input][paired_stream(input)] = 180;
        }

        let pairs = [
            [0, 2],
            [0, 3],
            [2, 3],
            [1, 7],
            [1, 4],
            [4, 6],
            [4, 7],
            [0, 7],
            [1, 3],
            [3, 4],
            [6, 7],
            [0, 4],
            [1, 6],
            [2, 7],
            [1, 2],
            [3, 7],
        ];
        for (offset, sensors) in pairs.iter().enumerate() {
            let pattern = 16 + offset;
            set_sensor_motif(&mut weights[pattern], sensors, 500, 420, -80);
        }

        let onset_tail = [
            ([0, 2, 3], [7, 7]),
            ([0, 1, 3], [4, 7]),
            ([1, 3, 7], [4, 4]),
            ([1, 2, 7], [0, 0]),
            ([3, 4, 7], [6, 6]),
            ([4, 6, 7], [0, 0]),
            ([0, 4, 6], [7, 7]),
            ([1, 4, 6], [7, 7]),
            ([0, 1, 7], [2, 2]),
            ([2, 3, 7], [1, 1]),
            ([0, 3, 4], [6, 6]),
            ([1, 6, 7], [4, 4]),
            ([0, 2, 7], [3, 3]),
            ([1, 3, 4], [7, 7]),
            ([0, 6, 7], [4, 4]),
            ([2, 4, 7], [1, 1]),
        ];
        for (offset, (fast, tail)) in onset_tail.iter().enumerate() {
            let pattern = 32 + offset;
            weights[pattern] = [-85; SNN_INPUTS];
            for sensor in fast {
                weights[pattern][ACTIVE_SENSORS + *sensor] = 430;
            }
            for sensor in tail {
                weights[pattern][*sensor] = 360;
            }
        }

        let clusters: [&[usize]; 16] = [
            &[0, 2, 3],
            &[1, 7],
            &[1, 3, 7],
            &[0, 1, 7],
            &[1, 4, 7],
            &[1, 4, 6, 7],
            &[0, 4, 7],
            &[0, 4, 6, 7],
            &[0, 7],
            &[0, 1, 4, 7],
            &[0, 4, 6],
            &[1, 3, 4, 7],
            &[0, 1, 2],
            &[2, 3, 4],
            &[4, 6],
            &[0, 1, 3, 6],
        ];
        for (offset, sensors) in clusters.iter().enumerate() {
            let pattern = 48 + offset;
            let rate_weight = if offset % 2 == 0 { 380 } else { 260 };
            let latency_weight = if offset % 2 == 0 { 260 } else { 380 };
            set_sensor_motif(
                &mut weights[pattern],
                sensors,
                rate_weight,
                latency_weight,
                -70,
            );
        }

        Self { weights }
    }

    fn forward_masks(&self, input_masks: &[u16]) -> Vec<u64> {
        let mut membrane = [0_i32; PATTERN_NEURONS];
        let mut adaptation = [0_i32; PATTERN_NEURONS];
        let mut pattern_masks = Vec::with_capacity(input_masks.len());

        for input_mask in input_masks {
            let mut next_membrane = [0_i32; PATTERN_NEURONS];
            let mut candidates = Vec::new();
            for pattern in 0..PATTERN_NEURONS {
                let mut next = (membrane[pattern] * DECAY_ALPHA_Q8) >> 8;
                for input in 0..SNN_INPUTS {
                    if ((input_mask >> input) & 1) != 0 {
                        next += self.weights[pattern][input] as i32;
                    }
                }
                next_membrane[pattern] = next;
                if next >= THRESHOLD + adaptation[pattern] {
                    candidates.push((pattern, next));
                }
            }

            candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let mut mask = 0_u64;
            for (rank, (pattern, _)) in candidates.iter().enumerate() {
                if rank < PATTERN_WINNERS_PER_STEP {
                    mask |= 1_u64 << pattern;
                    next_membrane[*pattern] = 0;
                } else {
                    next_membrane[*pattern] -= THRESHOLD / 2;
                }
            }

            for pattern in 0..PATTERN_NEURONS {
                membrane[pattern] = next_membrane[pattern].clamp(MIN_MEMBRANE, THRESHOLD - 1);
                adaptation[pattern] = (adaptation[pattern] * ADAPT_DECAY_Q8) >> 8;
                if ((mask >> pattern) & 1) != 0
                    && pattern_uses_sensor(&self.weights[pattern], ADAPT_SENSOR)
                {
                    adaptation[pattern] = (adaptation[pattern] + ADAPT_INCREMENT).min(ADAPT_MAX);
                }
            }
            pattern_masks.push(mask);
        }

        pattern_masks
    }
}

impl PatternLinearModel {
    fn new() -> Self {
        Self {
            weights: [[0.0; PATTERN_NEURONS]; LABELS.len()],
            bias: [0.0; LABELS.len()],
        }
    }

    fn train_epoch(
        &mut self,
        samples: &[PatternEncodedSample],
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
                for pattern in 0..PATTERN_NEURONS {
                    self.weights[label_index][pattern] = (self.weights[label_index][pattern]
                        + update * sample.features[pattern])
                        .clamp(-16.0, 16.0);
                }
            }
        }
    }

    fn predict(&self, features: &[f32; PATTERN_NEURONS]) -> [f32; LABELS.len()] {
        let mut logits = self.bias;
        for (label_index, logit) in logits.iter_mut().enumerate() {
            for (pattern, feature) in features.iter().enumerate() {
                *logit += self.weights[label_index][pattern] * feature;
            }
        }
        logits
    }

    fn to_lif_float_weights(&self) -> [[f32; PATTERN_NEURONS]; LABELS.len()] {
        let mut weights = [[0.0_f32; PATTERN_NEURONS]; LABELS.len()];
        for label in 0..LABELS.len() {
            for pattern in 0..PATTERN_NEURONS {
                weights[label][pattern] = (self.weights[label][pattern]
                    * PATTERN_LABEL_WEIGHT_SCALE)
                    .clamp(-2048.0, 2048.0);
            }
        }
        weights
    }

    fn to_lif_float_bias(&self) -> [f32; LABELS.len()] {
        let mut bias = [0.0_f32; LABELS.len()];
        for (label, value) in bias.iter_mut().enumerate() {
            *value = (self.bias[label] * PATTERN_LABEL_BIAS_SCALE).clamp(-2048.0, 2048.0);
        }
        bias
    }
}

impl PatternLabelBank {
    fn from_linear(model: &PatternLinearModel) -> Self {
        Self::from_lif_params(&model.to_lif_float_weights(), &model.to_lif_float_bias())
    }

    fn from_lif_params(
        weights: &[[f32; PATTERN_NEURONS]; LABELS.len()],
        bias: &[f32; LABELS.len()],
    ) -> Self {
        let mut quantized = [[0_i16; PATTERN_NEURONS]; LABELS.len()];
        let mut quantized_bias = [0_i16; LABELS.len()];
        for label in 0..LABELS.len() {
            quantized_bias[label] = bias[label].round().clamp(-2048.0, 2048.0) as i16;
            for pattern in 0..PATTERN_NEURONS {
                quantized[label][pattern] =
                    weights[label][pattern].round().clamp(-2048.0, 2048.0) as i16;
            }
        }
        Self {
            weights: quantized,
            bias: quantized_bias,
        }
    }

    fn forward(&self, pattern_masks: &[u64]) -> [i32; LABELS.len()] {
        let mut membrane = [0_i32; LABELS.len()];
        let mut activations = [0_i32; LABELS.len()];

        for mask in pattern_masks {
            let mut next_values = [0_i32; LABELS.len()];
            for label in 0..LABELS.len() {
                let mut next = ((membrane[label] * DECAY_ALPHA_Q8) >> 8) + self.bias[label] as i32;
                for pattern in 0..PATTERN_NEURONS {
                    if ((mask >> pattern) & 1) != 0 {
                        next += self.weights[label][pattern] as i32;
                    }
                }
                next_values[label] = next;
            }
            apply_label_inhibition(&mut next_values);

            for label in 0..LABELS.len() {
                let next = next_values[label];
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

fn paired_stream(input: usize) -> usize {
    if input < ACTIVE_SENSORS {
        ACTIVE_SENSORS + input
    } else {
        input - ACTIVE_SENSORS
    }
}

fn set_sensor_motif(
    weights: &mut [i16; SNN_INPUTS],
    sensors: &[usize],
    rate_weight: i16,
    latency_weight: i16,
    background: i16,
) {
    *weights = [background; SNN_INPUTS];
    for sensor in sensors {
        weights[*sensor] = rate_weight;
        weights[ACTIVE_SENSORS + *sensor] = latency_weight;
    }
}

fn pattern_uses_sensor(weights: &[i16; SNN_INPUTS], sensor: usize) -> bool {
    weights[sensor] > 0 || weights[ACTIVE_SENSORS + sensor] > 0
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

    fn to_lif_float_bias(&self) -> [f32; LABELS.len()] {
        let mut bias = [0.0_f32; LABELS.len()];
        for (label, value) in bias.iter_mut().enumerate() {
            *value = (self.bias[label] * LIF_BIAS_SCALE).clamp(-2048.0, 2048.0);
        }
        bias
    }
}

impl LifBank {
    fn from_linear(model: &LinearModel) -> Self {
        Self::from_lif_params(&model.to_lif_float_weights(), &model.to_lif_float_bias())
    }

    fn from_lif_params(
        weights: &[[f32; SNN_INPUTS]; LABELS.len()],
        bias: &[f32; LABELS.len()],
    ) -> Self {
        let mut quantized = [[0_i16; SNN_INPUTS]; LABELS.len()];
        let mut quantized_bias = [0_i16; LABELS.len()];
        for label in 0..LABELS.len() {
            quantized_bias[label] = bias[label].round().clamp(-2048.0, 2048.0) as i16;
            for input in 0..SNN_INPUTS {
                quantized[label][input] =
                    weights[label][input].round().clamp(-2048.0, 2048.0) as i16;
            }
        }
        Self {
            weights: quantized,
            bias: quantized_bias,
        }
    }

    fn forward(&self, masks: &[u16]) -> [i32; LABELS.len()] {
        let mut membrane = [0_i32; LABELS.len()];
        let mut activations = [0_i32; LABELS.len()];

        for mask in masks {
            let mut next_values = [0_i32; LABELS.len()];
            for label in 0..LABELS.len() {
                let mut next = ((membrane[label] * DECAY_ALPHA_Q8) >> 8) + self.bias[label] as i32;
                for input in 0..SNN_INPUTS {
                    if ((mask >> input) & 1) != 0 {
                        next += self.weights[label][input] as i32;
                    }
                }
                next_values[label] = next;
            }
            apply_label_inhibition(&mut next_values);

            for label in 0..LABELS.len() {
                let next = next_values[label];
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

fn apply_label_inhibition(values: &mut [i32; LABELS.len()]) {
    let mut inhibition = 0;
    if values[LABEL_GREEN] >= THRESHOLD {
        inhibition += TOP_NOTE_INHIBITION;
    }
    if values[LABEL_WATER] >= THRESHOLD {
        inhibition += TOP_NOTE_INHIBITION;
    }
    values[LABEL_FLORAL] -= inhibition;
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
            let signal_range = peak - baseline;
            let amplitude = if signal_range >= MIN_SIGNAL_RANGE_ADC {
                ((row[channel] - baseline) / signal_range).clamp(0.0, 1.0)
            } else {
                0.0
            };
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
                let raw_delta = (row[channel] - previous).max(0.0);
                if raw_delta >= MIN_DELTA_ADC {
                    raw_delta / MAX_ADC
                } else {
                    0.0
                }
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
        writeln!(
            file,
            "bias.{}={}",
            LABELS[label_index], bank.bias[label_index]
        )?;
        writeln!(file, "weights.{}={weights}", LABELS[label_index])?;
    }
    Ok(())
}

fn save_accordion_model(
    output_path: &Path,
    config: &Config,
    model: &AccordionModel,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = fs::File::create(output_path)?;
    writeln!(file, "NOSEKNOWS_SNN_ACCORDION_V1")?;
    writeln!(file, "active_sensors={ACTIVE_SENSORS}")?;
    writeln!(file, "inputs={SNN_INPUTS}")?;
    writeln!(file, "patterns={PATTERN_NEURONS}")?;
    writeln!(file, "outputs={}", LABELS.len())?;
    writeln!(file, "bins={}", config.bins)?;
    writeln!(file, "subslots={}", config.subslots)?;
    writeln!(file, "rate_budget={}", config.rate_budget)?;
    writeln!(file, "latency_budget={}", config.latency_budget)?;
    writeln!(file, "include_designer={}", config.include_designer)?;
    writeln!(file, "threshold={THRESHOLD}")?;
    writeln!(file, "decay_alpha_q8={DECAY_ALPHA_Q8}")?;
    writeln!(file, "pattern_winners_per_step={PATTERN_WINNERS_PER_STEP}")?;
    writeln!(file, "adapt_sensor=adc{ADAPT_SENSOR}")?;
    writeln!(file, "adapt_increment={ADAPT_INCREMENT}")?;
    writeln!(file, "adapt_decay_q8={ADAPT_DECAY_Q8}")?;
    writeln!(file, "adapt_max={ADAPT_MAX}")?;
    writeln!(file, "labels={}", LABELS.join(","))?;
    for (pattern, row) in model.patterns.weights.iter().enumerate() {
        let weights = row
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        writeln!(file, "pattern.{pattern:02}.name={}", pattern_name(pattern))?;
        writeln!(file, "pattern.{pattern:02}.weights={weights}")?;
    }
    for (label_index, row) in model.labels.weights.iter().enumerate() {
        let weights = row
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        writeln!(
            file,
            "label_bias.{}={}",
            LABELS[label_index], model.labels.bias[label_index]
        )?;
        writeln!(file, "label_weights.{}={weights}", LABELS[label_index])?;
    }
    Ok(())
}

fn pattern_name(pattern: usize) -> String {
    if pattern < ACTIVE_SENSORS {
        return format!("single rate adc{pattern}");
    }
    if pattern < SNN_INPUTS {
        return format!("single latency adc{}", pattern - ACTIVE_SENSORS);
    }
    if pattern < 32 {
        return format!("paired coactivation motif {:02}", pattern - 16);
    }
    if pattern < 48 {
        return format!("onset plus tail motif {:02}", pattern - 32);
    }
    format!("cluster history motif {:02}", pattern - 48)
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
        let bank = LifBank {
            weights,
            bias: [0; LABELS.len()],
        };

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
