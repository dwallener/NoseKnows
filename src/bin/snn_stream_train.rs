use std::cmp::Ordering;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const CHANNELS: usize = 9;
const ACTIVE_SENSORS: usize = 8;
const FEATURES: usize = ACTIVE_SENSORS * 2;
const PATTERN_NEURONS: usize = 64;
const OUTPUTS: usize = 14;
const MAX_ADC: f32 = 4095.0;
const CLEAN_AIR_FLOOR_ADC: f32 = 300.0;
const MIN_DELTA_ADC: f32 = 25.0;
const DEFAULT_STREAM: &str = "data/streams/snn_comprehensive_stream.csv";
const DEFAULT_OUT: &str = "data/models/snn_stream_readout.nsm";

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
    stream_path: PathBuf,
    output_path: PathBuf,
    epochs: usize,
    learning_rate: f32,
    validation_fraction: f32,
    window: usize,
    stride: usize,
    rate_budget: usize,
    latency_budget: usize,
    silence_weight: f32,
}

#[derive(Clone)]
struct StreamRow {
    labels: [String; 3],
    target: [bool; OUTPUTS],
    adc: [f32; CHANNELS],
}

struct Frame {
    target: [bool; OUTPUTS],
    primary_label: Option<usize>,
    features: [f32; PATTERN_NEURONS],
}

struct StreamModel {
    weights: [[f32; PATTERN_NEURONS]; OUTPUTS],
    bias: [f32; OUTPUTS],
}

struct PatternBank {
    weights: [[i16; FEATURES]; PATTERN_NEURONS],
}

#[derive(Default, Clone, Copy)]
struct Metrics {
    p_at_1: f32,
    any_at_3: f32,
    coverage: f32,
    silence: f32,
    false_positive: f32,
}

#[derive(Default, Clone, Copy)]
struct BucketReport {
    total: usize,
    p_at_1: usize,
    any_at_3: usize,
    covered_labels: usize,
    target_labels: usize,
    emitted: usize,
    silent: usize,
    false_positive: usize,
}

#[derive(Default, Clone, Copy)]
struct LabelReport {
    support: usize,
    predicted: usize,
    true_positive: usize,
    false_positive: usize,
    false_negative: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("snn_stream_train error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let rows = load_stream(&config.stream_path)?;
    if rows.len() < 2 {
        return Err("stream training needs at least two rows".into());
    }
    let frames = build_frames(&rows, &config);
    if frames.len() < 2 {
        return Err("stream training needs at least two usable frames after stride".into());
    }

    let validation_count = ((frames.len() as f32 * config.validation_fraction).round() as usize)
        .max(1)
        .min(frames.len() - 1);
    let train_count = frames.len() - validation_count;
    println!(
        "Loaded stream rows={} frames={} train={} validation={} window={} stride={}",
        rows.len(),
        frames.len(),
        train_count,
        validation_count,
        config.window,
        config.stride
    );
    println!(
        "Stream model: rolling spike features {} rate + {} latency -> {} accordion motifs -> {} labels; no-scent is an all-false target",
        ACTIVE_SENSORS,
        ACTIVE_SENSORS,
        PATTERN_NEURONS,
        OUTPUTS
    );

    let mut model = StreamModel::new();
    let mut best_model = model.clone();
    let mut best_validation = Metrics::default();
    for epoch in 1..=config.epochs {
        model.train_epoch(&frames[..train_count], &config);
        if epoch == 1 || epoch == config.epochs || epoch % 10 == 0 {
            let train = evaluate(&model, &frames[..train_count]);
            let validation = evaluate(&model, &frames[train_count..]);
            if better_metrics(validation, best_validation) {
                best_validation = validation;
                best_model = model.clone();
            }
            println!(
                "stream epoch {epoch:>4} train p@1 {:.2}% any@3 {:.2}% coverage {:.2}% silence {:.2}% fp {:.2}% | val p@1 {:.2}% any@3 {:.2}% coverage {:.2}% silence {:.2}% fp {:.2}%",
                train.p_at_1 * 100.0,
                train.any_at_3 * 100.0,
                train.coverage * 100.0,
                train.silence * 100.0,
                train.false_positive * 100.0,
                validation.p_at_1 * 100.0,
                validation.any_at_3 * 100.0,
                validation.coverage * 100.0,
                validation.silence * 100.0,
                validation.false_positive * 100.0
            );
        }
    }

    println!(
        "best stream validation p@1 {:.2}% any@3 {:.2}% coverage {:.2}% silence {:.2}% fp {:.2}%",
        best_validation.p_at_1 * 100.0,
        best_validation.any_at_3 * 100.0,
        best_validation.coverage * 100.0,
        best_validation.silence * 100.0,
        best_validation.false_positive * 100.0
    );
    print_validation_report(&best_model, &frames[train_count..]);
    save_model(&config.output_path, &config, &best_model)?;
    println!("Saved stream model to {}", config.output_path.display());
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut stream_path = PathBuf::from(DEFAULT_STREAM);
    let mut output_path = PathBuf::from(DEFAULT_OUT);
    let mut epochs = 50;
    let mut learning_rate = 0.8;
    let mut validation_fraction = 0.2;
    let mut window = 30;
    let mut stride = 1;
    let mut rate_budget = 5;
    let mut latency_budget = 5;
    let mut silence_weight: f32 = 3.0;

    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--stream" => {
                index += 1;
                stream_path = PathBuf::from(args.get(index).ok_or("--stream requires a path")?);
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
            "--window" => {
                index += 1;
                window = args
                    .get(index)
                    .ok_or("--window requires a value")?
                    .parse()?;
            }
            "--stride" => {
                index += 1;
                stride = args
                    .get(index)
                    .ok_or("--stride requires a value")?
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
            "--silence-weight" => {
                index += 1;
                silence_weight = args
                    .get(index)
                    .ok_or("--silence-weight requires a value")?
                    .parse()?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin snn_stream_train -- [--stream data/streams/snn_comprehensive_stream.csv] [--out data/models/snn_stream_readout.nsm] [--epochs 50] [--window 30] [--stride 1] [--silence-weight 3.0]"
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
        stream_path,
        output_path,
        epochs,
        learning_rate,
        validation_fraction,
        window: window.max(1),
        stride: stride.max(1),
        rate_budget: rate_budget.max(1),
        latency_budget: latency_budget.max(1),
        silence_weight: silence_weight.max(0.0),
    })
}

fn build_frames(rows: &[StreamRow], config: &Config) -> Vec<Frame> {
    let mut rolling = [0.0_f32; FEATURES];
    let mut history = Vec::new();
    let mut frames = Vec::new();
    let mut previous_adc = rows[0].adc;
    let patterns = PatternBank::seeded();

    for (row_index, row) in rows.iter().enumerate() {
        let mut instant = [0.0_f32; FEATURES];
        for sensor in 0..ACTIVE_SENSORS {
            let amplitude = ((row.adc[sensor] - CLEAN_AIR_FLOOR_ADC)
                / (MAX_ADC - CLEAN_AIR_FLOOR_ADC))
                .clamp(0.0, 1.0);
            instant[sensor] = rate_feature(amplitude, config.rate_budget);

            let delta = row.adc[sensor] - previous_adc[sensor];
            instant[ACTIVE_SENSORS + sensor] = latency_feature(delta, config.latency_budget);
        }
        previous_adc = row.adc;

        for feature in 0..FEATURES {
            rolling[feature] += instant[feature];
        }
        history.push(instant);
        if history.len() > config.window {
            let expired = history.remove(0);
            for feature in 0..FEATURES {
                rolling[feature] -= expired[feature];
            }
        }

        if row_index % config.stride == 0 {
            let divisor = history.len().max(1) as f32;
            let mut features = rolling;
            for feature in &mut features {
                *feature /= divisor;
            }
            frames.push(Frame {
                target: row.target,
                primary_label: primary_label(&row.labels),
                features: patterns.forward(&features),
            });
        }
    }

    frames
}

fn rate_feature(amplitude: f32, budget: usize) -> f32 {
    if amplitude <= 0.0 {
        return 0.0;
    }
    let scaled = ((1.0 + amplitude * 9.0).ln() / 10.0_f32.ln()).clamp(0.0, 1.0);
    let count = (scaled * budget as f32).round();
    count / budget as f32
}

fn latency_feature(delta_adc: f32, budget: usize) -> f32 {
    if delta_adc <= MIN_DELTA_ADC {
        return 0.0;
    }
    let scaled = (delta_adc / 350.0).clamp(0.0, 1.0);
    let count = (scaled * budget as f32).ceil();
    count / budget as f32
}

impl StreamModel {
    fn new() -> Self {
        Self {
            weights: [[0.0; PATTERN_NEURONS]; OUTPUTS],
            bias: [0.0; OUTPUTS],
        }
    }

    fn train_epoch(&mut self, frames: &[Frame], config: &Config) {
        for frame in frames {
            let logits = self.predict(&frame.features);
            let no_scent = is_no_scent_target(&frame.target);
            for label in 0..OUTPUTS {
                let target = if frame.target[label] { 1.0 } else { 0.0 };
                let weight = if frame.target[label] {
                    3.0
                } else if no_scent {
                    config.silence_weight
                } else {
                    1.0
                };
                let error = target - sigmoid(logits[label]);
                let update = config.learning_rate * weight * error;
                self.bias[label] = (self.bias[label] + update * 0.01).clamp(-12.0, 12.0);
                for feature in 0..PATTERN_NEURONS {
                    self.weights[label][feature] = (self.weights[label][feature]
                        + update * frame.features[feature])
                        .clamp(-24.0, 24.0);
                }
            }
        }
    }

    fn predict(&self, features: &[f32; PATTERN_NEURONS]) -> [f32; OUTPUTS] {
        let mut logits = self.bias;
        for (label, logit) in logits.iter_mut().enumerate() {
            for (feature, value) in features.iter().enumerate() {
                *logit += self.weights[label][feature] * value;
            }
        }
        logits
    }
}

impl PatternBank {
    fn seeded() -> Self {
        let mut weights = [[-60_i16; FEATURES]; PATTERN_NEURONS];

        for input in 0..FEATURES {
            weights[input] = [-90; FEATURES];
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
            set_sensor_motif(&mut weights[16 + offset], sensors, 500, 420, -80);
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
            weights[pattern] = [-85; FEATURES];
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

    fn forward(&self, features: &[f32; FEATURES]) -> [f32; PATTERN_NEURONS] {
        let mut values = [0.0_f32; PATTERN_NEURONS];
        for (pattern, value) in values.iter_mut().enumerate() {
            let mut weighted_sum = 0.0;
            for (feature, feature_value) in features.iter().enumerate() {
                weighted_sum += self.weights[pattern][feature] as f32 * feature_value;
            }
            *value = (weighted_sum / 1200.0).clamp(0.0, 1.0);
        }
        values
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
    weights: &mut [i16; FEATURES],
    sensors: &[usize],
    rate_weight: i16,
    latency_weight: i16,
    background: i16,
) {
    *weights = [background; FEATURES];
    for sensor in sensors {
        weights[*sensor] = rate_weight;
        weights[ACTIVE_SENSORS + *sensor] = latency_weight;
    }
}

impl Clone for StreamModel {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights,
            bias: self.bias,
        }
    }
}

fn evaluate(model: &StreamModel, frames: &[Frame]) -> Metrics {
    if frames.is_empty() {
        return Metrics::default();
    }

    let mut metrics = Metrics::default();
    let mut active_count = 0.0_f32;
    let mut no_scent_count = 0.0_f32;
    let mut coverage_slots = 0.0_f32;
    for frame in frames {
        let logits = model.predict(&frame.features);
        if is_no_scent_target(&frame.target) {
            no_scent_count += 1.0;
            if logits.iter().all(|value| *value <= 0.0) {
                metrics.silence += 1.0;
            } else {
                metrics.false_positive += 1.0;
            }
            continue;
        }

        active_count += 1.0;
        let top = top_k(&logits, 3);
        if frame.primary_label == Some(top[0].0) {
            metrics.p_at_1 += 1.0;
        }
        if top.iter().any(|(label, _)| frame.target[*label]) {
            metrics.any_at_3 += 1.0;
        }
        for label in 0..OUTPUTS {
            if frame.target[label] {
                coverage_slots += 1.0;
                if top.iter().any(|(top_label, _)| *top_label == label) {
                    metrics.coverage += 1.0;
                }
            }
        }
    }

    if active_count > 0.0 {
        metrics.p_at_1 /= active_count;
        metrics.any_at_3 /= active_count;
    }
    if coverage_slots > 0.0 {
        metrics.coverage /= coverage_slots;
    }
    if no_scent_count > 0.0 {
        metrics.silence /= no_scent_count;
        metrics.false_positive /= no_scent_count;
    }
    metrics
}

fn better_metrics(candidate: Metrics, current: Metrics) -> bool {
    let candidate_score = candidate.silence * 2.0 + candidate.coverage + candidate.any_at_3
        - candidate.false_positive;
    let current_score =
        current.silence * 2.0 + current.coverage + current.any_at_3 - current.false_positive;
    candidate_score > current_score
}

fn print_validation_report(model: &StreamModel, frames: &[Frame]) {
    let (buckets, labels) = validation_report(model, frames);

    println!();
    println!("Validation bucket breakdown:");
    println!("bucket      frames emitted silence     fp    p@1   any@3 coverage targets covered");
    for (bucket, report) in buckets.iter().enumerate() {
        let name = match bucket {
            0 => "no-scent",
            1 => "1-note",
            2 => "2-note",
            3 => "3-note",
            _ => "other",
        };
        let active_total = if bucket == 0 { 0 } else { report.total };
        println!(
            "{name:<9} {:>7} {:>7} {:>7.2}% {:>6.2}% {:>6.2}% {:>7.2}% {:>8.2}% {:>7} {:>7}",
            report.total,
            report.emitted,
            percentage(report.silent, report.total),
            percentage(report.false_positive, report.total),
            percentage(report.p_at_1, active_total),
            percentage(report.any_at_3, active_total),
            percentage(report.covered_labels, report.target_labels),
            report.target_labels,
            report.covered_labels
        );
    }

    println!();
    println!("Validation label breakdown:");
    println!("label          support predicted      tp      fp      fn precision recall");
    for (label, report) in labels.iter().enumerate() {
        println!(
            "{:<13} {:>7} {:>9} {:>7} {:>7} {:>7} {:>8.2}% {:>6.2}%",
            LABELS[label],
            report.support,
            report.predicted,
            report.true_positive,
            report.false_positive,
            report.false_negative,
            percentage(report.true_positive, report.predicted),
            percentage(report.true_positive, report.support)
        );
    }

    let mut misses = labels
        .iter()
        .enumerate()
        .map(|(label, report)| (label, report.false_negative))
        .filter(|(_, count)| *count > 0)
        .collect::<Vec<_>>();
    misses.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| LABELS[a.0].cmp(LABELS[b.0])));
    if !misses.is_empty() {
        println!();
        println!("Top validation misses:");
        for (rank, (label, count)) in misses.iter().take(8).enumerate() {
            println!("{:>2}. {:<13} missed={}", rank + 1, LABELS[*label], count);
        }
    }

    let mut false_positives = labels
        .iter()
        .enumerate()
        .map(|(label, report)| (label, report.false_positive))
        .filter(|(_, count)| *count > 0)
        .collect::<Vec<_>>();
    false_positives.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| LABELS[a.0].cmp(LABELS[b.0])));
    if !false_positives.is_empty() {
        println!();
        println!("Top validation false positives:");
        for (rank, (label, count)) in false_positives.iter().take(8).enumerate() {
            println!("{:>2}. {:<13} fp={}", rank + 1, LABELS[*label], count);
        }
    }
    println!();
}

fn validation_report(
    model: &StreamModel,
    frames: &[Frame],
) -> ([BucketReport; 4], [LabelReport; OUTPUTS]) {
    let mut buckets = [BucketReport::default(); 4];
    let mut labels = [LabelReport::default(); OUTPUTS];

    for frame in frames {
        let logits = model.predict(&frame.features);
        let top = top_k(&logits, 3);
        let predicted = top
            .iter()
            .filter_map(|(label, score)| if *score > 0.0 { Some(*label) } else { None })
            .collect::<Vec<_>>();
        let target_count = frame.target.iter().filter(|value| **value).count();
        let bucket = target_count.min(3);
        let report = &mut buckets[bucket];
        report.total += 1;
        report.target_labels += target_count;
        if predicted.is_empty() {
            report.silent += 1;
        } else {
            report.emitted += 1;
        }

        if target_count == 0 {
            if !predicted.is_empty() {
                report.false_positive += 1;
            }
        } else {
            if frame.primary_label == Some(top[0].0) {
                report.p_at_1 += 1;
            }
            if top.iter().any(|(label, _)| frame.target[*label]) {
                report.any_at_3 += 1;
            }
            for label in 0..OUTPUTS {
                if frame.target[label] && top.iter().any(|(top_label, _)| *top_label == label) {
                    report.covered_labels += 1;
                }
            }
        }

        for label in 0..OUTPUTS {
            let is_target = frame.target[label];
            let is_predicted = predicted.contains(&label);
            if is_target {
                labels[label].support += 1;
            }
            if is_predicted {
                labels[label].predicted += 1;
            }
            match (is_target, is_predicted) {
                (true, true) => labels[label].true_positive += 1,
                (true, false) => labels[label].false_negative += 1,
                (false, true) => labels[label].false_positive += 1,
                (false, false) => {}
            }
        }
    }

    (buckets, labels)
}

fn percentage(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 * 100.0 / denominator as f32
    }
}

fn load_stream(path: &Path) -> Result<Vec<StreamRow>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = lines.next().ok_or("stream CSV is empty")?;
    let header_fields = parse_csv_line(header);
    let index = |name: &str| -> Result<usize, Box<dyn std::error::Error>> {
        header_fields
            .iter()
            .position(|field| field == name)
            .ok_or_else(|| format!("{} missing column {name}", path.display()).into())
    };

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
        rows.push(StreamRow {
            labels,
            target,
            adc,
        });
    }
    Ok(rows)
}

fn save_model(
    output_path: &Path,
    config: &Config,
    model: &StreamModel,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = fs::File::create(output_path)?;
    writeln!(file, "NOSEKNOWS_SNN_STREAM_READOUT_V1")?;
    writeln!(file, "features={PATTERN_NEURONS}")?;
    writeln!(file, "input_features={FEATURES}")?;
    writeln!(file, "patterns={PATTERN_NEURONS}")?;
    writeln!(file, "outputs={OUTPUTS}")?;
    writeln!(file, "window={}", config.window)?;
    writeln!(file, "stride={}", config.stride)?;
    writeln!(file, "rate_budget={}", config.rate_budget)?;
    writeln!(file, "latency_budget={}", config.latency_budget)?;
    writeln!(file, "silence_weight={}", config.silence_weight)?;
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

fn primary_label(labels: &[String; 3]) -> Option<usize> {
    labels.iter().find_map(|label| label_index(label))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_feature_is_budget_normalized() {
        assert_eq!(rate_feature(0.0, 5), 0.0);
        assert!(rate_feature(0.9, 5) <= 1.0);
        assert!(rate_feature(0.9, 5) > rate_feature(0.1, 5));
    }

    #[test]
    fn latency_feature_ignores_small_deltas() {
        assert_eq!(latency_feature(10.0, 5), 0.0);
        assert!(latency_feature(350.0, 5) > 0.0);
    }
}
