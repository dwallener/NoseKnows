use std::cmp::Ordering;
use std::collections::VecDeque;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const CHANNELS: usize = 9;
const ACTIVE_SENSORS: usize = 8;
const PAIRS: usize = ACTIVE_SENSORS * (ACTIVE_SENSORS - 1) / 2;
const FEATURES: usize = ACTIVE_SENSORS * 2 + PAIRS * 8;
const OUTPUTS: usize = 14;
const MAX_ADC: f32 = 4095.0;
const DEFAULT_STREAM: &str = "data/streams/smoke_stream.csv";
const DEFAULT_MODEL: &str = "data/models/peak_pair_readout.npm";

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
    model_path: PathBuf,
    gate_threshold: f32,
    max_examples: usize,
}

struct StreamRow {
    segment: String,
    target: [bool; OUTPUTS],
    adc: [f32; CHANNELS],
    elapsed_ms: u64,
}

struct Frame {
    segment: String,
    target: [bool; OUTPUTS],
    bins: [u8; ACTIVE_SENSORS],
    logits: [f32; OUTPUTS],
    segment_offset: usize,
}

#[derive(Clone)]
struct PeakModel {
    weights: [[f32; FEATURES]; OUTPUTS],
    bias: [f32; OUTPUTS],
    hold_secs: f32,
}

#[derive(Clone, Copy, Default)]
struct BucketMetrics {
    frames: usize,
    emitted: usize,
    p_at_1: usize,
    any_at_3: usize,
    covered_labels: usize,
    target_labels: usize,
    silent_no_scent: usize,
    false_positive: usize,
}

#[derive(Clone, Copy, Default)]
struct LabelMetrics {
    support: usize,
    predicted: usize,
    true_positive: usize,
    false_positive: usize,
    false_negative: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("peak_stream_eval error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let model = load_model(&config.model_path)?;
    let rows = load_stream(&config.stream_path)?;
    if rows.len() < 2 {
        return Err("stream needs at least two rows".into());
    }
    let period_ms = median_period_ms(&rows);
    let hold_rows = ((model.hold_secs * 1000.0) / period_ms as f32)
        .round()
        .max(1.0) as usize;
    let frames = build_frames(&rows, &model, hold_rows);

    println!(
        "Peak stream replay: rows={} hold_secs={:.2} period_ms={} hold_rows={} features={} gate>{:.2}",
        rows.len(),
        model.hold_secs,
        period_ms,
        hold_rows,
        FEATURES,
        config.gate_threshold
    );
    println!(
        "Model path={} stream={}",
        config.model_path.display(),
        config.stream_path.display()
    );

    println!();
    println!("All frames:");
    print_report(&frames, 0, config.gate_threshold);

    println!();
    println!(
        "Settled frames only: skipping first {} row(s) of each stream segment",
        hold_rows
    );
    print_report(&frames, hold_rows, config.gate_threshold);

    println!();
    println!("Segment-level held evidence after settling:");
    print_segment_report(&frames, hold_rows, config.gate_threshold);

    print_examples(&frames, config.gate_threshold, config.max_examples);
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut stream_path = PathBuf::from(DEFAULT_STREAM);
    let mut model_path = PathBuf::from(DEFAULT_MODEL);
    let mut gate_threshold = 0.0;
    let mut max_examples = 16;

    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--stream" => {
                index += 1;
                stream_path = PathBuf::from(args.get(index).ok_or("--stream requires a path")?);
            }
            "--model" => {
                index += 1;
                model_path = PathBuf::from(args.get(index).ok_or("--model requires a path")?);
            }
            "--gate-threshold" => {
                index += 1;
                gate_threshold = args
                    .get(index)
                    .ok_or("--gate-threshold requires a value")?
                    .parse()?;
            }
            "--examples" => {
                index += 1;
                max_examples = args
                    .get(index)
                    .ok_or("--examples requires a value")?
                    .parse()?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin peak_stream_eval -- [--stream data/streams/smoke_stream.csv] [--model data/models/peak_pair_readout.npm] [--gate-threshold 0] [--examples 16]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        index += 1;
    }

    Ok(Config {
        stream_path,
        model_path,
        gate_threshold,
        max_examples,
    })
}

fn build_frames(rows: &[StreamRow], model: &PeakModel, hold_rows: usize) -> Vec<Frame> {
    let mut windows = (0..ACTIVE_SENSORS)
        .map(|_| VecDeque::<f32>::with_capacity(hold_rows))
        .collect::<Vec<_>>();
    let mut frames = Vec::with_capacity(rows.len());
    let mut current_segment = String::new();
    let mut segment_offset = 0_usize;

    for row in rows {
        if row.segment != current_segment {
            current_segment = row.segment.clone();
            segment_offset = 0;
        }

        let mut bins = [0_u8; ACTIVE_SENSORS];
        for sensor in 0..ACTIVE_SENSORS {
            let window = &mut windows[sensor];
            window.push_back(row.adc[sensor]);
            while window.len() > hold_rows {
                window.pop_front();
            }
            let peak = window.iter().fold(0.0_f32, |acc, value| acc.max(*value));
            bins[sensor] = quantize_8(peak);
        }

        let features = pairwise_features(&bins);
        frames.push(Frame {
            segment: row.segment.clone(),
            target: row.target,
            bins,
            logits: model.predict(&features),
            segment_offset,
        });
        segment_offset += 1;
    }

    frames
}

fn print_report(frames: &[Frame], skip_segment_rows: usize, gate_threshold: f32) {
    let mut buckets = [BucketMetrics::default(); 4];
    let mut labels = [LabelMetrics::default(); OUTPUTS];

    for frame in frames
        .iter()
        .filter(|frame| frame.segment_offset >= skip_segment_rows)
    {
        let active_count = frame.target.iter().filter(|active| **active).count().min(3);
        record_bucket(&mut buckets[active_count], frame, gate_threshold);
        record_labels(&mut labels, frame, gate_threshold);
    }

    println!(
        "{:<9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8} {:>8}",
        "bucket", "frames", "emitted", "silence", "fp", "p@1", "any@3", "coverage", "covered"
    );
    for (index, bucket) in buckets.iter().enumerate() {
        let name = match index {
            0 => "no-scent",
            1 => "1-note",
            2 => "2-note",
            _ => "3-note",
        };
        println!(
            "{:<9} {:>8} {:>8} {:>7.2}% {:>7.2}% {:>7.2}% {:>7.2}% {:>8.2}% {:>4}/{:<4}",
            name,
            bucket.frames,
            bucket.emitted,
            percentage(bucket.silent_no_scent, bucket.frames),
            percentage(bucket.false_positive, bucket.frames),
            percentage(bucket.p_at_1, bucket.frames),
            percentage(bucket.any_at_3, bucket.frames),
            percentage(bucket.covered_labels, bucket.target_labels),
            bucket.covered_labels,
            bucket.target_labels
        );
    }

    println!();
    println!(
        "{:<14} {:>8} {:>9} {:>7} {:>7} {:>7} {:>9} {:>8}",
        "label", "support", "predicted", "tp", "fp", "fn", "precision", "recall"
    );
    for (label, metrics) in labels.iter().enumerate() {
        println!(
            "{:<14} {:>8} {:>9} {:>7} {:>7} {:>7} {:>8.2}% {:>7.2}%",
            LABELS[label],
            metrics.support,
            metrics.predicted,
            metrics.true_positive,
            metrics.false_positive,
            metrics.false_negative,
            percentage(metrics.true_positive, metrics.predicted),
            percentage(metrics.true_positive, metrics.support)
        );
    }
}

fn print_segment_report(frames: &[Frame], skip_segment_rows: usize, gate_threshold: f32) {
    let mut buckets = [BucketMetrics::default(); 4];
    let mut labels = [LabelMetrics::default(); OUTPUTS];
    let mut index = 0;
    while index < frames.len() {
        let segment = frames[index].segment.as_str();
        let start = index;
        while index < frames.len() && frames[index].segment == segment {
            index += 1;
        }
        if let Some(summary) = summarize_segment(&frames[start..index], skip_segment_rows) {
            let active_count = summary.target.iter().filter(|active| **active).count().min(3);
            record_bucket(&mut buckets[active_count], &summary, gate_threshold);
            record_labels(&mut labels, &summary, gate_threshold);
        }
    }

    println!(
        "{:<9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8} {:>8}",
        "bucket", "segments", "emitted", "silence", "fp", "p@1", "any@3", "coverage", "covered"
    );
    for (index, bucket) in buckets.iter().enumerate() {
        let name = match index {
            0 => "no-scent",
            1 => "1-note",
            2 => "2-note",
            _ => "3-note",
        };
        println!(
            "{:<9} {:>8} {:>8} {:>7.2}% {:>7.2}% {:>7.2}% {:>7.2}% {:>8.2}% {:>4}/{:<4}",
            name,
            bucket.frames,
            bucket.emitted,
            percentage(bucket.silent_no_scent, bucket.frames),
            percentage(bucket.false_positive, bucket.frames),
            percentage(bucket.p_at_1, bucket.frames),
            percentage(bucket.any_at_3, bucket.frames),
            percentage(bucket.covered_labels, bucket.target_labels),
            bucket.covered_labels,
            bucket.target_labels
        );
    }

    println!();
    println!(
        "{:<14} {:>8} {:>9} {:>7} {:>7} {:>7} {:>9} {:>8}",
        "label", "support", "predicted", "tp", "fp", "fn", "precision", "recall"
    );
    for (label, metrics) in labels.iter().enumerate() {
        println!(
            "{:<14} {:>8} {:>9} {:>7} {:>7} {:>7} {:>8.2}% {:>7.2}%",
            LABELS[label],
            metrics.support,
            metrics.predicted,
            metrics.true_positive,
            metrics.false_positive,
            metrics.false_negative,
            percentage(metrics.true_positive, metrics.predicted),
            percentage(metrics.true_positive, metrics.support)
        );
    }
}

fn summarize_segment(frames: &[Frame], skip_segment_rows: usize) -> Option<Frame> {
    let mut selected = frames
        .iter()
        .filter(|frame| frame.segment_offset >= skip_segment_rows);
    let first = selected.next()?;
    let mut logits = first.logits;
    let mut bins = first.bins;
    for frame in selected {
        for label in 0..OUTPUTS {
            logits[label] = logits[label].max(frame.logits[label]);
        }
        for sensor in 0..ACTIVE_SENSORS {
            bins[sensor] = bins[sensor].max(frame.bins[sensor]);
        }
    }

    Some(Frame {
        segment: first.segment.clone(),
        target: first.target,
        bins,
        logits,
        segment_offset: first.segment_offset,
    })
}

fn record_bucket(bucket: &mut BucketMetrics, frame: &Frame, gate_threshold: f32) {
    bucket.frames += 1;
    let predicted = predicted_labels(&frame.logits, gate_threshold);
    if !predicted.is_empty() {
        bucket.emitted += 1;
    }

    let no_scent = frame.target.iter().all(|active| !*active);
    if no_scent {
        if predicted.is_empty() {
            bucket.silent_no_scent += 1;
        } else {
            bucket.false_positive += 1;
        }
        return;
    }

    let top = top_k(&frame.logits, 3);
    if frame.target[top[0].0] && top[0].1 > gate_threshold {
        bucket.p_at_1 += 1;
    }
    if top
        .iter()
        .any(|(label, score)| *score > gate_threshold && frame.target[*label])
    {
        bucket.any_at_3 += 1;
    }
    for (label, active) in frame.target.iter().enumerate() {
        if *active {
            bucket.target_labels += 1;
            if predicted.contains(&label) {
                bucket.covered_labels += 1;
            }
        }
    }
}

fn record_labels(labels: &mut [LabelMetrics; OUTPUTS], frame: &Frame, gate_threshold: f32) {
    let predicted = predicted_labels(&frame.logits, gate_threshold);
    for label in 0..OUTPUTS {
        if frame.target[label] {
            labels[label].support += 1;
        }
        if predicted.contains(&label) {
            labels[label].predicted += 1;
        }
        match (frame.target[label], predicted.contains(&label)) {
            (true, true) => labels[label].true_positive += 1,
            (true, false) => labels[label].false_negative += 1,
            (false, true) => labels[label].false_positive += 1,
            (false, false) => {}
        }
    }
}

fn print_examples(frames: &[Frame], gate_threshold: f32, max_examples: usize) {
    println!();
    println!("Transition examples:");
    let mut printed = 0;
    let mut last_segment = "";
    for frame in frames {
        if frame.segment == last_segment || frame.segment_offset < 5 {
            continue;
        }
        last_segment = &frame.segment;
        let expected = expected_names(&frame.target);
        let predicted = predicted_labels(&frame.logits, gate_threshold)
            .into_iter()
            .map(|label| LABELS[label].to_string())
            .collect::<Vec<_>>();
        println!(
            "{} offset={} labels=[{}] bins={:?} predicted=[{}] top=[{}]",
            frame.segment,
            frame.segment_offset,
            expected.join(", "),
            frame.bins,
            if predicted.is_empty() {
                "silent".to_string()
            } else {
                predicted.join(", ")
            },
            top_k(&frame.logits, 3)
                .into_iter()
                .map(|(label, score)| format!("{} {:.2}", LABELS[label], score))
                .collect::<Vec<_>>()
                .join(", ")
        );
        printed += 1;
        if printed >= max_examples {
            break;
        }
    }
}

fn predicted_labels(logits: &[f32; OUTPUTS], gate_threshold: f32) -> Vec<usize> {
    top_k(logits, 3)
        .into_iter()
        .filter_map(|(label, score)| (score > gate_threshold).then_some(label))
        .collect()
}

fn expected_names(target: &[bool; OUTPUTS]) -> Vec<&'static str> {
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

    features
}

impl PeakModel {
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

fn load_model(path: &Path) -> Result<PeakModel, Box<dyn std::error::Error>> {
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

fn split_once_eq(line: &str) -> Option<(&str, &str)> {
    line.split_once('=')
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
    let elapsed_index = index("host_elapsed_ms")?;
    let segment_index = header_fields
        .iter()
        .position(|field| field == "stream_segment")
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

        rows.push(StreamRow {
            segment: segment_index
                .and_then(|index| fields.get(index))
                .cloned()
                .unwrap_or_else(|| format!("row_{:010}", rows.len())),
            target,
            adc,
            elapsed_ms: fields[elapsed_index].parse::<u64>().unwrap_or(0),
        });
    }

    Ok(rows)
}

fn median_period_ms(rows: &[StreamRow]) -> u64 {
    let mut deltas = rows
        .windows(2)
        .filter_map(|pair| pair[1].elapsed_ms.checked_sub(pair[0].elapsed_ms))
        .filter(|delta| *delta > 0)
        .collect::<Vec<_>>();
    deltas.sort_unstable();
    deltas.get(deltas.len() / 2).copied().unwrap_or(100).max(1)
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

fn percentage(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 * 100.0 / denominator as f32
    }
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
    fn prediction_gate_suppresses_negative_logits() {
        let mut logits = [-1.0; OUTPUTS];
        logits[10] = 0.2;
        assert_eq!(predicted_labels(&logits, 0.0), vec![10]);
        assert!(predicted_labels(&logits, 0.3).is_empty());
    }

    #[test]
    fn pairwise_feature_shape_matches_model() {
        let bins = [0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(pairwise_features(&bins).len(), FEATURES);
    }
}
