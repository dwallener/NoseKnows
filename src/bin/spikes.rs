use std::env;
use std::fs;
use std::path::PathBuf;

const CHANNELS: usize = 9;
const CHANNEL_NAMES: [&str; CHANNELS] = [
    "adc0 / IO1 / MQ-2",
    "adc1 / IO2 / MQ-3",
    "adc2 / IO16 / MQ-5",
    "adc3 / IO17 / MQ-6",
    "adc4 / IO18 / MQ-7",
    "adc5 / IO21 / MQ-8",
    "adc6 / IO22 / MQ-9",
    "adc7 / IO23 / MQ-135",
    "adc8 / MQ-4 placeholder",
];
const DEFAULT_INPUT: &str = "data/raw/synthetic_0000.csv";
const DEFAULT_OUTPUT: &str = "data/spikes.svg";
const DEFAULT_MODEL: &str = "data/models/snn_lif.nsm";
const DEFAULT_BINS: usize = 180;
const DEFAULT_SUBSLOTS: usize = 5;
const DEFAULT_RATE_BUDGET: usize = 5;
const DEFAULT_LATENCY_BUDGET: usize = 5;
const DEFAULT_GATE_MIN_TOP: usize = 3;
const DEFAULT_GATE_MARGIN: isize = 1;
const DEFAULT_GATE_MIN_ACTIVITY: usize = 12;
const DEFAULT_GATE_WINDOW_SAMPLES: usize = 6;
const ACTIVE_SENSORS: usize = 8;
const SNN_INPUTS: usize = ACTIVE_SENSORS * 2;
const PATTERN_NEURONS: usize = 64;
const PATTERN_WINNERS_PER_STEP: usize = 3;
const SNN_OUTPUTS: usize = 14;
const THRESHOLD: i32 = 1000;
const DECAY_ALPHA_Q8: i32 = 235;
const ADAPT_SENSOR: usize = 4;
const ADAPT_INCREMENT: i32 = 450;
const ADAPT_DECAY_Q8: i32 = 224;
const ADAPT_MAX: i32 = 2400;
const MIN_MEMBRANE: i32 = -3000;
const MAX_ADC: f32 = 4095.0;
const MIN_SIGNAL_RANGE_ADC: f32 = 80.0;
const MIN_DELTA_ADC: f32 = 25.0;
const LABEL_FLORAL: usize = 0;
const LABEL_FLORAL_AMBER: usize = 2;
const LABEL_AMBER: usize = 3;
const LABEL_WOODY_AMBER: usize = 5;
const LABEL_DRY_WOODS: usize = 8;
const LABEL_WATER: usize = 11;
const LABEL_GREEN: usize = 12;
const BASE_GATE_MIN_TOP: usize = 2;
const BASE_GATE_WINDOW_SAMPLES: usize = 18;
const TOP_NOTE_INHIBITION: i32 = 1000;
const LABELS: [&str; SNN_OUTPUTS] = [
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
struct Sample {
    id: String,
    name: String,
    labels: [String; 3],
    rows: Vec<[f32; CHANNELS]>,
}

#[derive(Clone, Copy)]
enum Encoding {
    Rate,
    Latency,
    Mixed,
}

#[derive(Clone, Copy)]
enum SpikeStream {
    Rate,
    Latency,
}

struct Config {
    input: PathBuf,
    output: PathBuf,
    model: PathBuf,
    bins: usize,
    subslots: usize,
    rate_budget: usize,
    latency_budget: usize,
    gate_min_top: usize,
    gate_margin: isize,
    gate_min_activity: usize,
    gate_window_samples: usize,
}

struct SpikeEvent {
    sample_index: usize,
    subslot: usize,
    channel: usize,
    stream: SpikeStream,
    slot: usize,
}

struct SpikeView {
    rate: Vec<SpikeEvent>,
    latency: Vec<SpikeEvent>,
    mixed: Vec<SpikeEvent>,
    pattern: Option<Vec<PatternSpike>>,
    pattern_names: Option<Vec<String>>,
    output: Option<Vec<OutputSpike>>,
    gated: Option<Vec<GatedDecision>>,
}

struct PatternSpike {
    sample_index: usize,
    subslot: usize,
    pattern: usize,
}

struct OutputSpike {
    sample_index: usize,
    subslot: usize,
    label: usize,
}

struct GatedDecision {
    sample_index: usize,
    subslot: usize,
    label: usize,
    rank: usize,
    score: usize,
}

enum SnnModel {
    Direct(DirectLifModel),
    Accordion(AccordionLifModel),
}

struct DirectLifModel {
    weights: [[i16; SNN_INPUTS]; SNN_OUTPUTS],
    bias: [i16; SNN_OUTPUTS],
    threshold: i32,
    decay_alpha_q8: i32,
}

struct AccordionLifModel {
    pattern_weights: [[i16; SNN_INPUTS]; PATTERN_NEURONS],
    label_weights: [[i16; PATTERN_NEURONS]; SNN_OUTPUTS],
    label_bias: [i16; SNN_OUTPUTS],
    pattern_names: Vec<String>,
    threshold: i32,
    decay_alpha_q8: i32,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("spikes error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let sample = load_sample(&config.input)?;
    if sample.rows.len() < 2 {
        return Err("sample must contain at least two sensor rows".into());
    }

    let bins = config.bins.min(sample.rows.len()).max(8);
    let subslots = config.subslots.max(1);
    let binned = bin_rows(&sample.rows, bins);
    let view = SpikeView {
        rate: encode_spikes(&binned, Encoding::Rate, &config),
        latency: encode_spikes(&binned, Encoding::Latency, &config),
        mixed: encode_spikes(&binned, Encoding::Mixed, &config),
        pattern: None,
        pattern_names: None,
        output: None,
        gated: None,
    };
    let model = load_snn_model(&config.model)?;
    let input_masks = input_masks_from_events(&view.mixed, bins, subslots);
    let view = match model {
        SnnModel::Direct(model) => {
            let output_spikes = model.forward_spikes(&input_masks, subslots);
            let gated = gated_decisions(
                &output_spikes,
                None,
                Some(&view.mixed),
                bins,
                subslots,
                &config,
            );
            SpikeView {
                output: Some(output_spikes),
                gated: Some(gated),
                ..view
            }
        }
        SnnModel::Accordion(model) => {
            let pattern_masks = model.forward_pattern_masks(&input_masks);
            let pattern_spikes = pattern_spikes_from_masks(&pattern_masks, subslots);
            let output_spikes = model.forward_label_spikes(&pattern_masks, subslots);
            let gated = gated_decisions(
                &output_spikes,
                Some(&pattern_spikes),
                Some(&view.mixed),
                bins,
                subslots,
                &config,
            );
            print_accordion_contribution_summary(&model, &pattern_spikes, &gated, &sample.labels);
            SpikeView {
                pattern: Some(pattern_spikes),
                pattern_names: Some(model.pattern_names),
                output: Some(output_spikes),
                gated: Some(gated),
                ..view
            }
        }
    };
    let svg = render_svg(&sample, &view, bins, subslots);
    if let Some(parent) = config.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&config.output, svg)?;

    println!(
        "Wrote spike-train view for {} labels=[{}, {}, {}] to {}",
        sample.id,
        sample.labels[0],
        sample.labels[1],
        sample.labels[2],
        config.output.display()
    );
    println!(
        "budgets subslots={} rate_budget={} latency_budget={} mixed_max_events_per_sample={}",
        subslots,
        config.rate_budget,
        config.latency_budget,
        ACTIVE_SENSORS * (config.rate_budget + config.latency_budget)
    );
    print_spike_summary("rate", &view.rate);
    print_spike_summary("latency", &view.latency);
    print_spike_summary("mixed", &view.mixed);
    if let Some(pattern) = &view.pattern {
        print_pattern_summary("pattern", pattern);
    }
    if let Some(output) = &view.output {
        print_output_summary("output", output);
    }
    if let Some(gated) = &view.gated {
        print_gated_summary("gated", gated);
    }
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut input = PathBuf::from(DEFAULT_INPUT);
    let mut output = PathBuf::from(DEFAULT_OUTPUT);
    let mut model = PathBuf::from(DEFAULT_MODEL);
    let mut bins = DEFAULT_BINS;
    let mut subslots = DEFAULT_SUBSLOTS;
    let mut rate_budget = DEFAULT_RATE_BUDGET;
    let mut latency_budget = DEFAULT_LATENCY_BUDGET;
    let mut gate_min_top = DEFAULT_GATE_MIN_TOP;
    let mut gate_margin = DEFAULT_GATE_MARGIN;
    let mut gate_min_activity = DEFAULT_GATE_MIN_ACTIVITY;
    let mut gate_window_samples = DEFAULT_GATE_WINDOW_SAMPLES;

    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--input" => {
                index += 1;
                input = PathBuf::from(args.get(index).ok_or("--input requires a path")?);
            }
            "--out" => {
                index += 1;
                output = PathBuf::from(args.get(index).ok_or("--out requires a path")?);
            }
            "--model" => {
                index += 1;
                model = PathBuf::from(args.get(index).ok_or("--model requires a path")?);
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
            "--gate-min-top" => {
                index += 1;
                gate_min_top = args
                    .get(index)
                    .ok_or("--gate-min-top requires a value")?
                    .parse()?;
            }
            "--gate-margin" => {
                index += 1;
                gate_margin = args
                    .get(index)
                    .ok_or("--gate-margin requires a value")?
                    .parse()?;
            }
            "--gate-min-activity" => {
                index += 1;
                gate_min_activity = args
                    .get(index)
                    .ok_or("--gate-min-activity requires a value")?
                    .parse()?;
            }
            "--gate-window" => {
                index += 1;
                gate_window_samples = args
                    .get(index)
                    .ok_or("--gate-window requires a value")?
                    .parse()?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin spikes -- [--input data/raw/synthetic_0000.csv] [--out data/spikes.svg] [--model data/models/snn_lif.nsm] [--bins 180] [--subslots 5] [--rate-budget 5] [--latency-budget 5] [--gate-min-top 3] [--gate-margin 1] [--gate-min-activity 12] [--gate-window 6]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        index += 1;
    }

    Ok(Config {
        input,
        output,
        model,
        bins,
        subslots,
        rate_budget,
        latency_budget,
        gate_min_top,
        gate_margin,
        gate_min_activity,
        gate_window_samples: gate_window_samples.max(1),
    })
}

fn load_sample(path: &PathBuf) -> Result<Sample, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| format!("{} is empty", path.display()))?;
    let header_fields = parse_csv_line(header);
    let index = |name: &str| -> Result<usize, Box<dyn std::error::Error>> {
        header_fields
            .iter()
            .position(|field| field == name)
            .ok_or_else(|| format!("{} missing column {name}", path.display()).into())
    };

    let id_index = index("sample_id")?;
    let name_index = index("sample_name")?;
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

    let mut sample = Sample {
        id: String::new(),
        name: String::new(),
        labels: [String::new(), String::new(), String::new()],
        rows: Vec::new(),
    };

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_line(line);
        if fields.len() <= *adc_indexes.iter().max().expect("adc indexes") {
            continue;
        }

        if sample.id.is_empty() {
            sample.id = fields[id_index].clone();
            sample.name = fields[name_index].clone();
            sample.labels = [
                fields[label_indexes[0]].clone(),
                fields[label_indexes[1]].clone(),
                fields[label_indexes[2]].clone(),
            ];
        }

        let mut row = [0.0_f32; CHANNELS];
        for (channel, field_index) in adc_indexes.iter().enumerate() {
            row[channel] = fields[*field_index].parse::<f32>().unwrap_or(0.0);
        }
        sample.rows.push(row);
    }

    if sample.id.is_empty() {
        return Err(format!("{} contains no usable rows", path.display()).into());
    }
    Ok(sample)
}

fn bin_rows(rows: &[[f32; CHANNELS]], bins: usize) -> Vec<[f32; CHANNELS]> {
    let mut binned = Vec::with_capacity(bins);
    for bin in 0..bins {
        let start = bin * rows.len() / bins;
        let end = ((bin + 1) * rows.len() / bins).max(start + 1);
        let mut row = [0.0_f32; CHANNELS];
        for source in &rows[start..end.min(rows.len())] {
            for channel in 0..CHANNELS {
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

fn encode_spikes(rows: &[[f32; CHANNELS]], encoding: Encoding, config: &Config) -> Vec<SpikeEvent> {
    let mut events = Vec::new();
    let subslots = config.subslots.max(1);
    for channel in 0..CHANNELS {
        let baseline = rows
            .iter()
            .take(rows.len().min(10))
            .map(|row| row[channel])
            .sum::<f32>()
            / rows.len().min(10) as f32;
        let peak = rows
            .iter()
            .map(|row| row[channel])
            .fold(baseline, f32::max)
            .max(baseline + 1.0);
        let mut previous = rows[0][channel];

        for (index, row) in rows.iter().enumerate() {
            let signal_range = peak - baseline;
            let amplitude = if signal_range >= MIN_SIGNAL_RANGE_ADC {
                ((row[channel] - baseline) / signal_range).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let delta = if index == 0 {
                0.0
            } else {
                let raw_delta = (row[channel] - previous).max(0.0);
                if raw_delta >= MIN_DELTA_ADC {
                    raw_delta / MAX_ADC
                } else {
                    0.0
                }
            };
            if matches!(encoding, Encoding::Rate | Encoding::Mixed) {
                for slot in 0..rate_spike_count(amplitude, config.rate_budget) {
                    events.push(SpikeEvent {
                        sample_index: index,
                        subslot: rate_subslot(slot, config.rate_budget, subslots),
                        channel,
                        stream: SpikeStream::Rate,
                        slot,
                    });
                }
            }
            if matches!(encoding, Encoding::Latency | Encoding::Mixed) {
                if let Some(subslot) = latency_subslot(delta, subslots) {
                    for slot in 0..config.latency_budget {
                        events.push(SpikeEvent {
                            sample_index: index,
                            subslot: (subslot + slot).min(subslots - 1),
                            channel,
                            stream: SpikeStream::Latency,
                            slot,
                        });
                    }
                }
            }
            previous = row[channel];
        }
    }
    events
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

fn input_masks_from_events(events: &[SpikeEvent], bins: usize, subslots: usize) -> Vec<u16> {
    let mut masks = vec![0_u16; bins * subslots.max(1)];
    for event in events {
        if event.channel >= ACTIVE_SENSORS {
            continue;
        }
        let bit = match event.stream {
            SpikeStream::Rate => event.channel,
            SpikeStream::Latency => ACTIVE_SENSORS + event.channel,
        };
        let index = event.sample_index * subslots.max(1) + event.subslot.min(subslots.max(1) - 1);
        if let Some(mask) = masks.get_mut(index) {
            *mask |= 1 << bit;
        }
    }
    masks
}

fn load_snn_model(path: &PathBuf) -> Result<SnnModel, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    if text.lines().next() == Some("NOSEKNOWS_SNN_ACCORDION_V1") {
        load_accordion_model(path, &text)
    } else {
        load_direct_model(path, &text)
    }
}

fn load_direct_model(path: &PathBuf, text: &str) -> Result<SnnModel, Box<dyn std::error::Error>> {
    let mut weights = [[0_i16; SNN_INPUTS]; SNN_OUTPUTS];
    let mut bias = [0_i16; SNN_OUTPUTS];
    let mut threshold = THRESHOLD;
    let mut decay_alpha_q8 = DECAY_ALPHA_Q8;

    for line in text.lines() {
        if let Some(value) = line.strip_prefix("threshold=") {
            threshold = value.parse()?;
        } else if let Some(value) = line.strip_prefix("decay_alpha_q8=") {
            decay_alpha_q8 = value.parse()?;
        } else if let Some(rest) = line.strip_prefix("bias.") {
            let (label, value) = rest
                .split_once('=')
                .ok_or_else(|| format!("invalid bias line in {}", path.display()))?;
            let label_index = LABELS
                .iter()
                .position(|candidate| *candidate == label)
                .ok_or_else(|| format!("unknown label in model: {label}"))?;
            bias[label_index] = value.parse()?;
        } else if let Some(rest) = line.strip_prefix("weights.") {
            let (label, values) = rest
                .split_once('=')
                .ok_or_else(|| format!("invalid weights line in {}", path.display()))?;
            let label_index = LABELS
                .iter()
                .position(|candidate| *candidate == label)
                .ok_or_else(|| format!("unknown label in model: {label}"))?;
            let parsed = values
                .split(',')
                .map(str::parse::<i16>)
                .collect::<Result<Vec<_>, _>>()?;
            if parsed.len() != SNN_INPUTS {
                return Err(format!(
                    "{} weights for {label} expected {SNN_INPUTS}, got {}",
                    path.display(),
                    parsed.len()
                )
                .into());
            }
            weights[label_index].copy_from_slice(&parsed);
        }
    }

    Ok(SnnModel::Direct(DirectLifModel {
        weights,
        bias,
        threshold,
        decay_alpha_q8,
    }))
}

fn load_accordion_model(
    path: &PathBuf,
    text: &str,
) -> Result<SnnModel, Box<dyn std::error::Error>> {
    let mut pattern_weights = [[0_i16; SNN_INPUTS]; PATTERN_NEURONS];
    let mut label_weights = [[0_i16; PATTERN_NEURONS]; SNN_OUTPUTS];
    let mut label_bias = [0_i16; SNN_OUTPUTS];
    let mut pattern_names = (0..PATTERN_NEURONS)
        .map(|index| format!("pattern_{index:02}"))
        .collect::<Vec<_>>();
    let mut threshold = THRESHOLD;
    let mut decay_alpha_q8 = DECAY_ALPHA_Q8;

    for line in text.lines() {
        if let Some(value) = line.strip_prefix("threshold=") {
            threshold = value.parse()?;
        } else if let Some(value) = line.strip_prefix("decay_alpha_q8=") {
            decay_alpha_q8 = value.parse()?;
        } else if let Some(rest) = line.strip_prefix("pattern.") {
            let (head, value) = rest
                .split_once('=')
                .ok_or_else(|| format!("invalid pattern line in {}", path.display()))?;
            let (index_text, field) = head
                .split_once('.')
                .ok_or_else(|| format!("invalid pattern key in {}", path.display()))?;
            let pattern: usize = index_text.parse()?;
            if pattern >= PATTERN_NEURONS {
                return Err(format!("pattern index {pattern} out of range").into());
            }
            match field {
                "name" => pattern_names[pattern] = value.to_string(),
                "weights" => {
                    let parsed = value
                        .split(',')
                        .map(str::parse::<i16>)
                        .collect::<Result<Vec<_>, _>>()?;
                    if parsed.len() != SNN_INPUTS {
                        return Err(format!(
                            "{} pattern {pattern} expected {SNN_INPUTS}, got {}",
                            path.display(),
                            parsed.len()
                        )
                        .into());
                    }
                    pattern_weights[pattern].copy_from_slice(&parsed);
                }
                _ => {}
            }
        } else if let Some(rest) = line.strip_prefix("label_bias.") {
            let (label, value) = rest
                .split_once('=')
                .ok_or_else(|| format!("invalid label_bias line in {}", path.display()))?;
            let label_index = LABELS
                .iter()
                .position(|candidate| *candidate == label)
                .ok_or_else(|| format!("unknown label in model: {label}"))?;
            label_bias[label_index] = value.parse()?;
        } else if let Some(rest) = line.strip_prefix("label_weights.") {
            let (label, values) = rest
                .split_once('=')
                .ok_or_else(|| format!("invalid label_weights line in {}", path.display()))?;
            let label_index = LABELS
                .iter()
                .position(|candidate| *candidate == label)
                .ok_or_else(|| format!("unknown label in model: {label}"))?;
            let parsed = values
                .split(',')
                .map(str::parse::<i16>)
                .collect::<Result<Vec<_>, _>>()?;
            if parsed.len() != PATTERN_NEURONS {
                return Err(format!(
                    "{} label weights for {label} expected {PATTERN_NEURONS}, got {}",
                    path.display(),
                    parsed.len()
                )
                .into());
            }
            label_weights[label_index].copy_from_slice(&parsed);
        }
    }

    Ok(SnnModel::Accordion(AccordionLifModel {
        pattern_weights,
        label_weights,
        label_bias,
        pattern_names,
        threshold,
        decay_alpha_q8,
    }))
}

impl DirectLifModel {
    fn forward_spikes(&self, masks: &[u16], subslots: usize) -> Vec<OutputSpike> {
        let mut membrane = [0_i32; SNN_OUTPUTS];
        let mut spikes = Vec::new();

        for (step, mask) in masks.iter().enumerate() {
            let sample_index = step / subslots.max(1);
            let subslot = step % subslots.max(1);
            let mut next_values = [0_i32; SNN_OUTPUTS];
            for label in 0..SNN_OUTPUTS {
                let mut next =
                    ((membrane[label] * self.decay_alpha_q8) >> 8) + self.bias[label] as i32;
                for input in 0..SNN_INPUTS {
                    if ((mask >> input) & 1) != 0 {
                        next += self.weights[label][input] as i32;
                    }
                }
                next_values[label] = next;
            }
            apply_label_inhibition(&mut next_values, self.threshold);

            for label in 0..SNN_OUTPUTS {
                let next = next_values[label];
                if next >= self.threshold {
                    spikes.push(OutputSpike {
                        sample_index,
                        subslot,
                        label,
                    });
                    membrane[label] = 0;
                } else {
                    membrane[label] = next.clamp(MIN_MEMBRANE, self.threshold - 1);
                }
            }
        }

        spikes
    }
}

impl AccordionLifModel {
    fn forward_pattern_masks(&self, input_masks: &[u16]) -> Vec<u64> {
        let mut membrane = [0_i32; PATTERN_NEURONS];
        let mut adaptation = [0_i32; PATTERN_NEURONS];
        let mut pattern_masks = Vec::with_capacity(input_masks.len());

        for input_mask in input_masks {
            let mut next_membrane = [0_i32; PATTERN_NEURONS];
            let mut candidates = Vec::new();
            for pattern in 0..PATTERN_NEURONS {
                let mut next = (membrane[pattern] * self.decay_alpha_q8) >> 8;
                for input in 0..SNN_INPUTS {
                    if ((input_mask >> input) & 1) != 0 {
                        next += self.pattern_weights[pattern][input] as i32;
                    }
                }
                next_membrane[pattern] = next;
                if next >= self.threshold + adaptation[pattern] {
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
                    next_membrane[*pattern] -= self.threshold / 2;
                }
            }

            for pattern in 0..PATTERN_NEURONS {
                membrane[pattern] = next_membrane[pattern].clamp(MIN_MEMBRANE, self.threshold - 1);
                adaptation[pattern] = (adaptation[pattern] * ADAPT_DECAY_Q8) >> 8;
                if ((mask >> pattern) & 1) != 0
                    && pattern_uses_sensor(&self.pattern_weights[pattern], ADAPT_SENSOR)
                {
                    adaptation[pattern] = (adaptation[pattern] + ADAPT_INCREMENT).min(ADAPT_MAX);
                }
            }
            pattern_masks.push(mask);
        }

        pattern_masks
    }

    fn forward_label_spikes(&self, pattern_masks: &[u64], subslots: usize) -> Vec<OutputSpike> {
        let mut membrane = [0_i32; SNN_OUTPUTS];
        let mut spikes = Vec::new();

        for (step, mask) in pattern_masks.iter().enumerate() {
            let sample_index = step / subslots.max(1);
            let subslot = step % subslots.max(1);
            let mut next_values = [0_i32; SNN_OUTPUTS];
            for label in 0..SNN_OUTPUTS {
                let mut next =
                    ((membrane[label] * self.decay_alpha_q8) >> 8) + self.label_bias[label] as i32;
                for pattern in 0..PATTERN_NEURONS {
                    if ((mask >> pattern) & 1) != 0 {
                        next += self.label_weights[label][pattern] as i32;
                    }
                }
                next_values[label] = next;
            }
            apply_label_inhibition(&mut next_values, self.threshold);

            for label in 0..SNN_OUTPUTS {
                let next = next_values[label];
                if next >= self.threshold {
                    spikes.push(OutputSpike {
                        sample_index,
                        subslot,
                        label,
                    });
                    membrane[label] = 0;
                } else {
                    membrane[label] = next.clamp(MIN_MEMBRANE, self.threshold - 1);
                }
            }
        }

        spikes
    }
}

fn pattern_spikes_from_masks(pattern_masks: &[u64], subslots: usize) -> Vec<PatternSpike> {
    let mut spikes = Vec::new();
    for (step, mask) in pattern_masks.iter().enumerate() {
        let sample_index = step / subslots.max(1);
        let subslot = step % subslots.max(1);
        for pattern in 0..PATTERN_NEURONS {
            if ((mask >> pattern) & 1) != 0 {
                spikes.push(PatternSpike {
                    sample_index,
                    subslot,
                    pattern,
                });
            }
        }
    }
    spikes
}

fn gated_decisions(
    output_spikes: &[OutputSpike],
    pattern_spikes: Option<&[PatternSpike]>,
    input_spikes: Option<&[SpikeEvent]>,
    bins: usize,
    subslots: usize,
    config: &Config,
) -> Vec<GatedDecision> {
    let steps = bins * subslots.max(1);
    let mut label_counts_by_step = vec![[0_usize; SNN_OUTPUTS]; steps];
    let mut activity_by_step = vec![0_usize; steps];

    for spike in output_spikes {
        let step = spike.sample_index * subslots.max(1) + spike.subslot.min(subslots.max(1) - 1);
        if let Some(counts) = label_counts_by_step.get_mut(step) {
            counts[spike.label] += 1;
        }
    }

    if let Some(pattern_spikes) = pattern_spikes {
        for spike in pattern_spikes {
            let step =
                spike.sample_index * subslots.max(1) + spike.subslot.min(subslots.max(1) - 1);
            if let Some(activity) = activity_by_step.get_mut(step) {
                *activity += 1;
            }
        }
    } else if let Some(input_spikes) = input_spikes {
        for spike in input_spikes {
            let step =
                spike.sample_index * subslots.max(1) + spike.subslot.min(subslots.max(1) - 1);
            if let Some(activity) = activity_by_step.get_mut(step) {
                *activity += 1;
            }
        }
    }

    let mut decisions = Vec::new();
    let activity_window_steps = config.gate_window_samples.max(1) * subslots.max(1);

    for step in 0..steps {
        if step % subslots.max(1) != subslots.max(1) - 1 {
            continue;
        }

        let mut window_labels = [0_usize; SNN_OUTPUTS];
        for label in 0..SNN_OUTPUTS {
            let label_window_steps = label_gate_window_samples(label, config) * subslots.max(1);
            let label_window_start = (step + 1).saturating_sub(label_window_steps);
            for window_step in label_window_start..=step {
                window_labels[label] += label_counts_by_step[window_step][label];
            }
        }

        let mut window_activity = 0_usize;
        let activity_window_start = (step + 1).saturating_sub(activity_window_steps);
        for window_step in activity_window_start..=step {
            window_activity += activity_by_step[window_step];
        }

        let mut ranked = window_labels
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let top_score = ranked.first().map(|(_, score)| *score).unwrap_or(0);
        let fourth_score = ranked.get(3).map(|(_, score)| *score).unwrap_or(0);
        let margin = top_score as isize - fourth_score as isize;
        if top_score == 0
            || margin < config.gate_margin
            || window_activity < config.gate_min_activity
        {
            continue;
        }

        for (rank, (label, score)) in ranked.into_iter().take(3).enumerate() {
            if score < label_gate_min_top(label, config) {
                continue;
            }
            decisions.push(GatedDecision {
                sample_index: step / subslots.max(1),
                subslot: step % subslots.max(1),
                label,
                rank,
                score,
            });
        }
    }

    decisions
}

fn label_gate_min_top(label: usize, config: &Config) -> usize {
    if is_base_gate_label(label) {
        BASE_GATE_MIN_TOP.min(config.gate_min_top)
    } else {
        config.gate_min_top
    }
}

fn label_gate_window_samples(label: usize, config: &Config) -> usize {
    if is_base_gate_label(label) {
        config.gate_window_samples.max(BASE_GATE_WINDOW_SAMPLES)
    } else {
        config.gate_window_samples
    }
}

fn is_base_gate_label(label: usize) -> bool {
    matches!(
        label,
        LABEL_FLORAL_AMBER | LABEL_AMBER | LABEL_WOODY_AMBER | LABEL_DRY_WOODS
    )
}

fn apply_label_inhibition(values: &mut [i32; SNN_OUTPUTS], threshold: i32) {
    let mut inhibition = 0;
    if values[LABEL_GREEN] >= threshold {
        inhibition += TOP_NOTE_INHIBITION;
    }
    if values[LABEL_WATER] >= threshold {
        inhibition += TOP_NOTE_INHIBITION;
    }
    values[LABEL_FLORAL] -= inhibition;
}

fn render_svg(sample: &Sample, view: &SpikeView, bins: usize, subslots: usize) -> String {
    let panel_width = 980.0;
    let left = 170.0;
    let top = 78.0;
    let row_height = 18.0;
    let pattern_row_height = 10.0;
    let panel_gap = 64.0;
    let input_panel_height = row_height * CHANNELS as f32 + 28.0;
    let output_panel_height = row_height * SNN_OUTPUTS as f32 + 28.0;
    let pattern_panel_height = pattern_row_height * PATTERN_NEURONS as f32 + 28.0;
    let has_pattern = view.pattern.is_some();
    let has_gated = view.gated.is_some();
    let width = left + panel_width + 48.0;
    let mut height = top + input_panel_height * 3.0 + panel_gap * 2.0 + 30.0;
    if has_pattern {
        height += pattern_panel_height + panel_gap;
    }
    if view.output.is_some() {
        height += output_panel_height + panel_gap;
    }
    if has_gated {
        height += output_panel_height + panel_gap;
    }
    let x_step = panel_width / bins.max(1) as f32;
    let mut svg = String::new();

    svg.push_str(&format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0}" height="{height:.0}" viewBox="0 0 {width:.0} {height:.0}">
<rect width="100%" height="100%" fill="#f6f7f7"/>
<text x="28" y="30" font-family="system-ui, -apple-system, sans-serif" font-size="20" font-weight="700" fill="#202729">NoseKnows spike-train encoding preview</text>
<text x="28" y="54" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#526064">{id} / {name} / labels: {l1}, {l2}, {l3}</text>
<text x="748" y="54" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#526064">subslots/sample: {subslots} · default budget: rate 5 + latency 5</text>
"##,
        id = escape_xml(&sample.id),
        name = escape_xml(&sample.name),
        l1 = escape_xml(&sample.labels[0]),
        l2 = escape_xml(&sample.labels[1]),
        l3 = escape_xml(&sample.labels[2])
    ));

    render_panel(
        &mut svg,
        "pure latency: positive dV/dt maps to quantized sub-sample latency slots",
        &view.latency,
        top,
        left,
        panel_width,
        row_height,
        x_step,
        subslots,
        "#d05b35",
    );
    render_panel(
        &mut svg,
        "pure rate: log-scaled amplitude emits up to the rate budget per sample",
        &view.rate,
        top + input_panel_height + panel_gap,
        left,
        panel_width,
        row_height,
        x_step,
        subslots,
        "#2b77b9",
    );
    render_panel(
        &mut svg,
        "mixed: preserves rate and latency events in separate subslots",
        &view.mixed,
        top + (input_panel_height + panel_gap) * 2.0,
        left,
        panel_width,
        row_height,
        x_step,
        subslots,
        "#5d7c3b",
    );
    let mut final_top = top + (input_panel_height + panel_gap) * 3.0;
    if let Some(pattern_spikes) = &view.pattern {
        let names = view.pattern_names.as_deref().unwrap_or(&[]);
        render_pattern_panel(
            &mut svg,
            "accordion differentiation layer: 64 emergent-pattern spike trains",
            pattern_spikes,
            names,
            final_top,
            left,
            panel_width,
            pattern_row_height,
            x_step,
            subslots,
        );
        final_top += pattern_panel_height + panel_gap;
    }
    if let Some(output_spikes) = &view.output {
        render_output_panel(
            &mut svg,
            "final SNN layer: 14 fragrance output spike trains",
            output_spikes,
            final_top,
            left,
            panel_width,
            row_height,
            x_step,
            subslots,
        );
        final_top += output_panel_height + panel_gap;
    }
    if let Some(gated) = &view.gated {
        render_gated_panel(
            &mut svg,
            "rolling gated readout: recent evidence clears threshold",
            gated,
            final_top,
            left,
            panel_width,
            row_height,
            x_step,
            subslots,
        );
    }

    svg.push_str("</svg>\n");
    svg
}

#[allow(clippy::too_many_arguments)]
fn render_pattern_panel(
    svg: &mut String,
    title: &str,
    spikes: &[PatternSpike],
    names: &[String],
    top: f32,
    left: f32,
    width: f32,
    row_height: f32,
    x_step: f32,
    subslots: usize,
) {
    svg.push_str(&format!(
        r##"<text x="28" y="{title_y:.1}" font-family="system-ui, -apple-system, sans-serif" font-size="15" font-weight="650" fill="#202729">{}</text>
<line x1="{left:.1}" y1="{axis_y:.1}" x2="{right:.1}" y2="{axis_y:.1}" stroke="#cbd2d4" stroke-width="1"/>
"##,
        escape_xml(title),
        title_y = top,
        axis_y = top + 15.0,
        right = left + width
    ));

    for pattern in 0..PATTERN_NEURONS {
        let y = top + 30.0 + pattern as f32 * row_height;
        let name = names
            .get(pattern)
            .cloned()
            .unwrap_or_else(|| format!("pattern_{pattern:02}"));
        svg.push_str(&format!(
            r##"<text x="28" y="{label_y:.1}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="8" fill="#526064">p{pattern:02} {}</text>
<line x1="{left:.1}" y1="{y:.1}" x2="{right:.1}" y2="{y:.1}" stroke="#e0e4e5" stroke-width="1"/>
"##,
            escape_xml(&name),
            label_y = y + 3.0,
            right = left + width
        ));
        for spike in spikes.iter().filter(|spike| spike.pattern == pattern) {
            let x = left
                + spike.sample_index as f32 * x_step
                + ((spike.subslot as f32 + 0.5) / subslots.max(1) as f32) * x_step;
            svg.push_str(&format!(
                r##"<line x1="{x:.1}" y1="{y1:.1}" x2="{x:.1}" y2="{y2:.1}" stroke="#6b5bb8" stroke-width="1.05" stroke-linecap="round"/>
"##,
                y1 = y - 3.2,
                y2 = y + 3.2
            ));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_output_panel(
    svg: &mut String,
    title: &str,
    spikes: &[OutputSpike],
    top: f32,
    left: f32,
    width: f32,
    row_height: f32,
    x_step: f32,
    subslots: usize,
) {
    svg.push_str(&format!(
        r##"<text x="28" y="{title_y:.1}" font-family="system-ui, -apple-system, sans-serif" font-size="15" font-weight="650" fill="#202729">{}</text>
<line x1="{left:.1}" y1="{axis_y:.1}" x2="{right:.1}" y2="{axis_y:.1}" stroke="#cbd2d4" stroke-width="1"/>
"##,
        escape_xml(title),
        title_y = top,
        axis_y = top + 15.0,
        right = left + width
    ));

    for label in 0..SNN_OUTPUTS {
        let y = top + 30.0 + label as f32 * row_height;
        svg.push_str(&format!(
            r##"<text x="28" y="{label_y:.1}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" fill="#526064">{}</text>
<line x1="{left:.1}" y1="{y:.1}" x2="{right:.1}" y2="{y:.1}" stroke="#e0e4e5" stroke-width="1"/>
"##,
            escape_xml(LABELS[label]),
            label_y = y + 4.0,
            right = left + width
        ));
        for spike in spikes.iter().filter(|spike| spike.label == label) {
            let x = left
                + spike.sample_index as f32 * x_step
                + ((spike.subslot as f32 + 0.5) / subslots.max(1) as f32) * x_step;
            svg.push_str(&format!(
                r##"<circle cx="{x:.1}" cy="{y:.1}" r="2.4" fill="#9b3fa7"/>
"##
            ));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_gated_panel(
    svg: &mut String,
    title: &str,
    decisions: &[GatedDecision],
    top: f32,
    left: f32,
    width: f32,
    row_height: f32,
    x_step: f32,
    subslots: usize,
) {
    svg.push_str(&format!(
        r##"<text x="28" y="{title_y:.1}" font-family="system-ui, -apple-system, sans-serif" font-size="15" font-weight="650" fill="#202729">{}</text>
<line x1="{left:.1}" y1="{axis_y:.1}" x2="{right:.1}" y2="{axis_y:.1}" stroke="#cbd2d4" stroke-width="1"/>
"##,
        escape_xml(title),
        title_y = top,
        axis_y = top + 15.0,
        right = left + width
    ));

    for label in 0..SNN_OUTPUTS {
        let y = top + 30.0 + label as f32 * row_height;
        svg.push_str(&format!(
            r##"<text x="28" y="{label_y:.1}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" fill="#526064">{}</text>
<line x1="{left:.1}" y1="{y:.1}" x2="{right:.1}" y2="{y:.1}" stroke="#e0e4e5" stroke-width="1"/>
"##,
            escape_xml(LABELS[label]),
            label_y = y + 4.0,
            right = left + width
        ));
        for decision in decisions.iter().filter(|decision| decision.label == label) {
            let x = left
                + decision.sample_index as f32 * x_step
                + ((decision.subslot as f32 + 0.5) / subslots.max(1) as f32) * x_step;
            let (color, radius) = match decision.rank {
                0 => ("#008b8b", 3.0),
                1 => ("#2c9f7a", 2.5),
                _ => ("#68a357", 2.1),
            };
            svg.push_str(&format!(
                r##"<path d="M {x:.1} {top_y:.1} L {right_x:.1} {y:.1} L {x:.1} {bottom_y:.1} L {left_x:.1} {y:.1} Z" fill="{color}" opacity="0.84">
<title>rank {} score {}</title>
</path>
"##,
                decision.rank + 1,
                decision.score,
                top_y = y - radius,
                right_x = x + radius,
                bottom_y = y + radius,
                left_x = x - radius,
            ));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_panel(
    svg: &mut String,
    title: &str,
    spikes: &[SpikeEvent],
    top: f32,
    left: f32,
    width: f32,
    row_height: f32,
    x_step: f32,
    subslots: usize,
    color: &str,
) {
    svg.push_str(&format!(
        r##"<text x="28" y="{title_y:.1}" font-family="system-ui, -apple-system, sans-serif" font-size="15" font-weight="650" fill="#202729">{}</text>
<line x1="{left:.1}" y1="{axis_y:.1}" x2="{right:.1}" y2="{axis_y:.1}" stroke="#cbd2d4" stroke-width="1"/>
"##,
        escape_xml(title),
        title_y = top,
        axis_y = top + 15.0,
        right = left + width
    ));

    for channel in 0..CHANNELS {
        let y = top + 30.0 + channel as f32 * row_height;
        svg.push_str(&format!(
            r##"<text x="28" y="{label_y:.1}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" fill="#526064">{}</text>
<line x1="{left:.1}" y1="{y:.1}" x2="{right:.1}" y2="{y:.1}" stroke="#e0e4e5" stroke-width="1"/>
"##,
            escape_xml(CHANNEL_NAMES[channel]),
            label_y = y + 4.0,
            right = left + width
        ));
        for event in spikes.iter().filter(|event| event.channel == channel) {
            let x = left
                + event.sample_index as f32 * x_step
                + ((event.subslot as f32 + 0.5) / subslots.max(1) as f32) * x_step;
            let y_offset = event.slot as f32 * 1.35;
            let y1 = y - 6.0 + y_offset;
            let y2 = y + 6.0 + y_offset;
            let event_color = match event.stream {
                SpikeStream::Rate => color,
                SpikeStream::Latency => "#d05b35",
            };
            let stroke_width = match event.stream {
                SpikeStream::Rate => 1.15,
                SpikeStream::Latency => 1.7,
            };
            svg.push_str(&format!(
                r#"<line x1="{x:.1}" y1="{y1:.1}" x2="{x:.1}" y2="{y2:.1}" stroke="{event_color}" stroke-width="{stroke_width:.2}" stroke-linecap="round"/>
"#,
            ));
        }
    }
}

fn print_spike_summary(name: &str, spikes: &[SpikeEvent]) {
    let total = spikes.len();
    let per_channel = (0..CHANNELS)
        .map(|channel| {
            spikes
                .iter()
                .filter(|event| event.channel == channel)
                .count()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join(",");
    println!("{name:>7} spikes total={total} per_channel=[{per_channel}]");
}

fn print_pattern_summary(name: &str, spikes: &[PatternSpike]) {
    let total = spikes.len();
    let active = (0..PATTERN_NEURONS)
        .filter(|pattern| spikes.iter().any(|event| event.pattern == *pattern))
        .count();
    let top = top_pattern_counts(spikes, 8)
        .into_iter()
        .map(|(pattern, count)| format!("p{pattern:02}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    println!("{name:>7} spikes total={total} active_patterns={active} top=[{top}]");
}

fn print_output_summary(name: &str, spikes: &[OutputSpike]) {
    let total = spikes.len();
    let per_label = (0..SNN_OUTPUTS)
        .map(|label| {
            spikes
                .iter()
                .filter(|event| event.label == label)
                .count()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join(",");
    println!("{name:>7} spikes total={total} per_label=[{per_label}]");
}

fn print_gated_summary(name: &str, decisions: &[GatedDecision]) {
    let total = decisions.len();
    let timepoints = decisions
        .iter()
        .map(|decision| (decision.sample_index, decision.subslot))
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let per_label = (0..SNN_OUTPUTS)
        .map(|label| {
            decisions
                .iter()
                .filter(|decision| decision.label == label)
                .count()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join(",");
    println!("{name:>7} decisions total={total} timepoints={timepoints} per_label=[{per_label}]");
}

fn print_accordion_contribution_summary(
    model: &AccordionLifModel,
    pattern_spikes: &[PatternSpike],
    gated: &[GatedDecision],
    stored_labels: &[String; 3],
) {
    let mut pattern_counts = [0_usize; PATTERN_NEURONS];
    for spike in pattern_spikes {
        pattern_counts[spike.pattern] += 1;
    }

    let mut gated_counts = [0_usize; SNN_OUTPUTS];
    for decision in gated {
        gated_counts[decision.label] += 1;
    }

    let mut labels_to_explain = Vec::new();
    for label in stored_labels {
        if let Some(index) = label_index(label) {
            labels_to_explain.push(index);
        }
    }
    let mut gated_ranked = gated_counts.iter().copied().enumerate().collect::<Vec<_>>();
    gated_ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (label, count) in gated_ranked.into_iter().take(5) {
        if count > 0 {
            labels_to_explain.push(label);
        }
    }
    labels_to_explain.sort_unstable();
    labels_to_explain.dedup();

    println!();
    println!("Accordion contribution diagnostics:");
    for label in labels_to_explain {
        let mut contributions = (0..PATTERN_NEURONS)
            .filter(|pattern| pattern_counts[*pattern] > 0)
            .map(|pattern| {
                let count = pattern_counts[pattern] as i32;
                let weight = model.label_weights[label][pattern] as i32;
                let contribution = count * weight;
                (pattern, count, weight, contribution)
            })
            .collect::<Vec<_>>();
        let total: i32 = contributions
            .iter()
            .map(|(_, _, _, contribution)| *contribution)
            .sum();
        contributions.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| a.0.cmp(&b.0)));

        println!(
            "  {} gated_count={} weighted_sum={}",
            LABELS[label], gated_counts[label], total
        );
        for (pattern, count, weight, contribution) in contributions.iter().take(6) {
            let name = model
                .pattern_names
                .get(*pattern)
                .map(String::as_str)
                .unwrap_or("unnamed pattern");
            println!(
                "    + p{pattern:02} count={count:>3} weight={weight:>5} contrib={contribution:>7} {name}"
            );
        }

        let mut negative = contributions
            .iter()
            .filter(|(_, _, _, contribution)| *contribution < 0)
            .copied()
            .collect::<Vec<_>>();
        negative.sort_by(|a, b| a.3.cmp(&b.3).then_with(|| a.0.cmp(&b.0)));
        for (pattern, count, weight, contribution) in negative.iter().take(3) {
            let name = model
                .pattern_names
                .get(*pattern)
                .map(String::as_str)
                .unwrap_or("unnamed pattern");
            println!(
                "    - p{pattern:02} count={count:>3} weight={weight:>5} contrib={contribution:>7} {name}"
            );
        }
    }
    println!();
}

fn top_pattern_counts(spikes: &[PatternSpike], count: usize) -> Vec<(usize, usize)> {
    let mut counts = (0..PATTERN_NEURONS)
        .map(|pattern| {
            (
                pattern,
                spikes
                    .iter()
                    .filter(|event| event.pattern == pattern)
                    .count(),
            )
        })
        .collect::<Vec<_>>();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    counts.truncate(count);
    counts
}

fn label_index(label: &str) -> Option<usize> {
    LABELS
        .iter()
        .position(|candidate| candidate.eq_ignore_ascii_case(label))
}

fn pattern_uses_sensor(weights: &[i16; SNN_INPUTS], sensor: usize) -> bool {
    weights[sensor] > 0 || weights[ACTIVE_SENSORS + sensor] > 0
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

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_encoding_increases_spike_count_with_amplitude() {
        let low = rate_spike_count(0.10, DEFAULT_RATE_BUDGET);
        let high = rate_spike_count(0.90, DEFAULT_RATE_BUDGET);

        assert!(high > low);
        assert!(high <= DEFAULT_RATE_BUDGET);
    }

    #[test]
    fn latency_encoding_responds_to_rising_edges() {
        let config = test_config();
        let mut rows = Vec::new();
        for index in 0..40 {
            let mut row = [200.0; CHANNELS];
            row[0] = if index < 10 { 200.0 } else { 3000.0 };
            rows.push(row);
        }

        let spikes = encode_spikes(&rows, Encoding::Latency, &config);

        assert!(count_channel(&spikes, 0) > 0);
        assert_eq!(count_channel(&spikes, 1), 0);
    }

    #[test]
    fn steeper_deltas_get_earlier_latency_subslots() {
        let shallow = latency_subslot(0.01, 5).expect("shallow spike");
        let steep = latency_subslot(0.08, 5).expect("steep spike");

        assert!(steep < shallow);
    }

    #[test]
    fn mixed_encoding_respects_per_sample_budget() {
        let config = test_config();
        let mut rows = Vec::new();
        for index in 0..40 {
            let mut row = [200.0; CHANNELS];
            row[0] = 200.0 + index as f32 * 120.0;
            rows.push(row);
        }

        let spikes = encode_spikes(&rows, Encoding::Mixed, &config);
        let max_for_channel_sample = spikes
            .iter()
            .filter(|event| event.channel == 0)
            .map(|event| {
                spikes
                    .iter()
                    .filter(|other| {
                        other.channel == event.channel && other.sample_index == event.sample_index
                    })
                    .count()
            })
            .max()
            .unwrap_or(0);

        assert!(max_for_channel_sample <= config.rate_budget + config.latency_budget);
    }

    #[test]
    fn parses_lif_model_weights() {
        let mut text = String::from(
            "NOSEKNOWS_SNN_LIF_V1\nthreshold=1000\ndecay_alpha_q8=235\nlabels=Floral,Soft Floral,Floral Amber,Amber,Soft Amber,Woody Amber,Woods,Mossy Woods,Dry Woods,Aromatic,Citrus,Water,Green,Fruity\n",
        );
        for label in LABELS {
            text.push_str("weights.");
            text.push_str(label);
            text.push('=');
            text.push_str(&vec!["7"; SNN_INPUTS].join(","));
            text.push('\n');
        }
        let path = std::env::temp_dir().join("noseknows_spikes_model_test.nsm");
        fs::write(&path, text).expect("write model");

        let model = load_snn_model(&path).expect("load model");

        match model {
            SnnModel::Direct(model) => {
                assert_eq!(model.threshold, 1000);
                assert_eq!(model.decay_alpha_q8, 235);
                assert_eq!(model.weights[0][0], 7);
            }
            SnnModel::Accordion(_) => panic!("expected direct model"),
        }
    }

    #[test]
    fn parses_accordion_model_weights() {
        let mut text = String::from(
            "NOSEKNOWS_SNN_ACCORDION_V1\nthreshold=1000\ndecay_alpha_q8=235\nlabels=Floral,Soft Floral,Floral Amber,Amber,Soft Amber,Woody Amber,Woods,Mossy Woods,Dry Woods,Aromatic,Citrus,Water,Green,Fruity\n",
        );
        for pattern in 0..PATTERN_NEURONS {
            text.push_str(&format!(
                "pattern.{pattern:02}.name=test pattern {pattern}\n"
            ));
            text.push_str(&format!(
                "pattern.{pattern:02}.weights={}\n",
                vec!["3"; SNN_INPUTS].join(",")
            ));
        }
        for label in LABELS {
            text.push_str("label_weights.");
            text.push_str(label);
            text.push('=');
            text.push_str(&vec!["5"; PATTERN_NEURONS].join(","));
            text.push('\n');
        }
        let path = std::env::temp_dir().join("noseknows_spikes_accordion_model_test.nsm");
        fs::write(&path, text).expect("write model");

        let model = load_snn_model(&path).expect("load model");

        match model {
            SnnModel::Accordion(model) => {
                assert_eq!(model.threshold, 1000);
                assert_eq!(model.pattern_weights[0][0], 3);
                assert_eq!(model.label_weights[0][0], 5);
                assert_eq!(model.pattern_names[3], "test pattern 3");
            }
            SnnModel::Direct(_) => panic!("expected accordion model"),
        }
    }

    #[test]
    fn parses_quoted_csv_fields() {
        let fields = parse_csv_line(r#"a,"b,c","d""e""#);

        assert_eq!(fields, vec!["a", "b,c", "d\"e"]);
    }

    #[test]
    fn gated_readout_waits_for_enough_evidence() {
        let mut config = test_config();
        config.gate_min_top = 2;
        config.gate_margin = 1;
        config.gate_min_activity = 3;
        let output = vec![
            OutputSpike {
                sample_index: 0,
                subslot: 0,
                label: 0,
            },
            OutputSpike {
                sample_index: 1,
                subslot: 0,
                label: 0,
            },
        ];
        let pattern = vec![
            PatternSpike {
                sample_index: 0,
                subslot: 0,
                pattern: 0,
            },
            PatternSpike {
                sample_index: 0,
                subslot: 1,
                pattern: 1,
            },
            PatternSpike {
                sample_index: 1,
                subslot: 0,
                pattern: 2,
            },
        ];

        let decisions = gated_decisions(&output, Some(&pattern), None, 4, 2, &config);

        assert!(!decisions.is_empty());
        assert!(decisions.iter().all(|decision| decision.sample_index >= 1));
        assert_eq!(decisions[0].label, 0);
    }

    #[test]
    fn gated_readout_drops_old_evidence_outside_window() {
        let mut config = test_config();
        config.gate_min_top = 2;
        config.gate_margin = 1;
        config.gate_min_activity = 2;
        config.gate_window_samples = 1;
        let output = vec![
            OutputSpike {
                sample_index: 0,
                subslot: 0,
                label: 0,
            },
            OutputSpike {
                sample_index: 1,
                subslot: 0,
                label: 0,
            },
        ];
        let pattern = vec![
            PatternSpike {
                sample_index: 0,
                subslot: 0,
                pattern: 0,
            },
            PatternSpike {
                sample_index: 1,
                subslot: 0,
                pattern: 1,
            },
        ];

        let decisions = gated_decisions(&output, Some(&pattern), None, 3, 2, &config);

        assert!(decisions.is_empty());
    }

    fn test_config() -> Config {
        Config {
            input: PathBuf::from(DEFAULT_INPUT),
            output: PathBuf::from(DEFAULT_OUTPUT),
            model: PathBuf::from(DEFAULT_MODEL),
            bins: DEFAULT_BINS,
            subslots: DEFAULT_SUBSLOTS,
            rate_budget: DEFAULT_RATE_BUDGET,
            latency_budget: DEFAULT_LATENCY_BUDGET,
            gate_min_top: DEFAULT_GATE_MIN_TOP,
            gate_margin: DEFAULT_GATE_MARGIN,
            gate_min_activity: DEFAULT_GATE_MIN_ACTIVITY,
            gate_window_samples: DEFAULT_GATE_WINDOW_SAMPLES,
        }
    }

    fn count_channel(spikes: &[SpikeEvent], channel: usize) -> usize {
        spikes
            .iter()
            .filter(|event| event.channel == channel)
            .count()
    }
}
