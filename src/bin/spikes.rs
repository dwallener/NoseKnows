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
const DEFAULT_MODEL: &str = "data/models/snn_lif_smoke.nsm";
const DEFAULT_BINS: usize = 180;
const DEFAULT_SUBSLOTS: usize = 5;
const DEFAULT_RATE_BUDGET: usize = 5;
const DEFAULT_LATENCY_BUDGET: usize = 5;
const ACTIVE_SENSORS: usize = 8;
const SNN_INPUTS: usize = ACTIVE_SENSORS * 2;
const SNN_OUTPUTS: usize = 14;
const THRESHOLD: i32 = 1000;
const DECAY_ALPHA_Q8: i32 = 235;
const MIN_MEMBRANE: i32 = -3000;
const MAX_ADC: f32 = 4095.0;
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
    output: Option<Vec<OutputSpike>>,
}

struct OutputSpike {
    sample_index: usize,
    subslot: usize,
    label: usize,
}

struct LifModel {
    weights: [[i16; SNN_INPUTS]; SNN_OUTPUTS],
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
        output: None,
    };
    let model = load_lif_model(&config.model)?;
    let input_masks = input_masks_from_events(&view.mixed, bins, subslots);
    let output_spikes = model.forward_spikes(&input_masks, subslots);
    let view = SpikeView {
        output: Some(output_spikes),
        ..view
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
        CHANNELS * (config.rate_budget + config.latency_budget)
    );
    print_spike_summary("rate", &view.rate);
    print_spike_summary("latency", &view.latency);
    print_spike_summary("mixed", &view.mixed);
    if let Some(output) = &view.output {
        print_output_summary("output", output);
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
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin spikes -- [--input data/raw/synthetic_0000.csv] [--out data/spikes.svg] [--model data/models/snn_lif_smoke.nsm] [--bins 180] [--subslots 5] [--rate-budget 5] [--latency-budget 5]"
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
            let amplitude = ((row[channel] - baseline) / (peak - baseline)).clamp(0.0, 1.0);
            let delta = if index == 0 {
                0.0
            } else {
                (row[channel] - previous).max(0.0) / MAX_ADC
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

fn load_lif_model(path: &PathBuf) -> Result<LifModel, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut weights = [[0_i16; SNN_INPUTS]; SNN_OUTPUTS];
    let mut threshold = THRESHOLD;
    let mut decay_alpha_q8 = DECAY_ALPHA_Q8;

    for line in text.lines() {
        if let Some(value) = line.strip_prefix("threshold=") {
            threshold = value.parse()?;
        } else if let Some(value) = line.strip_prefix("decay_alpha_q8=") {
            decay_alpha_q8 = value.parse()?;
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

    Ok(LifModel {
        weights,
        threshold,
        decay_alpha_q8,
    })
}

impl LifModel {
    fn forward_spikes(&self, masks: &[u16], subslots: usize) -> Vec<OutputSpike> {
        let mut membrane = [0_i32; SNN_OUTPUTS];
        let mut spikes = Vec::new();

        for (step, mask) in masks.iter().enumerate() {
            let sample_index = step / subslots.max(1);
            let subslot = step % subslots.max(1);
            for label in 0..SNN_OUTPUTS {
                let mut next = (membrane[label] * self.decay_alpha_q8) >> 8;
                for input in 0..SNN_INPUTS {
                    if ((mask >> input) & 1) != 0 {
                        next += self.weights[label][input] as i32;
                    }
                }

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

fn render_svg(sample: &Sample, view: &SpikeView, bins: usize, subslots: usize) -> String {
    let panel_width = 980.0;
    let left = 170.0;
    let top = 78.0;
    let row_height = 18.0;
    let panel_gap = 64.0;
    let input_panel_height = row_height * CHANNELS as f32 + 28.0;
    let output_panel_height = row_height * SNN_OUTPUTS as f32 + 28.0;
    let panel_count = if view.output.is_some() { 4.0 } else { 3.0 };
    let width = left + panel_width + 48.0;
    let height = if view.output.is_some() {
        top + input_panel_height * 3.0 + output_panel_height + panel_gap * 3.0 + 30.0
    } else {
        top + input_panel_height * panel_count + panel_gap * 2.0 + 30.0
    };
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
    if let Some(output_spikes) = &view.output {
        render_output_panel(
            &mut svg,
            "final SNN layer: 14 fragrance output spike trains",
            output_spikes,
            top + (input_panel_height + panel_gap) * 3.0,
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

        let model = load_lif_model(&path).expect("load model");

        assert_eq!(model.threshold, 1000);
        assert_eq!(model.decay_alpha_q8, 235);
        assert_eq!(model.weights[0][0], 7);
    }

    #[test]
    fn parses_quoted_csv_fields() {
        let fields = parse_csv_line(r#"a,"b,c","d""e""#);

        assert_eq!(fields, vec!["a", "b,c", "d\"e"]);
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
        }
    }

    fn count_channel(spikes: &[SpikeEvent], channel: usize) -> usize {
        spikes
            .iter()
            .filter(|event| event.channel == channel)
            .count()
    }
}
