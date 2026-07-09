use std::cmp::Ordering;
use std::env;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};

const CHANNELS: usize = 9;
const ACTIVE_SENSORS: usize = 8;
const FEATURES: usize = ACTIVE_SENSORS * 2;
const OUTPUTS: usize = 14;
const MAX_ADC: f32 = 4095.0;
const CLEAN_AIR_FLOOR_ADC: f32 = 300.0;
const MIN_DELTA_ADC: f32 = 25.0;
const DEFAULT_STREAM: &str = "data/streams/smoke_stream.csv";
const DEFAULT_MODEL: &str = "data/models/snn_stream_smoke.nsm";
const DEFAULT_OUT: &str = "data/streams/stream_preview.svg";
const DEFAULT_ROWS: usize = 3000;

const CHANNEL_NAMES: [&str; CHANNELS] = [
    "adc0 MQ-2",
    "adc1 MQ-3",
    "adc2 MQ-5",
    "adc3 MQ-6",
    "adc4 MQ-7",
    "adc5 MQ-8",
    "adc6 MQ-9",
    "adc7 MQ-135",
    "adc8 MQ-4",
];

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

const LABEL_COLORS: [&str; OUTPUTS] = [
    "#b44bb6", "#c46bbb", "#d36b93", "#b96a34", "#9b7a44", "#8c6a3f", "#627b3e",
    "#4f7c5a", "#6a6254", "#3a8d76", "#d99a2b", "#3d93b8", "#4d9b4d", "#c85b5b",
];

struct Config {
    stream_path: PathBuf,
    model_path: PathBuf,
    output_path: PathBuf,
    start_row: usize,
    rows: usize,
    columns: usize,
    gate_threshold: f32,
}

#[derive(Clone)]
struct StreamRow {
    target: [bool; OUTPUTS],
    adc: [f32; CHANNELS],
}

struct StreamModel {
    weights: [[f32; FEATURES]; OUTPUTS],
    bias: [f32; OUTPUTS],
    window: usize,
    rate_budget: usize,
    latency_budget: usize,
}

struct Frame {
    features: [f32; FEATURES],
    logits: [f32; OUTPUTS],
}

fn main() {
    if let Err(error) = run() {
        eprintln!("stream_viz error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let all_rows = load_stream(&config.stream_path)?;
    if all_rows.len() < 2 {
        return Err("stream visualization needs at least two rows".into());
    }
    let model = load_model(&config.model_path)?;
    let start = config.start_row.min(all_rows.len() - 1);
    let end = (start + config.rows).min(all_rows.len());
    let rows = &all_rows[start..end];
    if rows.len() < 2 {
        return Err("selected stream window needs at least two rows".into());
    }
    let frames = build_frames(rows, &model);
    let svg = render_svg(rows, &frames, &model, &config, start);
    if let Some(parent) = config.output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.output_path, svg)?;
    println!(
        "Wrote stream visualization to {} rows={}..{}",
        config.output_path.display(),
        start,
        end
    );
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut stream_path = PathBuf::from(DEFAULT_STREAM);
    let mut model_path = PathBuf::from(DEFAULT_MODEL);
    let mut output_path = PathBuf::from(DEFAULT_OUT);
    let mut start_row = 0;
    let mut rows = DEFAULT_ROWS;
    let mut columns = 900;
    let mut gate_threshold = 0.0;

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
            "--out" => {
                index += 1;
                output_path = PathBuf::from(args.get(index).ok_or("--out requires a path")?);
            }
            "--start-row" => {
                index += 1;
                start_row = args
                    .get(index)
                    .ok_or("--start-row requires a value")?
                    .parse()?;
            }
            "--rows" => {
                index += 1;
                rows = args.get(index).ok_or("--rows requires a value")?.parse()?;
            }
            "--columns" => {
                index += 1;
                columns = args
                    .get(index)
                    .ok_or("--columns requires a value")?
                    .parse()?;
            }
            "--gate-threshold" => {
                index += 1;
                gate_threshold = args
                    .get(index)
                    .ok_or("--gate-threshold requires a value")?
                    .parse()?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin stream_viz -- [--stream data/streams/smoke_stream.csv] [--model data/models/snn_stream_smoke.nsm] [--out data/streams/stream_preview.svg] [--start-row 0] [--rows 3000] [--columns 900]"
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
        output_path,
        start_row,
        rows: rows.max(2),
        columns: columns.clamp(100, 3000),
        gate_threshold,
    })
}

fn build_frames(rows: &[StreamRow], model: &StreamModel) -> Vec<Frame> {
    let mut rolling = [0.0_f32; FEATURES];
    let mut history = Vec::new();
    let mut frames = Vec::with_capacity(rows.len());
    let mut previous_adc = rows[0].adc;

    for row in rows {
        let mut instant = [0.0_f32; FEATURES];
        for sensor in 0..ACTIVE_SENSORS {
            let amplitude =
                ((row.adc[sensor] - CLEAN_AIR_FLOOR_ADC) / (MAX_ADC - CLEAN_AIR_FLOOR_ADC))
                    .clamp(0.0, 1.0);
            instant[sensor] = rate_feature(amplitude, model.rate_budget);

            let delta = row.adc[sensor] - previous_adc[sensor];
            instant[ACTIVE_SENSORS + sensor] = latency_feature(delta, model.latency_budget);
        }
        previous_adc = row.adc;

        for feature in 0..FEATURES {
            rolling[feature] += instant[feature];
        }
        history.push(instant);
        if history.len() > model.window {
            let expired = history.remove(0);
            for feature in 0..FEATURES {
                rolling[feature] -= expired[feature];
            }
        }

        let divisor = history.len().max(1) as f32;
        let mut features = rolling;
        for feature in &mut features {
            *feature /= divisor;
        }
        frames.push(Frame {
            features,
            logits: model.predict(&features),
        });
    }

    frames
}

fn render_svg(
    rows: &[StreamRow],
    frames: &[Frame],
    model: &StreamModel,
    config: &Config,
    start_row: usize,
) -> String {
    let left = 150.0_f32;
    let right = 24.0_f32;
    let top = 42.0_f32;
    let width = left + config.columns as f32 + right;
    let row_gap = 18.0_f32;
    let section_gap = 34.0_f32;
    let adc_height = CHANNELS as f32 * row_gap;
    let feature_height = FEATURES as f32 * 11.0;
    let label_height = OUTPUTS as f32 * row_gap;
    let total_height = top
        + 22.0
        + adc_height
        + section_gap
        + feature_height
        + section_gap
        + label_height
        + section_gap
        + label_height
        + 32.0;

    let mut svg = String::new();
    let _ = writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0}" height="{total_height:.0}" viewBox="0 0 {width:.0} {total_height:.0}">"#
    );
    let _ = writeln!(
        svg,
        r##"<rect width="100%" height="100%" fill="#f6f8f8"/><style>text{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:#273033}} .small{{font-size:11px}} .title{{font:700 18px system-ui,-apple-system,sans-serif}}</style>"##
    );
    let _ = writeln!(
        svg,
        r#"<text x="18" y="26" class="title">stream readout timeline: rows {}..{} window={}</text>"#,
        start_row,
        start_row + rows.len(),
        model.window
    );

    let mut y = top;
    render_truth_strip(&mut svg, rows, left, y, config.columns as f32);
    y += 22.0;
    render_adc(&mut svg, rows, left, y, config.columns as f32);
    y += adc_height + section_gap;
    render_features(&mut svg, frames, left, y, config.columns as f32);
    y += feature_height + section_gap;
    render_label_heatmap(&mut svg, frames, left, y, config.columns as f32);
    y += label_height + section_gap;
    render_gated(&mut svg, frames, left, y, config.columns as f32, config.gate_threshold);
    let _ = writeln!(svg, "</svg>");
    svg
}

fn render_truth_strip(svg: &mut String, rows: &[StreamRow], left: f32, y: f32, panel_width: f32) {
    let _ = writeln!(
        svg,
        r#"<text x="18" y="{:.1}" class="small">truth</text>"#,
        y + 10.0
    );
    let columns = panel_width as usize;
    let rows_per_col = rows.len() as f32 / columns as f32;
    for col in 0..columns {
        let start = (col as f32 * rows_per_col).floor() as usize;
        let end = (((col + 1) as f32 * rows_per_col).ceil() as usize).min(rows.len());
        let mut counts = [0_usize; OUTPUTS];
        for row in &rows[start..end] {
            for (label, is_active) in row.target.iter().enumerate() {
                if *is_active {
                    counts[label] += 1;
                }
            }
        }
        let active = counts.iter().any(|count| *count > 0);
        if !active {
            rect(svg, left + col as f32, y, 1.0, 12.0, "#d9dddd", 1.0);
        } else {
            let mut ranked = counts.iter().copied().enumerate().collect::<Vec<_>>();
            ranked.sort_by(|a, b| b.1.cmp(&a.1));
            for (slot, (label, count)) in ranked.into_iter().take(3).enumerate() {
                if count > 0 {
                    rect(
                        svg,
                        left + col as f32,
                        y + slot as f32 * 4.0,
                        1.0,
                        4.0,
                        LABEL_COLORS[label],
                        0.85,
                    );
                }
            }
        }
    }
}

fn render_adc(svg: &mut String, rows: &[StreamRow], left: f32, y: f32, panel_width: f32) {
    let _ = writeln!(
        svg,
        r#"<text x="18" y="{:.1}" class="title">ADC traces</text>"#,
        y - 8.0
    );
    for channel in 0..CHANNELS {
        let row_y = y + channel as f32 * 18.0;
        label(svg, CHANNEL_NAMES[channel], row_y + 4.0);
        line(svg, left, row_y, left + panel_width, row_y, "#d8dede", 1.0);
        let mut points = String::new();
        let columns = panel_width as usize;
        let rows_per_col = rows.len() as f32 / columns as f32;
        for col in 0..columns {
            let start = (col as f32 * rows_per_col).floor() as usize;
            let end = (((col + 1) as f32 * rows_per_col).ceil() as usize).min(rows.len());
            let mut sum = 0.0;
            let mut count = 0.0;
            for row in &rows[start..end] {
                sum += row.adc[channel];
                count += 1.0;
            }
            let mean = if count > 0.0 { sum / count } else { 0.0 };
            let amp = (mean / MAX_ADC).clamp(0.0, 1.0);
            let px = left + col as f32;
            let py = row_y + 7.0 - amp * 14.0;
            let _ = write!(points, "{px:.1},{py:.1} ");
        }
        let _ = writeln!(
            svg,
            r##"<polyline points="{}" fill="none" stroke="#2878a8" stroke-width="1.2" opacity="0.85"/>"##,
            points.trim()
        );
    }
}

fn render_features(svg: &mut String, frames: &[Frame], left: f32, y: f32, panel_width: f32) {
    let _ = writeln!(
        svg,
        r#"<text x="18" y="{:.1}" class="title">rolling input features</text>"#,
        y - 8.0
    );
    for feature in 0..FEATURES {
        let row_y = y + feature as f32 * 11.0;
        let name = if feature < ACTIVE_SENSORS {
            format!("r{feature}")
        } else {
            format!("d{}", feature - ACTIVE_SENSORS)
        };
        label(svg, &name, row_y + 4.0);
        heat_row(svg, frames.len(), panel_width as usize, left, row_y, 7.0, |start, end| {
            let mut sum = 0.0;
            let mut count = 0.0;
            for frame in &frames[start..end] {
                sum += frame.features[feature];
                count += 1.0;
            }
            if count > 0.0 {
                (sum / count).clamp(0.0, 1.0)
            } else {
                0.0
            }
        });
    }
}

fn render_label_heatmap(svg: &mut String, frames: &[Frame], left: f32, y: f32, panel_width: f32) {
    let _ = writeln!(
        svg,
        r#"<text x="18" y="{:.1}" class="title">label evidence</text>"#,
        y - 8.0
    );
    let max_abs = frames
        .iter()
        .flat_map(|frame| frame.logits.iter())
        .fold(1.0_f32, |max_value, value| max_value.max(value.abs()));
    for label_index in 0..OUTPUTS {
        let row_y = y + label_index as f32 * 18.0;
        label(svg, LABELS[label_index], row_y + 4.0);
        line(svg, left, row_y, left + panel_width, row_y, "#d8dede", 1.0);
        heat_row(svg, frames.len(), panel_width as usize, left, row_y - 5.0, 10.0, |start, end| {
            let mut sum = 0.0;
            let mut count = 0.0;
            for frame in &frames[start..end] {
                sum += frame.logits[label_index].max(0.0) / max_abs;
                count += 1.0;
            }
            if count > 0.0 {
                (sum / count).clamp(0.0, 1.0)
            } else {
                0.0
            }
        });
    }
}

fn render_gated(
    svg: &mut String,
    frames: &[Frame],
    left: f32,
    y: f32,
    panel_width: f32,
    gate_threshold: f32,
) {
    let _ = writeln!(
        svg,
        r#"<text x="18" y="{:.1}" class="title">gated readout</text>"#,
        y - 8.0
    );
    let columns = panel_width as usize;
    let frames_per_col = frames.len() as f32 / columns as f32;
    for label_index in 0..OUTPUTS {
        let row_y = y + label_index as f32 * 18.0;
        label(svg, LABELS[label_index], row_y + 4.0);
        line(svg, left, row_y, left + panel_width, row_y, "#d8dede", 1.0);
    }
    for col in 0..columns {
        let start = (col as f32 * frames_per_col).floor() as usize;
        let end = (((col + 1) as f32 * frames_per_col).ceil() as usize).min(frames.len());
        let mut scores = [0.0_f32; OUTPUTS];
        for frame in &frames[start..end] {
            for (label, score) in scores.iter_mut().enumerate() {
                *score += frame.logits[label];
            }
        }
        let divisor = (end.saturating_sub(start)).max(1) as f32;
        for score in &mut scores {
            *score /= divisor;
        }
        let mut top = scores.iter().copied().enumerate().collect::<Vec<_>>();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        for (rank, (label_index, score)) in top.into_iter().take(3).enumerate() {
            if score > gate_threshold {
                let opacity = [0.95, 0.55, 0.32][rank];
                rect(
                    svg,
                    left + col as f32,
                    y + label_index as f32 * 18.0 - 5.0,
                    1.0,
                    10.0,
                    LABEL_COLORS[label_index],
                    opacity,
                );
            }
        }
    }
}

fn heat_row<F>(
    svg: &mut String,
    len: usize,
    columns: usize,
    left: f32,
    y: f32,
    height: f32,
    mut value_at: F,
) where
    F: FnMut(usize, usize) -> f32,
{
    let rows_per_col = len as f32 / columns as f32;
    for col in 0..columns {
        let start = (col as f32 * rows_per_col).floor() as usize;
        let end = (((col + 1) as f32 * rows_per_col).ceil() as usize).min(len);
        let value = value_at(start, end).clamp(0.0, 1.0);
        if value > 0.001 {
            let opacity = 0.12 + value * 0.82;
            rect(svg, left + col as f32, y, 1.0, height, "#219c98", opacity);
        }
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
            target,
            adc,
        });
    }
    Ok(rows)
}

fn load_model(path: &Path) -> Result<StreamModel, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    if lines.next() != Some("NOSEKNOWS_SNN_STREAM_READOUT_V1") {
        return Err(format!("{} is not a stream readout model", path.display()).into());
    }

    let mut model = StreamModel {
        weights: [[0.0; FEATURES]; OUTPUTS],
        bias: [0.0; OUTPUTS],
        window: 30,
        rate_budget: 5,
        latency_budget: 5,
    };

    for line in lines {
        if let Some(value) = line.strip_prefix("window=") {
            model.window = value.parse()?;
        } else if let Some(value) = line.strip_prefix("rate_budget=") {
            model.rate_budget = value.parse()?;
        } else if let Some(value) = line.strip_prefix("latency_budget=") {
            model.latency_budget = value.parse()?;
        } else if let Some(rest) = line.strip_prefix("bias.") {
            let (label, value) = rest
                .split_once('=')
                .ok_or_else(|| format!("malformed bias line: {line}"))?;
            if let Some(index) = label_index(label) {
                model.bias[index] = value.parse()?;
            }
        } else if let Some(rest) = line.strip_prefix("weights.") {
            let (label, values) = rest
                .split_once('=')
                .ok_or_else(|| format!("malformed weights line: {line}"))?;
            if let Some(index) = label_index(label) {
                let parsed = values
                    .split(',')
                    .map(|value| value.parse::<f32>())
                    .collect::<Result<Vec<_>, _>>()?;
                if parsed.len() != FEATURES {
                    return Err(format!("weights for {label} have wrong length").into());
                }
                model.weights[index].copy_from_slice(&parsed);
            }
        }
    }

    Ok(model)
}

impl StreamModel {
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

fn label(svg: &mut String, text: &str, y: f32) {
    let _ = writeln!(
        svg,
        r#"<text x="18" y="{y:.1}" class="small">{}</text>"#,
        escape_xml(text)
    );
}

fn rect(svg: &mut String, x: f32, y: f32, w: f32, h: f32, fill: &str, opacity: f32) {
    let _ = writeln!(
        svg,
        r#"<rect x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}" fill="{fill}" opacity="{opacity:.3}"/>"#
    );
}

fn line(svg: &mut String, x1: f32, y1: f32, x2: f32, y2: f32, stroke: &str, width: f32) {
    let _ = writeln!(
        svg,
        r#"<line x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}" stroke="{stroke}" stroke-width="{width:.1}"/>"#
    );
}

fn label_index(label: &str) -> Option<usize> {
    LABELS
        .iter()
        .position(|candidate| candidate.eq_ignore_ascii_case(label))
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
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
    fn xml_escape_rewrites_reserved_chars() {
        assert_eq!(escape_xml("a&b<c>"), "a&amp;b&lt;c&gt;");
    }

    #[test]
    fn baseline_relative_rate_keeps_clean_air_silent() {
        let amplitude = ((250.0 - CLEAN_AIR_FLOOR_ADC) / (MAX_ADC - CLEAN_AIR_FLOOR_ADC))
            .clamp(0.0, 1.0);
        assert_eq!(rate_feature(amplitude, 5), 0.0);
    }
}
