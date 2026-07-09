use std::cmp::Ordering;
use std::env;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};

const CHANNELS: usize = 9;
const ACTIVE_SENSORS: usize = 8;
const FEATURES: usize = ACTIVE_SENSORS * 2;
const PATTERN_NEURONS: usize = 64;
const OUTPUTS: usize = 14;
const MAX_ADC: f32 = 4095.0;
const CLEAN_AIR_FLOOR_ADC: f32 = 300.0;
const MIN_DELTA_ADC: f32 = 25.0;
const DEFAULT_STREAM: &str = "data/streams/smoke_stream.csv";
const DEFAULT_MODEL: &str = "data/models/snn_stream_smoke.nsm";
const DEFAULT_OUT: &str = "data/streams/stream_preview.svg";
const DEFAULT_ROWS: usize = 3000;
const GRID: &str = "#d9e0e0";
const TEXT: &str = "#263235";
const MUTED: &str = "#657073";
const TEAL: &str = "#208f8a";
const BLUE: &str = "#2878a8";

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
    patterns: [f32; PATTERN_NEURONS],
    logits: [f32; OUTPUTS],
}

struct PatternBank {
    weights: [[i16; FEATURES]; PATTERN_NEURONS],
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
    let patterns = PatternBank::seeded();

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
            patterns: patterns.forward(&features),
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
    let left = 184.0_f32;
    let right = 28.0_f32;
    let top = 86.0_f32;
    let width = left + config.columns as f32 + right;
    let row_gap = 17.0_f32;
    let feature_gap = 10.0_f32;
    let pattern_gap = 5.5_f32;
    let section_gap = 48.0_f32;
    let adc_height = CHANNELS as f32 * row_gap;
    let feature_height = FEATURES as f32 * feature_gap;
    let pattern_height = PATTERN_NEURONS as f32 * pattern_gap;
    let label_height = OUTPUTS as f32 * row_gap;
    let total_height = top
        + 30.0
        + adc_height
        + section_gap
        + feature_height
        + section_gap
        + pattern_height
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
        r##"<rect width="100%" height="100%" fill="#f6f8f8"/><style>text{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:{TEXT}}} .small{{font-size:11px}} .tiny{{font-size:9px;fill:{MUTED}}} .title{{font:700 18px system-ui,-apple-system,sans-serif}} .section{{font:700 15px system-ui,-apple-system,sans-serif;fill:{TEXT}}}</style>"##
    );
    let _ = writeln!(
        svg,
        r#"<text x="18" y="30" class="title">stream timeline</text><text x="18" y="52" class="small">rows {}..{}  window={}  columns={}  gate&gt;{:.2}</text>"#,
        start_row,
        start_row + rows.len(),
        model.window,
        config.columns,
        config.gate_threshold
    );

    let mut y = top;
    render_truth_strip(&mut svg, rows, left, y, config.columns as f32, start_row);
    y += 30.0;
    render_adc(&mut svg, rows, left, y, config.columns as f32);
    y += adc_height + section_gap;
    render_features(&mut svg, frames, left, y, config.columns as f32, feature_gap);
    y += feature_height + section_gap;
    render_accordion(&mut svg, frames, left, y, config.columns as f32, pattern_gap);
    y += pattern_height + section_gap;
    render_label_heatmap(&mut svg, frames, left, y, config.columns as f32);
    y += label_height + section_gap;
    render_gated(&mut svg, frames, left, y, config.columns as f32, config.gate_threshold);
    let _ = writeln!(svg, "</svg>");
    svg
}

fn render_truth_strip(
    svg: &mut String,
    rows: &[StreamRow],
    left: f32,
    y: f32,
    panel_width: f32,
    start_row: usize,
) {
    let _ = writeln!(
        svg,
        r#"<text x="18" y="{:.1}" class="section">truth</text><text x="18" y="{:.1}" class="tiny">gray=no scent, color=active labels</text>"#,
        y - 12.0,
        y + 2.0
    );
    let columns = panel_width as usize;
    let rows_per_col = rows.len() as f32 / columns as f32;
    rounded_panel(svg, left, y - 2.0, panel_width, 18.0);
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
    for tick in 0..=4 {
        let x = left + panel_width * tick as f32 / 4.0;
        let row = start_row + (rows.len() * tick / 4);
        line(svg, x, y + 17.0, x, y + 23.0, GRID, 1.0);
        let _ = writeln!(
            svg,
            r#"<text x="{:.1}" y="{:.1}" class="tiny" text-anchor="middle">{}</text>"#,
            x,
            y + 34.0,
            row
        );
    }
}

fn render_adc(svg: &mut String, rows: &[StreamRow], left: f32, y: f32, panel_width: f32) {
    let _ = writeln!(
        svg,
        r#"<text x="18" y="{:.1}" class="section">ADC traces</text>"#,
        y - 16.0
    );
    for channel in 0..CHANNELS {
        let row_y = y + channel as f32 * 18.0;
        label(svg, CHANNEL_NAMES[channel], row_y + 4.0);
        line(svg, left, row_y, left + panel_width, row_y, GRID, 1.0);
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
            r##"<polyline points="{}" fill="none" stroke="{BLUE}" stroke-width="1.15" opacity="0.82"/>"##,
            points.trim()
        );
    }
}

fn render_features(
    svg: &mut String,
    frames: &[Frame],
    left: f32,
    y: f32,
    panel_width: f32,
    feature_gap: f32,
) {
    let _ = writeln!(
        svg,
        r#"<text x="18" y="{:.1}" class="section">rolling input features</text><text x="18" y="{:.1}" class="tiny">rate lanes first, delta lanes second</text>"#,
        y - 18.0,
        y - 5.0
    );
    for feature in 0..FEATURES {
        let row_y = y + feature as f32 * feature_gap;
        let name = if feature < ACTIVE_SENSORS {
            format!("rate adc{feature}")
        } else {
            format!("delta adc{}", feature - ACTIVE_SENSORS)
        };
        label(svg, &name, row_y + 4.0);
        if feature == ACTIVE_SENSORS {
            line(svg, left, row_y - 4.0, left + panel_width, row_y - 4.0, "#c8d1d1", 1.0);
        }
        heat_row(svg, frames.len(), panel_width as usize, left, row_y, 6.0, |start, end| {
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

fn render_accordion(
    svg: &mut String,
    frames: &[Frame],
    left: f32,
    y: f32,
    panel_width: f32,
    pattern_gap: f32,
) {
    let _ = writeln!(
        svg,
        r#"<text x="18" y="{:.1}" class="section">accordion motifs</text><text x="18" y="{:.1}" class="tiny">64 seeded pattern responses over rolling input features</text>"#,
        y - 18.0,
        y - 5.0
    );
    let groups = [
        (0, "single"),
        (16, "pair"),
        (32, "onset/tail"),
        (48, "cluster"),
    ];
    for (group_start, group_name) in groups {
        let group_y = y + group_start as f32 * pattern_gap - 1.0;
        line(svg, left, group_y, left + panel_width, group_y, "#c8d1d1", 1.0);
        let _ = writeln!(
            svg,
            r#"<text x="18" y="{:.1}" class="tiny">{group_name}</text>"#,
            group_y + 6.0
        );
    }

    for pattern in 0..PATTERN_NEURONS {
        let row_y = y + pattern as f32 * pattern_gap;
        if pattern % 4 == 0 {
            label(svg, &format!("p{pattern:02}"), row_y + 3.5);
        }
        heat_row(svg, frames.len(), panel_width as usize, left, row_y, 3.8, |start, end| {
            let mut sum = 0.0;
            let mut count = 0.0;
            for frame in &frames[start..end] {
                sum += frame.patterns[pattern];
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
        r#"<text x="18" y="{:.1}" class="section">label evidence</text><text x="18" y="{:.1}" class="tiny">positive model evidence, normalized in-window</text>"#,
        y - 18.0,
        y - 5.0
    );
    let max_abs = frames
        .iter()
        .flat_map(|frame| frame.logits.iter())
        .fold(1.0_f32, |max_value, value| max_value.max(value.abs()));
    for label_index in 0..OUTPUTS {
        let row_y = y + label_index as f32 * 18.0;
        label(svg, LABELS[label_index], row_y + 4.0);
        line(svg, left, row_y, left + panel_width, row_y, GRID, 1.0);
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
        r#"<text x="18" y="{:.1}" class="section">gated readout</text><text x="18" y="{:.1}" class="tiny">top 3 labels when score clears threshold</text>"#,
        y - 18.0,
        y - 5.0
    );
    let columns = panel_width as usize;
    let frames_per_col = frames.len() as f32 / columns as f32;
    for label_index in 0..OUTPUTS {
        let row_y = y + label_index as f32 * 18.0;
        label(svg, LABELS[label_index], row_y + 4.0);
        line(svg, left, row_y, left + panel_width, row_y, GRID, 1.0);
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
                let opacity = [0.78, 0.42, 0.24][rank];
                rect(
                    svg,
                    left + col as f32,
                    y + label_index as f32 * 18.0 - 5.0,
                    1.0,
                    8.0,
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
            let opacity = 0.08 + value * 0.68;
            rect(svg, left + col as f32, y, 1.0, height, TEAL, opacity);
        }
    }
}

fn rounded_panel(svg: &mut String, x: f32, y: f32, w: f32, h: f32) {
    let _ = writeln!(
        svg,
        r##"<rect x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}" rx="2" fill="#eef2f2" stroke="{GRID}" stroke-width="1"/>"##
    );
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
