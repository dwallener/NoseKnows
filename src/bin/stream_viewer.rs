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
const DEFAULT_OUT: &str = "data/streams/stream_viewer.html";
const DEFAULT_ROWS: usize = usize::MAX;
const DEFAULT_MAX_BUCKETS: usize = 12_000;
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
    "#b44bb6", "#c46bbb", "#d36b93", "#b96a34", "#9b7a44", "#8c6a3f", "#627b3e", "#4f7c5a",
    "#6a6254", "#3a8d76", "#d99a2b", "#3d93b8", "#4d9b4d", "#c85b5b",
];

struct Config {
    stream_path: PathBuf,
    model_path: PathBuf,
    output_path: PathBuf,
    start_row: usize,
    rows: usize,
    columns: usize,
    gate_threshold: f32,
    max_buckets: usize,
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

struct Bucket {
    start_row: usize,
    end_row: usize,
    truth_mask: u16,
    adc: [f32; CHANNELS],
    features: [f32; FEATURES],
    patterns: [f32; PATTERN_NEURONS],
    logits: [f32; OUTPUTS],
}

fn main() {
    if let Err(error) = run() {
        eprintln!("stream_viewer error: {error}");
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
    let end = start.saturating_add(config.rows).min(all_rows.len());
    let rows = &all_rows[start..end];
    if rows.len() < 2 {
        return Err("selected stream window needs at least two rows".into());
    }
    let frames = build_frames(rows, &model);
    let buckets = build_buckets(rows, &frames, start, config.max_buckets);
    let html = render_html(&buckets, &model, &config, start, end);
    if let Some(parent) = config.output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.output_path, html)?;
    println!(
        "Wrote stream viewer to {} rows={}..{} buckets={}",
        config.output_path.display(),
        start,
        end,
        buckets.len()
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
    let mut max_buckets = DEFAULT_MAX_BUCKETS;

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
            "--max-buckets" => {
                index += 1;
                max_buckets = args
                    .get(index)
                    .ok_or("--max-buckets requires a value")?
                    .parse()?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin stream_viewer -- [--stream data/streams/smoke_stream.csv] [--model data/models/snn_stream_smoke.nsm] [--out data/streams/stream_viewer.html] [--start-row 0] [--rows all] [--columns 900] [--max-buckets 12000]"
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
        max_buckets: max_buckets.clamp(100, 200_000),
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
            let amplitude = ((row.adc[sensor] - CLEAN_AIR_FLOOR_ADC)
                / (MAX_ADC - CLEAN_AIR_FLOOR_ADC))
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

fn build_buckets(
    rows: &[StreamRow],
    frames: &[Frame],
    global_start: usize,
    max_buckets: usize,
) -> Vec<Bucket> {
    let bucket_count = rows.len().min(max_buckets).max(1);
    let rows_per_bucket = rows.len() as f32 / bucket_count as f32;
    let mut buckets = Vec::with_capacity(bucket_count);

    for bucket_index in 0..bucket_count {
        let start = (bucket_index as f32 * rows_per_bucket).floor() as usize;
        let end = (((bucket_index + 1) as f32 * rows_per_bucket).ceil() as usize).min(rows.len());
        let end = end.max(start + 1).min(rows.len());
        let divisor = (end - start) as f32;
        let mut truth_counts = [0_usize; OUTPUTS];
        let mut adc = [0.0_f32; CHANNELS];
        let mut features = [0.0_f32; FEATURES];
        let mut patterns = [0.0_f32; PATTERN_NEURONS];
        let mut logits = [0.0_f32; OUTPUTS];

        for row in &rows[start..end] {
            for (label, active) in row.target.iter().enumerate() {
                if *active {
                    truth_counts[label] += 1;
                }
            }
            for (channel, value) in row.adc.iter().enumerate() {
                adc[channel] += *value;
            }
        }
        for frame in &frames[start..end] {
            for (feature, value) in frame.features.iter().enumerate() {
                features[feature] += *value;
            }
            for (pattern, value) in frame.patterns.iter().enumerate() {
                patterns[pattern] += *value;
            }
            for (label, value) in frame.logits.iter().enumerate() {
                logits[label] += *value;
            }
        }

        for value in &mut adc {
            *value /= divisor;
        }
        for value in &mut features {
            *value /= divisor;
        }
        for value in &mut patterns {
            *value /= divisor;
        }
        for value in &mut logits {
            *value /= divisor;
        }

        let mut truth_mask = 0_u16;
        for (label, count) in truth_counts.iter().enumerate() {
            if *count > 0 {
                truth_mask |= 1 << label;
            }
        }

        buckets.push(Bucket {
            start_row: global_start + start,
            end_row: global_start + end,
            truth_mask,
            adc,
            features,
            patterns,
            logits,
        });
    }

    buckets
}

fn render_html(
    buckets: &[Bucket],
    model: &StreamModel,
    config: &Config,
    start: usize,
    end: usize,
) -> String {
    let adc = matrix_json(buckets, |bucket| &bucket.adc);
    let features = matrix_json(buckets, |bucket| &bucket.features);
    let patterns = matrix_json(buckets, |bucket| &bucket.patterns);
    let logits = matrix_json(buckets, |bucket| &bucket.logits);
    let truth = buckets
        .iter()
        .map(|bucket| bucket.truth_mask.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let bucket_rows = buckets
        .iter()
        .map(|bucket| format!("[{},{}]", bucket.start_row, bucket.end_row))
        .collect::<Vec<_>>()
        .join(",");
    let labels = string_array_json(&LABELS);
    let label_colors = string_array_json(&LABEL_COLORS);
    let channels = string_array_json(&CHANNEL_NAMES);
    let feature_names = (0..FEATURES)
        .map(|feature| {
            if feature < ACTIVE_SENSORS {
                format!("rate adc{feature}")
            } else {
                format!("delta adc{}", feature - ACTIVE_SENSORS)
            }
        })
        .collect::<Vec<_>>();
    let feature_names = string_vec_json(&feature_names);
    let pattern_names = (0..PATTERN_NEURONS)
        .map(|pattern| {
            if pattern % 4 == 0 {
                format!("p{pattern:02}")
            } else {
                String::new()
            }
        })
        .collect::<Vec<_>>();
    let pattern_names = string_vec_json(&pattern_names);

    format!(
        r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NoseKnows Stream Viewer</title>
<style>
:root {{ color-scheme: light; --bg:#f6f8f8; --text:#263235; --muted:#657073; --grid:#d9e0e0; --teal:#208f8a; --blue:#2878a8; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
header {{ position:sticky; top:0; z-index:5; display:grid; grid-template-columns:1fr auto; gap:14px; align-items:center; padding:12px 14px; background:rgba(246,248,248,.94); border-bottom:1px solid var(--grid); backdrop-filter:blur(6px); }}
h1 {{ margin:0; font:700 22px system-ui,-apple-system,sans-serif; }}
.meta {{ margin-top:4px; color:var(--muted); font-size:12px; }}
.controls {{ display:flex; align-items:center; gap:8px; white-space:nowrap; }}
button {{ border:1px solid #b9c4c4; background:#fff; color:var(--text); border-radius:6px; padding:6px 9px; font:600 12px system-ui,-apple-system,sans-serif; cursor:pointer; }}
button:hover {{ background:#eef2f2; }}
input[type=range] {{ width:min(42vw,620px); }}
main {{ padding:10px 14px 22px; }}
canvas {{ display:block; width:100%; height:auto; background:var(--bg); }}
.hint {{ color:var(--muted); font-size:12px; margin:8px 0 0; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>stream viewer</h1>
    <div class="meta" id="meta"></div>
  </div>
  <div class="controls">
    <button id="prev">Prev</button>
    <input id="scrub" type="range" min="0" max="0" value="0" step="1">
    <button id="next">Next</button>
    <button id="active">Next active</button>
  </div>
</header>
<main>
  <canvas id="view" width="1400" height="1200"></canvas>
  <div class="hint">Drag the slider or use left/right arrow keys. The minimap shows the full downsampled stream; the main panels show the selected window.</div>
</main>
<script>
const DATA = {{
  rows:[{bucket_rows}],
  truth:[{truth}],
  adc:{adc},
  features:{features},
  patterns:{patterns},
  logits:{logits},
  labels:{labels},
  labelColors:{label_colors},
  channels:{channels},
  featureNames:{feature_names},
  patternNames:{pattern_names},
  streamStart:{start},
  streamEnd:{end},
  window:{window},
  gateThreshold:{gate_threshold},
  modelWindow:{model_window}
}};

const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const scrub = document.getElementById('scrub');
const meta = document.getElementById('meta');
const left = 218, right = 24, top = 74, panelW = 1120;
const viewportBuckets = Math.min(DATA.window, DATA.rows.length);
scrub.max = Math.max(0, DATA.rows.length - viewportBuckets);
scrub.value = 0;

function color(hex, alpha) {{
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${{(n >> 16) & 255}},${{(n >> 8) & 255}},${{n & 255}},${{alpha}})`;
}}
function line(x1,y1,x2,y2,stroke='#d9e0e0',w=1) {{
  ctx.strokeStyle = stroke; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
}}
function text(s,x,y,size=12,fill='#263235',weight='400',align='left',family='ui-monospace,SFMono-Regular,Menlo,monospace') {{
  ctx.font = `${{weight}} ${{size}}px ${{family}}`; ctx.fillStyle = fill; ctx.textAlign = align; ctx.fillText(s,x,y);
}}
function rect(x,y,w,h,fill) {{ ctx.fillStyle = fill; ctx.fillRect(x,y,w,h); }}
function section(title, subtitle, y) {{
  text(title, 8, y - 18, 18, '#263235', '700', 'left', 'system-ui,-apple-system,sans-serif');
  if (subtitle) text(subtitle, 8, y - 4, 11, '#657073');
}}
function bucketsFor(start) {{
  const end = Math.min(DATA.rows.length, start + viewportBuckets);
  return [start, end];
}}
function valueRange(matrix, start, end, lane) {{
  let max = 1;
  for (let i=start; i<end; i++) max = Math.max(max, Math.abs(matrix[i][lane]));
  return max;
}}
function heatRow(matrix, lane, start, end, y, h, col, normalizer=1) {{
  const span = end - start;
  for (let x=0; x<panelW; x++) {{
    const a = start + Math.floor(x * span / panelW);
    const b = Math.min(end, start + Math.ceil((x + 1) * span / panelW));
    let sum = 0, count = 0;
    for (let i=a; i<b; i++) {{ sum += matrix[i][lane]; count++; }}
    const v = Math.max(0, Math.min(1, (sum / Math.max(1,count)) / normalizer));
    if (v > 0.001) rect(left + x, y, 1, h, color(col, 0.08 + v * 0.68));
  }}
}}
function renderMinimap(start) {{
  const y = 48, h = 18;
  rect(left, y, panelW, h, '#eef2f2');
  const span = DATA.rows.length;
  for (let x=0; x<panelW; x++) {{
    const a = Math.floor(x * span / panelW);
    const b = Math.min(span, Math.ceil((x + 1) * span / panelW));
    let active = 0;
    for (let i=a; i<b; i++) active |= DATA.truth[i];
    if (active === 0) rect(left + x, y, 1, h, 'rgba(190,198,198,.42)');
    else {{
      let slot = 0;
      for (let label=0; label<DATA.labels.length && slot<3; label++) {{
        if (active & (1 << label)) rect(left + x, y + slot++ * 6, 1, 6, color(DATA.labelColors[label], .78));
      }}
    }}
  }}
  const vx = left + start / Math.max(1, span - viewportBuckets) * (panelW - Math.max(4, panelW * viewportBuckets / span));
  const vw = Math.max(4, panelW * viewportBuckets / span);
  ctx.strokeStyle = '#263235'; ctx.lineWidth = 1.5; ctx.strokeRect(vx, y - 2, vw, h + 4);
}}
function renderTruth(start, end, y) {{
  section('truth', 'gray=no scent, color=active labels', y);
  for (let x=0; x<panelW; x++) {{
    const idx = start + Math.floor(x * (end - start) / panelW);
    const mask = DATA.truth[idx] || 0;
    if (mask === 0) rect(left + x, y, 1, 12, 'rgba(190,198,198,.48)');
    else {{
      let slot = 0;
      for (let label=0; label<DATA.labels.length && slot<3; label++) {{
        if (mask & (1 << label)) rect(left + x, y + slot++ * 4, 1, 4, color(DATA.labelColors[label], .85));
      }}
    }}
  }}
  for (let tick=0; tick<=4; tick++) {{
    const x = left + panelW * tick / 4;
    const idx = start + Math.floor((end - start) * tick / 4);
    line(x, y + 17, x, y + 23);
    text(DATA.rows[idx]?.[0] ?? DATA.streamEnd, x, y + 34, 11, '#657073', '400', 'center');
  }}
}}
function renderAdc(start,end,y) {{
  section('ADC traces', '', y);
  for (let lane=0; lane<DATA.channels.length; lane++) {{
    const rowY = y + lane * 18;
    text(DATA.channels[lane], 8, rowY + 4, 12);
    line(left, rowY, left + panelW, rowY);
    ctx.strokeStyle = '#2878a8'; ctx.lineWidth = 1.15; ctx.globalAlpha = .82; ctx.beginPath();
    for (let x=0; x<panelW; x++) {{
      const idx = start + Math.floor(x * (end - start) / panelW);
      const amp = Math.max(0, Math.min(1, DATA.adc[idx][lane] / 4095));
      const px = left + x, py = rowY + 7 - amp * 14;
      if (x === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }}
    ctx.stroke(); ctx.globalAlpha = 1;
  }}
}}
function renderHeatPanel(title, subtitle, names, matrix, start, end, y, rowGap, h, normalizer, dividers=[]) {{
  section(title, subtitle, y);
  for (let lane=0; lane<names.length; lane++) {{
    const rowY = y + lane * rowGap;
    if (names[lane]) text(names[lane], lane < 16 && title === 'accordion motifs' ? 88 : 8, rowY + 4, 12);
    if (dividers.includes(lane)) line(left, rowY - 4, left + panelW, rowY - 4, '#c8d1d1');
    heatRow(matrix, lane, start, end, rowY, h, '#208f8a', normalizer(lane));
  }}
}}
function renderLabelPanel(start,end,y,gated=false) {{
  section(gated ? 'gated readout' : 'label evidence', gated ? 'top 3 labels when score clears threshold' : 'positive model evidence, normalized in-window', y);
  let maxLogit = 1;
  for (let i=start; i<end; i++) for (const v of DATA.logits[i]) maxLogit = Math.max(maxLogit, Math.abs(v));
  for (let label=0; label<DATA.labels.length; label++) {{
    const rowY = y + label * 18;
    text(DATA.labels[label], 8, rowY + 4, 12);
    line(left, rowY, left + panelW, rowY);
  }}
  for (let x=0; x<panelW; x++) {{
    const idx = start + Math.floor(x * (end - start) / panelW);
    if (!gated) {{
      for (let label=0; label<DATA.labels.length; label++) {{
        const v = Math.max(0, DATA.logits[idx][label]) / maxLogit;
        if (v > .001) rect(left + x, y + label * 18 - 5, 1, 10, color('#208f8a', .08 + v * .68));
      }}
    }} else {{
      const top = DATA.logits[idx].map((v,i)=>[i,v]).sort((a,b)=>b[1]-a[1]).slice(0,3);
      top.forEach(([label,score], rank) => {{
        if (score > DATA.gateThreshold) rect(left + x, y + label * 18 - 5, 1, 8, color(DATA.labelColors[label], [.78,.42,.24][rank]));
      }});
    }}
  }}
}}
function draw() {{
  const start = Number(scrub.value);
  const [a,b] = bucketsFor(start);
  canvas.width = Math.max(1220, window.innerWidth - 28);
  const scale = Math.min(1, (canvas.width - left - right) / panelW);
  canvas.height = 1540 * scale;
  ctx.setTransform(scale,0,0,scale,0,0);
  rect(0,0,canvas.width / scale,canvas.height / scale,'#f6f8f8');
  text('stream timeline', 8, 24, 22, '#263235', '700', 'left', 'system-ui,-apple-system,sans-serif');
  meta.textContent = `rows ${{DATA.rows[a][0]}}..${{DATA.rows[b-1][1]}} / ${{DATA.streamStart}}..${{DATA.streamEnd}} · buckets ${{a}}..${{b}} / ${{DATA.rows.length}} · model window=${{DATA.modelWindow}}`;
  renderMinimap(start);
  let y = top;
  renderTruth(a,b,y); y += 58;
  renderAdc(a,b,y); y += 210;
  renderHeatPanel('rolling input features', 'rate lanes first, delta lanes second', DATA.featureNames, DATA.features, a,b,y,11,6,()=>1,[8]); y += 238;
  renderHeatPanel('accordion motifs', '64 seeded pattern responses over rolling input features', DATA.patternNames, DATA.patterns, a,b,y,7,3.8,()=>1,[0,16,32,48]);
  text('single',8,y+5,11,'#657073'); text('pair',8,y+16*7+5,11,'#657073'); text('onset/tail',8,y+32*7+5,11,'#657073'); text('cluster',8,y+48*7+5,11,'#657073');
  y += 506;
  renderLabelPanel(a,b,y,false); y += 300;
  renderLabelPanel(a,b,y,true);
}}
function jumpActive(dir=1) {{
  let i = Number(scrub.value) + dir;
  while (i >= 0 && i < DATA.rows.length - viewportBuckets) {{
    let active = false;
    for (let j=i; j<Math.min(DATA.rows.length, i + viewportBuckets); j++) if (DATA.truth[j]) {{ active = true; break; }}
    if (active) {{ scrub.value = i; draw(); return; }}
    i += dir * viewportBuckets;
  }}
}}
scrub.addEventListener('input', draw);
document.getElementById('prev').onclick = () => {{ scrub.value = Math.max(0, Number(scrub.value) - Math.floor(viewportBuckets * .75)); draw(); }};
document.getElementById('next').onclick = () => {{ scrub.value = Math.min(Number(scrub.max), Number(scrub.value) + Math.floor(viewportBuckets * .75)); draw(); }};
document.getElementById('active').onclick = () => jumpActive(1);
window.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft') document.getElementById('prev').click();
  if (e.key === 'ArrowRight') document.getElementById('next').click();
}});
window.addEventListener('resize', draw);
draw();
</script>
</body>
</html>"##,
        bucket_rows = bucket_rows,
        truth = truth,
        adc = adc,
        features = features,
        patterns = patterns,
        logits = logits,
        labels = labels,
        label_colors = label_colors,
        channels = channels,
        feature_names = feature_names,
        pattern_names = pattern_names,
        start = start,
        end = end,
        window = config.columns,
        gate_threshold = config.gate_threshold,
        model_window = model.window,
    )
}

fn matrix_json<const N: usize>(buckets: &[Bucket], value: impl Fn(&Bucket) -> &[f32; N]) -> String {
    let mut output = String::new();
    output.push('[');
    for (bucket_index, bucket) in buckets.iter().enumerate() {
        if bucket_index > 0 {
            output.push(',');
        }
        output.push('[');
        for (index, value) in value(bucket).iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            let _ = write!(output, "{value:.4}");
        }
        output.push(']');
    }
    output.push(']');
    output
}

fn string_array_json<const N: usize>(values: &[&str; N]) -> String {
    let mut output = String::new();
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(output, "\"{}\"", escape_json(value));
    }
    output.push(']');
    output
}

fn string_vec_json(values: &[String]) -> String {
    let mut output = String::new();
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(output, "\"{}\"", escape_json(value));
    }
    output.push(']');
    output
}

fn escape_json(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
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
        rows.push(StreamRow { target, adc });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escape_rewrites_reserved_chars() {
        assert_eq!(escape_json(r#"a\b"c"#), r#"a\\b\"c"#);
    }

    #[test]
    fn baseline_relative_rate_keeps_clean_air_silent() {
        let amplitude =
            ((250.0 - CLEAN_AIR_FLOOR_ADC) / (MAX_ADC - CLEAN_AIR_FLOOR_ADC)).clamp(0.0, 1.0);
        assert_eq!(rate_feature(amplitude, 5), 0.0);
    }
}
