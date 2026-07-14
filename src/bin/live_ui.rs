use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_STATE: &str = "data/live/injector_state.json";
const DEFAULT_INPUT: &str = "data/live/input_frames.csv";
const DEFAULT_INPUT_EVENTS: &str = "data/live/input_events.csv";
const DEFAULT_RESULTS: &str = "data/live/model_results.csv";
const DEFAULT_EVENTS: &str = "data/live/events.csv";
const DEFAULT_MODEL: &str = "data/models/peak_pair_readout.npm";
const DEFAULT_GRID_RESULTS: &str = "data/live/grid_model_results.csv";
const DEFAULT_GRID_EVENTS: &str = "data/live/grid_events.csv";
const DEFAULT_GRID_MODEL: &str = "data/models/grid8_readout.ngm";
const MAX_BODY_BYTES: usize = 128 * 1024;

struct Config {
    state_path: PathBuf,
    input_path: PathBuf,
    input_events_path: PathBuf,
    results_path: PathBuf,
    events_path: PathBuf,
    model_path: PathBuf,
    grid_results_path: PathBuf,
    grid_events_path: PathBuf,
    grid_model_path: PathBuf,
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn main() -> std::io::Result<()> {
    let config = parse_args();
    let listener = bind_first_available(7890, 7909)?;
    let address = listener.local_addr()?;
    println!("NoseKnows live UI running at http://{address}");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(error) = handle_connection(stream, &config) {
                    eprintln!("live_ui request failed: {error}");
                }
            }
            Err(error) => eprintln!("live_ui connection failed: {error}"),
        }
    }
    Ok(())
}

fn parse_args() -> Config {
    let mut state_path = PathBuf::from(DEFAULT_STATE);
    let mut input_path = PathBuf::from(DEFAULT_INPUT);
    let mut input_events_path = PathBuf::from(DEFAULT_INPUT_EVENTS);
    let mut results_path = PathBuf::from(DEFAULT_RESULTS);
    let mut events_path = PathBuf::from(DEFAULT_EVENTS);
    let mut model_path = PathBuf::from(DEFAULT_MODEL);
    let mut grid_results_path = PathBuf::from(DEFAULT_GRID_RESULTS);
    let mut grid_events_path = PathBuf::from(DEFAULT_GRID_EVENTS);
    let mut grid_model_path = PathBuf::from(DEFAULT_GRID_MODEL);

    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--state" => {
                index += 1;
                state_path = PathBuf::from(args.get(index).expect("--state requires a path"));
            }
            "--input" => {
                index += 1;
                input_path = PathBuf::from(args.get(index).expect("--input requires a path"));
            }
            "--input-events" => {
                index += 1;
                input_events_path =
                    PathBuf::from(args.get(index).expect("--input-events requires a path"));
            }
            "--results" => {
                index += 1;
                results_path = PathBuf::from(args.get(index).expect("--results requires a path"));
            }
            "--events" => {
                index += 1;
                events_path = PathBuf::from(args.get(index).expect("--events requires a path"));
            }
            "--model" => {
                index += 1;
                model_path = PathBuf::from(args.get(index).expect("--model requires a path"));
            }
            "--grid-results" => {
                index += 1;
                grid_results_path =
                    PathBuf::from(args.get(index).expect("--grid-results requires a path"));
            }
            "--grid-events" => {
                index += 1;
                grid_events_path =
                    PathBuf::from(args.get(index).expect("--grid-events requires a path"));
            }
            "--grid-model" => {
                index += 1;
                grid_model_path =
                    PathBuf::from(args.get(index).expect("--grid-model requires a path"));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin live_ui -- [--state data/live/injector_state.json] [--input data/live/input_frames.csv] [--input-events data/live/input_events.csv] [--results data/live/model_results.csv] [--events data/live/events.csv] [--model data/models/peak_pair_readout.npm] [--grid-results data/live/grid_model_results.csv] [--grid-events data/live/grid_events.csv] [--grid-model data/models/grid8_readout.ngm]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
        index += 1;
    }

    Config {
        state_path,
        input_path,
        input_events_path,
        results_path,
        events_path,
        model_path,
        grid_results_path,
        grid_events_path,
        grid_model_path,
    }
}

fn bind_first_available(start_port: u16, end_port: u16) -> std::io::Result<TcpListener> {
    let mut last_error = None;
    for port in start_port..=end_port {
        match TcpListener::bind(("127.0.0.1", port)) {
            Ok(listener) => return Ok(listener),
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::AddrNotAvailable, "no candidate ports")
    }))
}

fn handle_connection(mut stream: TcpStream, config: &Config) -> std::io::Result<()> {
    let request = read_request(&mut stream)?;
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") => respond(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            app_html().as_bytes(),
        ),
        ("GET", "/api/state") => {
            let body = read_or_default_state(&config.state_path);
            respond(
                &mut stream,
                "200 OK",
                "application/json; charset=utf-8",
                body.as_bytes(),
            )
        }
        ("POST", "/api/state") => {
            if request.body.len() > MAX_BODY_BYTES {
                return respond_json(
                    &mut stream,
                    "413 Payload Too Large",
                    false,
                    "state too large",
                );
            }
            if let Some(parent) = config.state_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&config.state_path, &request.body)?;
            respond_json(&mut stream, "200 OK", true, "state saved")
        }
        ("POST", "/api/materialize") => {
            let output = run_materializer(config);
            respond_command(&mut stream, output)
        }
        ("POST", "/api/run-headless") => {
            let materialized = run_materializer(config);
            if !materialized.status.success() {
                return respond_command(&mut stream, materialized);
            }
            respond_command(&mut stream, run_headless(config))
        }
        ("POST", "/api/run-grid8") => {
            let materialized = run_materializer(config);
            if !materialized.status.success() {
                return respond_command(&mut stream, materialized);
            }
            respond_command(&mut stream, run_grid8_headless(config))
        }
        ("POST", "/api/train-grid8") => respond_command(&mut stream, train_grid8(config)),
        ("GET", "/api/input-frames.csv") => respond_file(&mut stream, &config.input_path),
        ("GET", "/api/input-events.csv") => respond_file(&mut stream, &config.input_events_path),
        ("GET", "/api/results.csv") => respond_file(&mut stream, &config.results_path),
        ("GET", "/api/events.csv") => respond_file(&mut stream, &config.events_path),
        ("GET", "/api/grid-results.csv") => respond_file(&mut stream, &config.grid_results_path),
        ("GET", "/api/grid-events.csv") => respond_file(&mut stream, &config.grid_events_path),
        _ => respond(
            &mut stream,
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"not found",
        ),
    }
}

fn read_request(stream: &mut TcpStream) -> std::io::Result<Request> {
    let mut buffer = Vec::new();
    let mut temp = [0_u8; 8192];
    let bytes_read = stream.read(&mut temp)?;
    buffer.extend_from_slice(&temp[..bytes_read]);

    let header_end = find_header_end(&buffer).unwrap_or(buffer.len());
    let header_text = String::from_utf8_lossy(&buffer[..header_end]);
    let first_line = header_text.lines().next().unwrap_or("GET / HTTP/1.1");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("GET").to_string();
    let path = parts.next().unwrap_or("/").to_string();
    let content_length = header_text
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0)
        .min(MAX_BODY_BYTES);

    let body_start = (header_end + 4).min(buffer.len());
    let mut body = buffer[body_start..].to_vec();
    while body.len() < content_length {
        let bytes_read = stream.read(&mut temp)?;
        if bytes_read == 0 {
            break;
        }
        body.extend_from_slice(&temp[..bytes_read]);
    }
    body.truncate(content_length);

    Ok(Request { method, path, body })
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn respond(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let headers = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes())?;
    stream.write_all(body)
}

fn respond_json(
    stream: &mut TcpStream,
    status: &str,
    ok: bool,
    message: &str,
) -> std::io::Result<()> {
    let body = format!("{{\"ok\":{},\"message\":\"{}\"}}", ok, json_escape(message));
    respond(
        stream,
        status,
        "application/json; charset=utf-8",
        body.as_bytes(),
    )
}

fn respond_file(stream: &mut TcpStream, path: &Path) -> std::io::Result<()> {
    match fs::read(path) {
        Ok(body) => respond(stream, "200 OK", "text/csv; charset=utf-8", &body),
        Err(_) => respond(stream, "200 OK", "text/csv; charset=utf-8", b""),
    }
}

fn respond_command(stream: &mut TcpStream, output: std::process::Output) -> std::io::Result<()> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let body = format!(
        "{{\"ok\":{},\"status\":{},\"stdout\":\"{}\",\"stderr\":\"{}\"}}",
        output.status.success(),
        output.status.code().unwrap_or(-1),
        json_escape(&stdout),
        json_escape(&stderr)
    );
    respond(
        stream,
        "200 OK",
        "application/json; charset=utf-8",
        body.as_bytes(),
    )
}

fn read_or_default_state(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|_| DEFAULT_STATE_JSON.to_string())
}

fn run_materializer(config: &Config) -> std::process::Output {
    Command::new("python3")
        .args([
            "tools/live/inject_chunks.py",
            "--state",
            &config.state_path.display().to_string(),
            "--out",
            &config.input_path.display().to_string(),
            "--events",
            &config.input_events_path.display().to_string(),
        ])
        .output()
        .unwrap_or_else(command_error)
}

fn run_headless(config: &Config) -> std::process::Output {
    Command::new("cargo")
        .args([
            "run",
            "--bin",
            "live_headless",
            "--",
            "--input",
            &config.input_path.display().to_string(),
            "--model",
            &config.model_path.display().to_string(),
            "--out-results",
            &config.results_path.display().to_string(),
            "--out-events",
            &config.events_path.display().to_string(),
            "--run-id",
            "live_ui",
        ])
        .output()
        .unwrap_or_else(command_error)
}

fn run_grid8_headless(config: &Config) -> std::process::Output {
    Command::new("cargo")
        .args([
            "run",
            "--bin",
            "grid_live_headless",
            "--",
            "--input",
            &config.input_path.display().to_string(),
            "--model",
            &config.grid_model_path.display().to_string(),
            "--out-results",
            &config.grid_results_path.display().to_string(),
            "--out-events",
            &config.grid_events_path.display().to_string(),
            "--run-id",
            "live_ui_grid8",
        ])
        .output()
        .unwrap_or_else(command_error)
}

fn train_grid8(config: &Config) -> std::process::Output {
    Command::new("cargo")
        .args([
            "run",
            "--bin",
            "grid_train",
            "--",
            "--data",
            "data/views/peak_single_note",
            "--out",
            &config.grid_model_path.display().to_string(),
            "--epochs",
            "250",
            "--lookback-secs",
            "8",
        ])
        .output()
        .unwrap_or_else(command_error)
}

fn command_error(error: std::io::Error) -> std::process::Output {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        std::process::Output {
            status: std::process::ExitStatus::from_raw(1),
            stdout: Vec::new(),
            stderr: error.to_string().into_bytes(),
        }
    }
    #[cfg(not(unix))]
    {
        panic!("command failed to start: {error}");
    }
}

fn json_escape(value: &str) -> String {
    value
        .chars()
        .flat_map(|ch| match ch {
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\n' => "\\n".chars().collect::<Vec<_>>(),
            '\r' => "\\r".chars().collect::<Vec<_>>(),
            '\t' => "\\t".chars().collect::<Vec<_>>(),
            _ => vec![ch],
        })
        .collect()
}

const DEFAULT_STATE_JSON: &str = r#"{
  "sample_period_ms": 100,
  "seed": 20260714,
  "sequence": [
    {"notes": [], "duration_secs": 4},
    {"notes": ["Citrus"], "duration_secs": 10, "intensity": 1.0},
    {"notes": [], "duration_secs": 4}
  ]
}
"#;

const LABELS_JSON: &str = r#"["Floral","Soft Floral","Floral Amber","Amber","Soft Amber","Woody Amber","Woods","Mossy Woods","Dry Woods","Aromatic","Citrus","Water","Green","Fruity"]"#;

const APP_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NoseKnows Live Injector</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1f2529;
      --muted: #65717a;
      --line: #d8e0e5;
      --bg: #f5f7f8;
      --panel: #ffffff;
      --teal: #208c86;
      --blue: #3478b6;
      --rose: #c95f8c;
      --amber: #c88436;
      --green: #6c8f45;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.94);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 {
      font-size: 20px;
      line-height: 1.1;
      margin: 0;
      letter-spacing: 0;
    }
    .sub {
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
    }
    main {
      display: grid;
      grid-template-columns: 370px 1fr;
      min-height: calc(100vh - 64px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 16px;
      overflow: auto;
    }
    section {
      min-width: 0;
      padding: 16px;
      overflow: hidden;
    }
    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    button {
      height: 36px;
      border: 1px solid #b9c5cc;
      border-radius: 7px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-weight: 700;
      padding: 0 12px;
      cursor: pointer;
    }
    button.primary {
      background: var(--teal);
      border-color: var(--teal);
      color: white;
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    .group {
      border-top: 1px solid var(--line);
      padding-top: 14px;
      margin-top: 14px;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 6px;
    }
    select, input {
      width: 100%;
      height: 34px;
      border: 1px solid #c9d3d9;
      border-radius: 7px;
      padding: 0 9px;
      font: inherit;
      background: white;
    }
    .toolbar select {
      width: 146px;
      height: 36px;
      font-weight: 700;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 84px 84px;
      gap: 8px;
      align-items: end;
      margin-bottom: 8px;
    }
    .checks {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px 10px;
      margin-top: 8px;
    }
    .checks label {
      display: flex;
      align-items: center;
      gap: 7px;
      color: var(--ink);
      font-weight: 500;
      margin: 0;
      min-height: 26px;
    }
    .checks input {
      width: 16px;
      height: 16px;
    }
    .sequence {
      display: grid;
      gap: 8px;
    }
    .segment {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #fafcfd;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
    }
    .segment strong {
      display: block;
      font-size: 13px;
      margin-bottom: 3px;
    }
    .segment span {
      color: var(--muted);
      font-size: 12px;
    }
    .timeline {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      overflow: auto;
      max-height: calc(100vh - 108px);
    }
    .timeline-note {
      min-width: 980px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .lane {
      display: grid;
      grid-template-columns: 132px 1fr;
      min-width: 980px;
      align-items: center;
      min-height: 27px;
      border-top: 1px solid #edf1f3;
    }
    .lane:first-child { border-top: 0; }
    .label {
      font-size: 12px;
      color: #3f4a50;
      white-space: nowrap;
      padding-right: 8px;
    }
    .track {
      height: 20px;
      position: relative;
      background-image: linear-gradient(to right, rgba(30,40,46,0.05) 1px, transparent 1px);
      background-size: 40px 100%;
    }
    .bar {
      position: absolute;
      top: 4px;
      height: 12px;
      border-radius: 2px;
      opacity: 0.72;
    }
    .bar.truth { background: var(--amber); }
    .bar.pred { background: var(--teal); }
    .bar.noise { background: var(--rose); }
    .playhead {
      position: absolute;
      top: 1px;
      bottom: 1px;
      width: 2px;
      background: var(--blue);
      box-shadow: 0 0 0 1px rgba(255,255,255,0.75);
      left: 0;
      pointer-events: none;
      z-index: 1;
    }
    .log {
      margin-top: 12px;
      height: 120px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #11181c;
      color: #cfe3df;
      font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;
    }
    .metric-strip {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 8px 10px;
      min-width: 112px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 3px;
    }
    .metric strong {
      font-size: 16px;
      font-variant-numeric: tabular-nums;
    }
    @media (max-width: 920px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>NoseKnows live injector</h1>
      <div class="sub">Daft/Python input orchestration, Rust model execution</div>
    </div>
    <div class="toolbar">
      <select id="model-kind" title="Headless model">
        <option value="peak">Peak-pair</option>
        <option value="grid8">Grid8 rolling</option>
      </select>
      <button id="train-grid8">Train Grid8</button>
      <button id="save">Save State</button>
      <button id="materialize">Materialize</button>
      <button id="run" class="primary">Run Headless</button>
      <button id="play">Play</button>
    </div>
  </header>
  <main>
    <aside>
      <div class="row">
        <div>
          <label for="preset">Preset</label>
          <select id="preset">
            <option value="custom">Custom</option>
            <option value="no-scent">No scent</option>
            <option value="single">Single note</option>
            <option value="pair">Two notes</option>
            <option value="triple">Three notes</option>
          </select>
        </div>
        <div>
          <label for="duration">Seconds</label>
          <input id="duration" type="number" min="1" max="120" value="10">
        </div>
        <div>
          <label for="intensity">Intensity</label>
          <input id="intensity" type="number" min="0" max="1.5" step="0.05" value="1">
        </div>
      </div>
      <div class="checks" id="notes"></div>
      <div class="toolbar group">
        <button id="add">Add Segment</button>
        <button id="add-gap">Add No Scent</button>
        <button id="clear">Clear</button>
      </div>
      <div class="group">
        <label>Sequence</label>
        <div id="sequence" class="sequence"></div>
      </div>
    </aside>
    <section>
      <div class="metric-strip">
        <div class="metric"><span>View</span><strong id="view-mode">Input</strong></div>
        <div class="metric"><span>Frames</span><strong id="frames">0</strong></div>
        <div class="metric"><span>Segments</span><strong id="segments">0</strong></div>
        <div class="metric"><span>Emitted Rows</span><strong id="emitted">0</strong></div>
        <div class="metric"><span>Silent Rows</span><strong id="silent">0</strong></div>
      </div>
      <div class="timeline">
        <div class="timeline-note" id="timeline-note"></div>
        <div id="timeline"></div>
      </div>
      <div class="log" id="log"></div>
    </section>
  </main>
<script>
const LABELS = __LABELS__;
let state = {sample_period_ms: 100, seed: 20260714, sequence: []};
let currentFrameCount = 0;
let currentFrame = 0;
let playTimer = null;
const colors = {
  "Floral":"#b44bb6","Soft Floral":"#c46bbb","Floral Amber":"#d36b93","Amber":"#b96a34",
  "Soft Amber":"#9b7a44","Woody Amber":"#8c6a3f","Woods":"#627b3e","Mossy Woods":"#4f7c5a",
  "Dry Woods":"#6a6254","Aromatic":"#3a8d76","Citrus":"#d99a2b","Water":"#3d93b8",
  "Green":"#4d9b4d","Fruity":"#c85b5b"
};

function initNotes() {
  const root = document.getElementById('notes');
  root.innerHTML = LABELS.map(label => `<label><input type="checkbox" value="${label}">${label}</label>`).join('');
}

function selectedNotes() {
  return [...document.querySelectorAll('#notes input:checked')].map(input => input.value).slice(0, 3);
}

function addSegment(notes = selectedNotes()) {
  const duration = Number(document.getElementById('duration').value || 10);
  const intensity = Number(document.getElementById('intensity').value || 1);
  state.sequence.push({notes, duration_secs: duration, intensity});
  renderSequence();
}

function applyPreset() {
  const preset = document.getElementById('preset').value;
  document.querySelectorAll('#notes input').forEach(input => input.checked = false);
  const pick = names => document.querySelectorAll('#notes input').forEach(input => input.checked = names.includes(input.value));
  if (preset === 'no-scent') pick([]);
  if (preset === 'single') pick(['Citrus']);
  if (preset === 'pair') pick(['Water', 'Citrus']);
  if (preset === 'triple') pick(['Amber', 'Woody Amber', 'Floral Amber']);
}

function renderSequence() {
  const root = document.getElementById('sequence');
  root.innerHTML = state.sequence.map((segment, index) => {
    const notes = segment.notes && segment.notes.length ? segment.notes.join(' + ') : 'No Scent';
    return `<div class="segment">
      <div><strong>${notes}</strong><span>${segment.duration_secs || 0}s · intensity ${segment.intensity ?? 1}</span></div>
      <button data-remove="${index}">Remove</button>
    </div>`;
  }).join('');
  root.querySelectorAll('[data-remove]').forEach(button => button.onclick = () => {
    state.sequence.splice(Number(button.dataset.remove), 1);
    renderSequence();
  });
}

async function loadState() {
  const response = await fetch('/api/state');
  state = await response.json();
  if (!Array.isArray(state.sequence)) state.sequence = [];
  renderSequence();
}

async function saveState() {
  const response = await fetch('/api/state', {method: 'POST', body: JSON.stringify(state, null, 2)});
  logJson(await response.json());
}

function selectedModelKind() {
  return document.getElementById('model-kind').value;
}

function selectedModelName() {
  return selectedModelKind() === 'grid8' ? 'Grid8' : 'Peak-pair';
}

async function postAction(path, loadModelResults = true) {
  setBusy(true);
  try {
    await saveState();
    const response = await fetch(path, {method: 'POST'});
    const json = await response.json();
    logJson(json);
    if (loadModelResults) {
      await loadResults();
    } else {
      await loadInputPreview();
    }
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  document.querySelectorAll('button').forEach(button => {
    if (button.id !== 'play') button.disabled = busy;
  });
}

function logJson(json) {
  const log = document.getElementById('log');
  const text = [`ok=${json.ok} status=${json.status ?? ''}`, json.stdout || '', json.stderr || '', json.message || ''].filter(Boolean).join('\n');
  log.textContent = text + '\n\n' + log.textContent;
}

async function loadResults() {
  const kind = selectedModelKind();
  const resultsPath = kind === 'grid8' ? '/api/grid-results.csv' : '/api/results.csv';
  const [resultsText, inputEventsText] = await Promise.all([
    fetch(resultsPath).then(response => response.text()),
    fetch('/api/input-events.csv').then(response => response.text())
  ]);
  const results = parseCsv(resultsText);
  const inputEvents = parseCsv(inputEventsText);
  renderTimeline(results, inputEvents, results.length, `Model: ${selectedModelName()}`);
}

async function loadInputPreview() {
  const [inputText, inputEventsText] = await Promise.all([
    fetch('/api/input-frames.csv').then(response => response.text()),
    fetch('/api/input-events.csv').then(response => response.text())
  ]);
  const inputRows = parseCsv(inputText);
  const inputEvents = parseCsv(inputEventsText);
  renderTimeline([], inputEvents, inputRows.length, 'Input');
}

function renderTimeline(results, inputEvents, frameCount, mode = 'Input') {
  currentFrameCount = Math.max(0, frameCount);
  currentFrame = 0;
  document.getElementById('view-mode').textContent = mode;
  document.getElementById('frames').textContent = frameCount;
  document.getElementById('segments').textContent = inputEvents.length;
  const modelView = mode.startsWith('Model');
  document.getElementById('emitted').textContent = modelView ? results.filter(row => row.silent === 'false').length : '—';
  document.getElementById('silent').textContent = modelView ? results.filter(row => row.silent === 'true').length : '—';
  document.getElementById('timeline-note').textContent =
    mode.startsWith('Model')
      ? `${mode} output from the latest headless run.`
      : mode === 'Unsaved'
        ? 'State changed. Save and materialize to refresh the input preview.'
        : 'Input preview only. Click Run Headless to generate model output lanes.';
  const maxRow = Math.max(1, frameCount - 1, ...results.map(row => Number(row.row_index || 0)));
  const labels = ['Truth', ...LABELS];
  const root = document.getElementById('timeline');
  root.innerHTML = labels.map(label => `<div class="lane"><div class="label">${label}</div><div class="track" data-lane="${label}"><div class="playhead"></div></div></div>`).join('');

  for (const event of inputEvents) {
    const labels = (event.labels || 'No Scent').split('|');
    const left = Number(event.row_start || 0) / maxRow * 100;
    const width = Math.max(0.4, (Number(event.row_end || 0) - Number(event.row_start || 0)) / maxRow * 100);
    const track = root.querySelector('[data-lane="Truth"]');
    const bar = document.createElement('div');
    bar.className = 'bar truth';
    bar.style.left = `${left}%`;
    bar.style.width = `${width}%`;
    bar.style.background = labels[0] && colors[labels[0]] ? colors[labels[0]] : '#cfd7dc';
    bar.title = labels.join(' + ');
    track.appendChild(bar);
  }

  for (const label of LABELS) {
    const track = root.querySelector(`[data-lane="${cssEscape(label)}"]`);
    let start = null;
    let last = null;
    for (const row of results) {
      const active = [row.pred_1, row.pred_2, row.pred_3].includes(label) && row.silent === 'false';
      const rowIndex = Number(row.row_index || 0);
      if (active && start === null) start = rowIndex;
      if (!active && start !== null) {
        appendPred(track, label, start, last ?? rowIndex, maxRow);
        start = null;
      }
      if (active) last = rowIndex;
    }
    if (start !== null) appendPred(track, label, start, last ?? start, maxRow);
  }
  updatePlayhead();
}

function appendPred(track, label, start, end, maxRow) {
  const bar = document.createElement('div');
  bar.className = 'bar pred';
  bar.style.left = `${start / maxRow * 100}%`;
  bar.style.width = `${Math.max(0.4, (end - start + 1) / maxRow * 100)}%`;
  bar.style.background = colors[label] || '#208c86';
  bar.title = label;
  track.appendChild(bar);
}

function cssEscape(value) {
  return value.replace(/"/g, '\\"');
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map(line => {
    const fields = splitCsvLine(line);
    const row = {};
    headers.forEach((header, index) => row[header] = fields[index] ?? '');
    return row;
  });
}

function togglePlay() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    document.getElementById('play').textContent = 'Play';
    return;
  }
  if (!currentFrameCount) return;
  document.getElementById('play').textContent = 'Pause';
  playTimer = setInterval(() => {
    currentFrame += Math.max(1, Math.round(currentFrameCount / 500));
    if (currentFrame >= currentFrameCount) currentFrame = 0;
    updatePlayhead();
  }, 33);
}

function updatePlayhead() {
  const pct = currentFrameCount <= 1 ? 0 : currentFrame / (currentFrameCount - 1) * 100;
  document.querySelectorAll('.playhead').forEach(head => {
    head.style.left = `${pct}%`;
  });
}

function splitCsvLine(line) {
  const fields = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"' && quoted && line[i + 1] === '"') {
      field += '"';
      i++;
    } else if (ch === '"') {
      quoted = !quoted;
    } else if (ch === ',' && !quoted) {
      fields.push(field);
      field = '';
    } else {
      field += ch;
    }
  }
  fields.push(field);
  return fields;
}

document.getElementById('preset').onchange = applyPreset;
document.getElementById('add').onclick = () => addSegment();
document.getElementById('add-gap').onclick = () => addSegment([]);
document.getElementById('clear').onclick = () => { state.sequence = []; renderSequence(); };
document.getElementById('save').onclick = async () => {
  await saveState();
  renderTimeline([], [], 0, 'Unsaved');
};
document.getElementById('materialize').onclick = () => postAction('/api/materialize', false);
document.getElementById('run').onclick = () => {
  const path = selectedModelKind() === 'grid8' ? '/api/run-grid8' : '/api/run-headless';
  postAction(path);
};
document.getElementById('train-grid8').onclick = () => postAction('/api/train-grid8', false);
document.getElementById('play').onclick = togglePlay;

initNotes();
loadState().then(loadInputPreview);
</script>
</body>
</html>
"##;

fn app_html() -> String {
    APP_HTML.replace("__LABELS__", LABELS_JSON)
}
