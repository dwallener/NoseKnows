use noseknows::csv::csv_escape;
use noseknows::grid::{load_model, GridRuntime, FEATURES};
use noseknows::peak::{
    expected_names, is_no_scent_target, predicted_labels, read_live_frames, top_k, CHANNELS,
    LABELS, OUTPUTS,
};
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_INPUT: &str = "data/live/input_frames.csv";
const DEFAULT_MODEL: &str = "data/models/grid8_readout.ngm";
const DEFAULT_RESULTS: &str = "data/live/grid_model_results.csv";
const DEFAULT_EVENTS: &str = "data/live/grid_events.csv";

struct Config {
    input_path: PathBuf,
    model_path: PathBuf,
    results_path: PathBuf,
    events_path: PathBuf,
    gate_threshold: f32,
    run_id: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("grid_live_headless error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let model = load_model(&config.model_path)?;
    let frames = read_live_frames(&config.input_path)?;
    if frames.is_empty() {
        return Err(format!("{} has no usable frames", config.input_path.display()).into());
    }
    let mut runtime = GridRuntime::new(model);

    write_results_header(&config)?;
    write_events_header(&config)?;
    let mut results_file = append_file(&config.results_path)?;
    let mut events_file = append_file(&config.events_path)?;
    let mut previous_predicted = Vec::<usize>::new();
    let mut emitted = 0_usize;
    let mut silent_no_scent = 0_usize;
    let mut false_positive = 0_usize;

    for frame in frames {
        let frame_clone = frame.clone();
        let (grid, logits) = runtime.step(frame.elapsed_ms, &frame.adc);
        let predicted = predicted_labels(&logits, config.gate_threshold);
        if !predicted.is_empty() {
            emitted += 1;
        }
        if is_no_scent_target(&frame_clone.target) {
            if predicted.is_empty() {
                silent_no_scent += 1;
            } else {
                false_positive += 1;
            }
        }
        write_result_row(
            &mut results_file,
            &config,
            &frame_clone,
            &grid,
            &logits,
            &predicted,
        )?;
        if predicted != previous_predicted {
            write_event_row(&mut events_file, &config, &frame_clone, &predicted)?;
            previous_predicted = predicted;
        }
    }

    println!(
        "Grid headless complete: emitted={} no_scent_silent={} false_positive={}",
        emitted, silent_no_scent, false_positive
    );
    println!("Grid results: {}", config.results_path.display());
    println!("Grid events:  {}", config.events_path.display());
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut input_path = PathBuf::from(DEFAULT_INPUT);
    let mut model_path = PathBuf::from(DEFAULT_MODEL);
    let mut results_path = PathBuf::from(DEFAULT_RESULTS);
    let mut events_path = PathBuf::from(DEFAULT_EVENTS);
    let mut gate_threshold = 0.0;
    let mut run_id = default_run_id();

    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--input" => {
                index += 1;
                input_path = PathBuf::from(args.get(index).ok_or("--input requires a path")?);
            }
            "--model" => {
                index += 1;
                model_path = PathBuf::from(args.get(index).ok_or("--model requires a path")?);
            }
            "--out-results" => {
                index += 1;
                results_path =
                    PathBuf::from(args.get(index).ok_or("--out-results requires a path")?);
            }
            "--out-events" => {
                index += 1;
                events_path = PathBuf::from(args.get(index).ok_or("--out-events requires a path")?);
            }
            "--gate-threshold" => {
                index += 1;
                gate_threshold = args
                    .get(index)
                    .ok_or("--gate-threshold requires a value")?
                    .parse()?;
            }
            "--run-id" => {
                index += 1;
                run_id = args.get(index).ok_or("--run-id requires a value")?.clone();
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin grid_live_headless -- [--input data/live/input_frames.csv] [--model data/models/grid8_readout.ngm] [--out-results data/live/grid_model_results.csv] [--out-events data/live/grid_events.csv] [--gate-threshold 0] [--run-id name]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        index += 1;
    }

    Ok(Config {
        input_path,
        model_path,
        results_path,
        events_path,
        gate_threshold,
        run_id,
    })
}

fn write_results_header(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = config.results_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(&config.results_path)?;
    writeln!(
        file,
        "run_id,row_index,elapsed_ms,stream_segment,source_sample_id,label_1,label_2,label_3,truth_labels,silent,pred_1,pred_2,pred_3,score_1,score_2,score_3,bins,held_peaks,adc_values"
    )?;
    Ok(())
}

fn write_events_header(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = config.events_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(&config.events_path)?;
    writeln!(
        file,
        "run_id,row_index,elapsed_ms,stream_segment,event_type,predicted_labels,truth_labels"
    )?;
    Ok(())
}

fn append_file(path: &PathBuf) -> Result<fs::File, Box<dyn std::error::Error>> {
    Ok(fs::OpenOptions::new().append(true).open(path)?)
}

fn write_result_row(
    file: &mut fs::File,
    config: &Config,
    frame: &noseknows::peak::LiveFrame,
    grid: &[u8; FEATURES],
    logits: &[f32; OUTPUTS],
    predicted: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let top = top_k(logits, 3);
    let grid_text = compact_grid(grid);
    let adc_values = frame
        .adc
        .iter()
        .take(CHANNELS)
        .map(|value| format!("{value:.1}"))
        .collect::<Vec<_>>()
        .join("|");
    let truth = expected_names(&frame.target).join("|");
    let silent = predicted.is_empty();
    writeln!(
        file,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{},{},{}",
        csv_escape(&config.run_id),
        frame.row_index,
        frame.elapsed_ms,
        csv_escape(&frame.segment),
        csv_escape(&frame.source_sample_id),
        csv_escape(&frame.labels[0]),
        csv_escape(&frame.labels[1]),
        csv_escape(&frame.labels[2]),
        csv_escape(&truth),
        silent,
        csv_escape(LABELS[top[0].0]),
        csv_escape(LABELS[top[1].0]),
        csv_escape(LABELS[top[2].0]),
        top[0].1,
        top[1].1,
        top[2].1,
        csv_escape(&grid_text),
        csv_escape("grid8"),
        csv_escape(&adc_values),
    )?;
    Ok(())
}

fn write_event_row(
    file: &mut fs::File,
    config: &Config,
    frame: &noseknows::peak::LiveFrame,
    predicted: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let predicted_labels = if predicted.is_empty() {
        "silent".to_string()
    } else {
        predicted
            .iter()
            .map(|label| LABELS[*label])
            .collect::<Vec<_>>()
            .join("|")
    };
    let truth = expected_names(&frame.target).join("|");
    writeln!(
        file,
        "{},{},{},{},{},{},{}",
        csv_escape(&config.run_id),
        frame.row_index,
        frame.elapsed_ms,
        csv_escape(&frame.segment),
        "readout_change",
        csv_escape(&predicted_labels),
        csv_escape(&truth),
    )?;
    Ok(())
}

fn compact_grid(grid: &[u8; FEATURES]) -> String {
    (0..8)
        .map(|sensor| {
            grid[sensor * 8..sensor * 8 + 8]
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join("")
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn default_run_id() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format!("grid_live_headless_{seconds}")
}
