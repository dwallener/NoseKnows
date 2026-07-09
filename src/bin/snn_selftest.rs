use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const CHANNELS: usize = 9;
const ACTIVE_SENSORS: usize = 8;
const SNN_INPUTS: usize = ACTIVE_SENSORS * 2;
const PATTERN_NEURONS: usize = 64;
const PATTERN_WINNERS_PER_STEP: usize = 3;
const SNN_OUTPUTS: usize = 14;
const DEFAULT_DATA: &str = "data/raw_single_note_probe";
const DEFAULT_MODEL: &str = "data/models/snn_accordion_single_note_probe.nsm";
const DEFAULT_BINS: usize = 180;
const DEFAULT_SUBSLOTS: usize = 5;
const DEFAULT_RATE_BUDGET: usize = 5;
const DEFAULT_LATENCY_BUDGET: usize = 5;
const DEFAULT_GATE_MIN_TOP: usize = 3;
const DEFAULT_GATE_MARGIN: isize = 1;
const DEFAULT_GATE_MIN_ACTIVITY: usize = 12;
const DEFAULT_GATE_WINDOW_SAMPLES: usize = 6;
const DEFAULT_MIN_CORRECT_DECISIONS: usize = 3;
const DEFAULT_MAX_SPILLOVER_LABELS: usize = 2;
const DEFAULT_MAX_SPILLOVER_DECISIONS: usize = 3;
const DEFAULT_DISPLAY_MAX_DOMINANT_GAP: usize = 8;
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
const NO_SCENT_LABEL: &str = "No Scent";
const SILENT_BUCKET: usize = SNN_OUTPUTS;
const CONFUSION_BUCKETS: usize = SNN_OUTPUTS + 1;
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

struct Config {
    data_dir: PathBuf,
    model: PathBuf,
    bins: usize,
    subslots: usize,
    rate_budget: usize,
    latency_budget: usize,
    gate_min_top: usize,
    gate_margin: isize,
    gate_min_activity: usize,
    gate_window_samples: usize,
    min_correct_decisions: usize,
    max_spillover_labels: usize,
    max_spillover_decisions: usize,
    display_max_dominant_gap: usize,
    rubric: Rubric,
    verbose: bool,
}

#[derive(Clone)]
struct Sample {
    id: String,
    labels: [String; 3],
    rows: Vec<[f32; CHANNELS]>,
}

struct EncodedSample {
    bins: usize,
    input_events: Vec<InputEvent>,
    input_masks: Vec<u16>,
}

struct InputEvent {
    sample_index: usize,
    subslot: usize,
}

struct PatternSpike {
    sample_index: usize,
    subslot: usize,
}

struct OutputSpike {
    sample_index: usize,
    subslot: usize,
    label: usize,
}

struct GatedDecision {
    label: usize,
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
    threshold: i32,
    decay_alpha_q8: i32,
}

#[derive(Default)]
struct Summary {
    checked: usize,
    skipped: usize,
    passed: usize,
    failed: usize,
    no_scent_checked: usize,
    no_scent_passed: usize,
    single_checked: usize,
    single_passed: usize,
}

#[derive(Default)]
struct FailureBreakdown {
    raw_silent: usize,
    gate_silent: usize,
    wrong_dominant: usize,
    spillover: usize,
    no_scent_false_positive: usize,
}

#[derive(Default)]
struct Confusion {
    counts: [[usize; CONFUSION_BUCKETS]; SNN_OUTPUTS],
    pass: [usize; SNN_OUTPUTS],
    total: [usize; SNN_OUTPUTS],
}

enum Expected {
    NoScent,
    Single { label: usize },
    Skip,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Rubric {
    Strict,
    Display,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("snn_selftest error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let model = load_snn_model(&config.model)?;
    let mut paths = fs::read_dir(&config.data_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("csv"))
        .collect::<Vec<_>>();
    paths.sort();

    let mut summary = Summary::default();
    let mut breakdown = FailureBreakdown::default();
    let mut confusion = Confusion::default();
    let mut failures = Vec::new();

    for path in paths {
        let sample = load_sample(&path)?;
        let expected = expected_from_labels(&sample.labels);
        if matches!(expected, Expected::Skip) {
            summary.skipped += 1;
            continue;
        }

        let encoded = encode_sample(&sample, &config);
        let (output_spikes, pattern_spikes) = match &model {
            SnnModel::Direct(model) => (
                model.forward_spikes(&encoded.input_masks, config.subslots),
                None,
            ),
            SnnModel::Accordion(model) => {
                let pattern_masks = model.forward_pattern_masks(&encoded.input_masks);
                let pattern_spikes = pattern_spikes_from_masks(&pattern_masks, config.subslots);
                let output_spikes = model.forward_label_spikes(&pattern_masks, config.subslots);
                (output_spikes, Some(pattern_spikes))
            }
        };
        let gated = gated_decisions(
            &output_spikes,
            pattern_spikes.as_deref(),
            Some(&encoded.input_events),
            encoded.bins,
            config.subslots,
            &config,
        );
        let raw_counts = output_counts(&output_spikes);
        let gated_counts = decision_counts(&gated);
        let verdict = evaluate_sample(&expected, &raw_counts, &gated_counts, &config);
        if let Expected::Single { label } = expected {
            confusion.record(label, top_label(&gated_counts), verdict.passed);
        }

        summary.checked += 1;
        if matches!(expected, Expected::NoScent) {
            summary.no_scent_checked += 1;
        } else {
            summary.single_checked += 1;
        }

        if verdict.passed {
            summary.passed += 1;
            if matches!(expected, Expected::NoScent) {
                summary.no_scent_passed += 1;
            } else {
                summary.single_passed += 1;
            }
        } else {
            summary.failed += 1;
            breakdown.record(verdict.kind);
            failures.push(format!("{}: {}", sample.id, verdict.reason));
        }

        if config.verbose || !verdict.passed {
            println!(
                "{} {:>4} expected={} raw_top={} raw_counts={} gated_top={} gated_counts={} kind={}",
                if verdict.passed { "PASS" } else { "FAIL" },
                sample.id,
                expected_name(&expected),
                format_top(&raw_counts),
                format_counts(&raw_counts),
                format_top(&gated_counts),
                format_counts(&gated_counts),
                verdict.kind.name()
            );
        }
    }

    println!(
        "Self-test checked={} passed={} failed={} skipped={}",
        summary.checked, summary.passed, summary.failed, summary.skipped
    );
    println!(
        "No Scent: {}/{} silent | Single note: {}/{} pass",
        summary.no_scent_passed,
        summary.no_scent_checked,
        summary.single_passed,
        summary.single_checked
    );
    print_failure_breakdown(&breakdown);
    print_confusion(&confusion);

    if !failures.is_empty() {
        println!();
        println!("Failures:");
        for failure in failures.iter().take(20) {
            println!("  {failure}");
        }
        if failures.len() > 20 {
            println!("  ... {} more", failures.len() - 20);
        }
        return Err("SNN self-test failed".into());
    }

    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut data_dir = PathBuf::from(DEFAULT_DATA);
    let mut model = PathBuf::from(DEFAULT_MODEL);
    let mut bins = DEFAULT_BINS;
    let mut subslots = DEFAULT_SUBSLOTS;
    let mut rate_budget = DEFAULT_RATE_BUDGET;
    let mut latency_budget = DEFAULT_LATENCY_BUDGET;
    let mut gate_min_top = DEFAULT_GATE_MIN_TOP;
    let mut gate_margin = DEFAULT_GATE_MARGIN;
    let mut gate_min_activity = DEFAULT_GATE_MIN_ACTIVITY;
    let mut gate_window_samples = DEFAULT_GATE_WINDOW_SAMPLES;
    let mut min_correct_decisions = DEFAULT_MIN_CORRECT_DECISIONS;
    let mut max_spillover_labels = DEFAULT_MAX_SPILLOVER_LABELS;
    let mut max_spillover_decisions = DEFAULT_MAX_SPILLOVER_DECISIONS;
    let mut display_max_dominant_gap = DEFAULT_DISPLAY_MAX_DOMINANT_GAP;
    let mut rubric = Rubric::Strict;
    let mut verbose = false;

    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--data" => {
                index += 1;
                data_dir = PathBuf::from(args.get(index).ok_or("--data requires a path")?);
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
            "--min-correct" => {
                index += 1;
                min_correct_decisions = args
                    .get(index)
                    .ok_or("--min-correct requires a value")?
                    .parse()?;
            }
            "--max-spillover-labels" => {
                index += 1;
                max_spillover_labels = args
                    .get(index)
                    .ok_or("--max-spillover-labels requires a value")?
                    .parse()?;
            }
            "--max-spillover" => {
                index += 1;
                max_spillover_decisions = args
                    .get(index)
                    .ok_or("--max-spillover requires a value")?
                    .parse()?;
            }
            "--display-max-dominant-gap" => {
                index += 1;
                display_max_dominant_gap = args
                    .get(index)
                    .ok_or("--display-max-dominant-gap requires a value")?
                    .parse()?;
            }
            "--rubric" => {
                index += 1;
                rubric = parse_rubric(args.get(index).ok_or("--rubric requires a value")?)?;
            }
            "--verbose" => verbose = true,
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin snn_selftest -- [--data data/raw_single_note_probe] [--model data/models/snn_accordion_single_note_probe.nsm] [--rubric strict|display] [--gate-window 6] [--min-correct 3] [--max-spillover-labels 2] [--max-spillover 3] [--display-max-dominant-gap 8] [--verbose]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        index += 1;
    }

    Ok(Config {
        data_dir,
        model,
        bins,
        subslots: subslots.max(1),
        rate_budget,
        latency_budget,
        gate_min_top,
        gate_margin,
        gate_min_activity,
        gate_window_samples: gate_window_samples.max(1),
        min_correct_decisions,
        max_spillover_labels,
        max_spillover_decisions,
        display_max_dominant_gap,
        rubric,
        verbose,
    })
}

fn parse_rubric(value: &str) -> Result<Rubric, Box<dyn std::error::Error>> {
    match value {
        "strict" => Ok(Rubric::Strict),
        "display" => Ok(Rubric::Display),
        other => Err(format!("unknown rubric {other}; expected strict or display").into()),
    }
}

fn expected_from_labels(labels: &[String; 3]) -> Expected {
    let real = labels
        .iter()
        .filter_map(|label| label_index(label))
        .collect::<Vec<_>>();
    if real.is_empty()
        && labels
            .iter()
            .all(|label| label.eq_ignore_ascii_case(NO_SCENT_LABEL))
    {
        return Expected::NoScent;
    }
    if real.len() == 1
        && labels
            .iter()
            .filter(|label| label.eq_ignore_ascii_case(NO_SCENT_LABEL))
            .count()
            == 2
    {
        return Expected::Single { label: real[0] };
    }
    Expected::Skip
}

struct Verdict {
    passed: bool,
    kind: FailureKind,
    reason: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FailureKind {
    Pass,
    RawSilent,
    GateSilent,
    WrongDominant,
    Spillover,
    NoScentFalsePositive,
}

impl FailureKind {
    fn name(self) -> &'static str {
        match self {
            FailureKind::Pass => "pass",
            FailureKind::RawSilent => "raw_silent",
            FailureKind::GateSilent => "gate_silent",
            FailureKind::WrongDominant => "wrong_dominant",
            FailureKind::Spillover => "spillover",
            FailureKind::NoScentFalsePositive => "no_scent_false_positive",
        }
    }
}

fn evaluate_sample(
    expected: &Expected,
    raw_counts: &[usize; SNN_OUTPUTS],
    gated_counts: &[usize; SNN_OUTPUTS],
    config: &Config,
) -> Verdict {
    match expected {
        Expected::NoScent => {
            let total: usize = gated_counts.iter().sum();
            if total == 0 {
                Verdict {
                    passed: true,
                    kind: FailureKind::Pass,
                    reason: "silent".to_string(),
                }
            } else {
                Verdict {
                    passed: false,
                    kind: FailureKind::NoScentFalsePositive,
                    reason: format!(
                        "expected silence, got gated={}",
                        format_counts(gated_counts)
                    ),
                }
            }
        }
        Expected::Single { label } => {
            if config.rubric == Rubric::Display {
                return evaluate_single_display(*label, raw_counts, gated_counts, config);
            }
            evaluate_single_strict(*label, raw_counts, gated_counts, config)
        }
        Expected::Skip => Verdict {
            passed: true,
            kind: FailureKind::Pass,
            reason: "skipped".to_string(),
        },
    }
}

fn evaluate_single_strict(
    label: usize,
    raw_counts: &[usize; SNN_OUTPUTS],
    gated_counts: &[usize; SNN_OUTPUTS],
    config: &Config,
) -> Verdict {
    let raw_total: usize = raw_counts.iter().sum();
    let gated_total: usize = gated_counts.iter().sum();
    let correct = gated_counts[label];
    let top = top_label(gated_counts);
    let spillovers = gated_counts
        .iter()
        .enumerate()
        .filter(|(index, count)| *index != label && **count > 0)
        .collect::<Vec<_>>();
    let over_limit = spillovers
        .iter()
        .filter(|(_, count)| **count > config.max_spillover_decisions)
        .count();

    if raw_total == 0 {
        return Verdict {
            passed: false,
            kind: FailureKind::RawSilent,
            reason: format!("raw output silent for {}", LABELS[label]),
        };
    }
    if gated_total == 0 {
        return Verdict {
            passed: false,
            kind: FailureKind::GateSilent,
            reason: format!(
                "gate silent for {}; raw={}",
                LABELS[label],
                format_counts(raw_counts)
            ),
        };
    }
    if correct < config.min_correct_decisions {
        return Verdict {
            passed: false,
            kind: FailureKind::GateSilent,
            reason: format!(
                "correct {} too weak in gate: {correct} < {}; raw={} gated={}",
                LABELS[label],
                config.min_correct_decisions,
                format_counts(raw_counts),
                format_counts(gated_counts)
            ),
        };
    }
    if top != Some(label) {
        return Verdict {
            passed: false,
            kind: FailureKind::WrongDominant,
            reason: format!(
                "correct {} not dominant; raw={} gated={}",
                LABELS[label],
                format_counts(raw_counts),
                format_counts(gated_counts)
            ),
        };
    }
    if spillovers.len() > config.max_spillover_labels {
        return Verdict {
            passed: false,
            kind: FailureKind::Spillover,
            reason: format!(
                "too many spillover labels: {} > {}; gated={}",
                spillovers.len(),
                config.max_spillover_labels,
                format_counts(gated_counts)
            ),
        };
    }
    if over_limit > 0 {
        return Verdict {
            passed: false,
            kind: FailureKind::Spillover,
            reason: format!(
                "spillover too persistent; max allowed per label {} gated={}",
                config.max_spillover_decisions,
                format_counts(gated_counts)
            ),
        };
    }
    Verdict {
        passed: true,
        kind: FailureKind::Pass,
        reason: "dominant correct label with bounded spillover".to_string(),
    }
}

fn evaluate_single_display(
    label: usize,
    raw_counts: &[usize; SNN_OUTPUTS],
    gated_counts: &[usize; SNN_OUTPUTS],
    config: &Config,
) -> Verdict {
    let raw_total: usize = raw_counts.iter().sum();
    let gated_total: usize = gated_counts.iter().sum();
    let correct = gated_counts[label];
    let ranked = ranked_labels(gated_counts);
    let top_count = ranked.first().map(|(_, count)| *count).unwrap_or(0);
    let correct_rank = ranked
        .iter()
        .position(|(candidate, count)| *candidate == label && *count > 0);

    if raw_total == 0 {
        return Verdict {
            passed: false,
            kind: FailureKind::RawSilent,
            reason: format!("raw output silent for {}", LABELS[label]),
        };
    }
    if gated_total == 0 {
        return Verdict {
            passed: false,
            kind: FailureKind::GateSilent,
            reason: format!(
                "gate silent for {}; raw={}",
                LABELS[label],
                format_counts(raw_counts)
            ),
        };
    }
    if correct < config.min_correct_decisions {
        return Verdict {
            passed: false,
            kind: FailureKind::GateSilent,
            reason: format!(
                "correct {} too weak for display: {correct} < {}; raw={} gated={}",
                LABELS[label],
                config.min_correct_decisions,
                format_counts(raw_counts),
                format_counts(gated_counts)
            ),
        };
    }
    if correct_rank.is_none_or(|rank| rank >= 3) {
        return Verdict {
            passed: false,
            kind: FailureKind::WrongDominant,
            reason: format!(
                "correct {} not in display top 3; raw={} gated={}",
                LABELS[label],
                format_counts(raw_counts),
                format_counts(gated_counts)
            ),
        };
    }
    if top_count.saturating_sub(correct) > config.display_max_dominant_gap {
        return Verdict {
            passed: false,
            kind: FailureKind::WrongDominant,
            reason: format!(
                "wrong note dominates display too strongly; max gap {} gated={}",
                config.display_max_dominant_gap,
                format_counts(gated_counts)
            ),
        };
    }
    Verdict {
        passed: true,
        kind: FailureKind::Pass,
        reason: "correct label display-visible with bounded wrong dominance".to_string(),
    }
}

impl FailureBreakdown {
    fn record(&mut self, kind: FailureKind) {
        match kind {
            FailureKind::Pass => {}
            FailureKind::RawSilent => self.raw_silent += 1,
            FailureKind::GateSilent => self.gate_silent += 1,
            FailureKind::WrongDominant => self.wrong_dominant += 1,
            FailureKind::Spillover => self.spillover += 1,
            FailureKind::NoScentFalsePositive => self.no_scent_false_positive += 1,
        }
    }
}

fn print_failure_breakdown(breakdown: &FailureBreakdown) {
    let total = breakdown.raw_silent
        + breakdown.gate_silent
        + breakdown.wrong_dominant
        + breakdown.spillover
        + breakdown.no_scent_false_positive;
    if total == 0 {
        return;
    }
    println!(
        "Failure kinds: raw_silent={} gate_silent={} wrong_dominant={} spillover={} no_scent_fp={}",
        breakdown.raw_silent,
        breakdown.gate_silent,
        breakdown.wrong_dominant,
        breakdown.spillover,
        breakdown.no_scent_false_positive
    );
}

impl Confusion {
    fn record(&mut self, expected: usize, predicted: Option<usize>, passed: bool) {
        let predicted = predicted.unwrap_or(SILENT_BUCKET);
        self.counts[expected][predicted] += 1;
        self.total[expected] += 1;
        if passed {
            self.pass[expected] += 1;
        }
    }
}

fn print_confusion(confusion: &Confusion) {
    if confusion.total.iter().all(|count| *count == 0) {
        return;
    }

    println!();
    println!("Single-note confusion by dominant gated label:");
    println!(
        "{:<16} {:>7} {:>7}  {}",
        "expected", "pass", "total", "dominant predictions"
    );
    for expected in 0..SNN_OUTPUTS {
        if confusion.total[expected] == 0 {
            continue;
        }
        let predictions = format_confusion_row(&confusion.counts[expected]);
        println!(
            "{:<16} {:>7} {:>7}  {}",
            LABELS[expected], confusion.pass[expected], confusion.total[expected], predictions
        );
    }

    let mut misses = Vec::new();
    for expected in 0..SNN_OUTPUTS {
        for predicted in 0..CONFUSION_BUCKETS {
            if predicted == expected || confusion.counts[expected][predicted] == 0 {
                continue;
            }
            misses.push((expected, predicted, confusion.counts[expected][predicted]));
        }
    }
    misses.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| LABELS[a.0].cmp(LABELS[b.0])));

    if !misses.is_empty() {
        println!();
        println!("Largest single-note confusions:");
        for (expected, predicted, count) in misses.into_iter().take(12) {
            println!(
                "  {:<16} -> {:<16} {}",
                LABELS[expected],
                prediction_name(predicted),
                count
            );
        }
    }
}

fn format_confusion_row(row: &[usize; CONFUSION_BUCKETS]) -> String {
    let mut predictions = row
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, count)| *count > 0)
        .collect::<Vec<_>>();
    predictions.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| prediction_name(a.0).cmp(prediction_name(b.0)))
    });
    predictions
        .into_iter()
        .map(|(predicted, count)| format!("{}={count}", prediction_name(predicted)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn prediction_name(index: usize) -> &'static str {
    if index == SILENT_BUCKET {
        "Silent"
    } else {
        LABELS[index]
    }
}

fn load_sample(path: &Path) -> Result<Sample, Box<dyn std::error::Error>> {
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

fn encode_sample(sample: &Sample, config: &Config) -> EncodedSample {
    let bins = config.bins.min(sample.rows.len()).max(8);
    let subslots = config.subslots.max(1);
    let binned = bin_rows(&sample.rows, bins);
    let mut events = Vec::new();
    let mut masks = vec![0_u16; binned.len() * subslots];

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
            for slot in 0..rate_spike_count(amplitude, config.rate_budget) {
                let subslot = rate_subslot(slot, config.rate_budget, subslots);
                let mask_index = bin_index * subslots + subslot;
                masks[mask_index] |= 1 << channel;
                events.push(InputEvent {
                    sample_index: bin_index,
                    subslot,
                });
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
                    events.push(InputEvent {
                        sample_index: bin_index,
                        subslot: event_subslot,
                    });
                }
            }
            previous = row[channel];
        }
    }

    EncodedSample {
        bins: binned.len(),
        input_events: events,
        input_masks: masks,
    }
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

fn gated_decisions(
    output_spikes: &[OutputSpike],
    pattern_spikes: Option<&[PatternSpike]>,
    input_spikes: Option<&[InputEvent]>,
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

        for (label, score) in ranked.into_iter().take(3) {
            if score >= label_gate_min_top(label, config) {
                decisions.push(GatedDecision { label });
            }
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
                });
            }
        }
    }
    spikes
}

fn load_snn_model(path: &Path) -> Result<SnnModel, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    if text.lines().next() == Some("NOSEKNOWS_SNN_ACCORDION_V1") {
        load_accordion_model(path, &text)
    } else {
        load_direct_model(path, &text)
    }
}

fn load_direct_model(path: &Path, text: &str) -> Result<SnnModel, Box<dyn std::error::Error>> {
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
            let label_index =
                label_index(label).ok_or_else(|| format!("unknown label: {label}"))?;
            bias[label_index] = value.parse()?;
        } else if let Some(rest) = line.strip_prefix("weights.") {
            let (label, values) = rest
                .split_once('=')
                .ok_or_else(|| format!("invalid weights line in {}", path.display()))?;
            let label_index =
                label_index(label).ok_or_else(|| format!("unknown label: {label}"))?;
            let parsed = parse_i16_list(values)?;
            if parsed.len() != SNN_INPUTS {
                return Err(format!("direct weights for {label} expected {SNN_INPUTS}").into());
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

fn load_accordion_model(path: &Path, text: &str) -> Result<SnnModel, Box<dyn std::error::Error>> {
    let mut pattern_weights = [[0_i16; SNN_INPUTS]; PATTERN_NEURONS];
    let mut label_weights = [[0_i16; PATTERN_NEURONS]; SNN_OUTPUTS];
    let mut label_bias = [0_i16; SNN_OUTPUTS];
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
            if field == "weights" {
                let pattern: usize = index_text.parse()?;
                if pattern >= PATTERN_NEURONS {
                    return Err(format!("pattern index {pattern} out of range").into());
                }
                let parsed = parse_i16_list(value)?;
                if parsed.len() != SNN_INPUTS {
                    return Err(format!("pattern {pattern} expected {SNN_INPUTS}").into());
                }
                pattern_weights[pattern].copy_from_slice(&parsed);
            }
        } else if let Some(rest) = line.strip_prefix("label_bias.") {
            let (label, value) = rest
                .split_once('=')
                .ok_or_else(|| format!("invalid label_bias line in {}", path.display()))?;
            let label_index =
                label_index(label).ok_or_else(|| format!("unknown label: {label}"))?;
            label_bias[label_index] = value.parse()?;
        } else if let Some(rest) = line.strip_prefix("label_weights.") {
            let (label, values) = rest
                .split_once('=')
                .ok_or_else(|| format!("invalid label_weights line in {}", path.display()))?;
            let label_index =
                label_index(label).ok_or_else(|| format!("unknown label: {label}"))?;
            let parsed = parse_i16_list(values)?;
            if parsed.len() != PATTERN_NEURONS {
                return Err(format!("label weights for {label} expected {PATTERN_NEURONS}").into());
            }
            label_weights[label_index].copy_from_slice(&parsed);
        }
    }

    Ok(SnnModel::Accordion(AccordionLifModel {
        pattern_weights,
        label_weights,
        label_bias,
        threshold,
        decay_alpha_q8,
    }))
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

fn pattern_uses_sensor(weights: &[i16; SNN_INPUTS], sensor: usize) -> bool {
    weights[sensor] > 0 || weights[ACTIVE_SENSORS + sensor] > 0
}

fn decision_counts(decisions: &[GatedDecision]) -> [usize; SNN_OUTPUTS] {
    let mut counts = [0_usize; SNN_OUTPUTS];
    for decision in decisions {
        counts[decision.label] += 1;
    }
    counts
}

fn output_counts(spikes: &[OutputSpike]) -> [usize; SNN_OUTPUTS] {
    let mut counts = [0_usize; SNN_OUTPUTS];
    for spike in spikes {
        counts[spike.label] += 1;
    }
    counts
}

fn top_label(counts: &[usize; SNN_OUTPUTS]) -> Option<usize> {
    ranked_labels(counts).first().map(|(label, _)| *label)
}

fn ranked_labels(counts: &[usize; SNN_OUTPUTS]) -> Vec<(usize, usize)> {
    let mut ranked = counts
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, count)| *count > 0)
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| LABELS[a.0].cmp(LABELS[b.0])));
    ranked
}

fn format_top(counts: &[usize; SNN_OUTPUTS]) -> &'static str {
    top_label(counts)
        .map(|label| LABELS[label])
        .unwrap_or("Silent")
}

fn expected_name(expected: &Expected) -> String {
    match expected {
        Expected::NoScent => NO_SCENT_LABEL.to_string(),
        Expected::Single { label } => LABELS[*label].to_string(),
        Expected::Skip => "skip".to_string(),
    }
}

fn format_counts(counts: &[usize; SNN_OUTPUTS]) -> String {
    let mut nonzero = counts
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, count)| *count > 0)
        .collect::<Vec<_>>();
    nonzero.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| LABELS[a.0].cmp(LABELS[b.0])));
    if nonzero.is_empty() {
        return "silent".to_string();
    }
    nonzero
        .into_iter()
        .map(|(label, count)| format!("{}={count}", LABELS[label]))
        .collect::<Vec<_>>()
        .join(",")
}

fn label_index(label: &str) -> Option<usize> {
    LABELS
        .iter()
        .position(|candidate| candidate.eq_ignore_ascii_case(label))
}

fn parse_i16_list(text: &str) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
    Ok(text
        .split(',')
        .map(str::parse::<i16>)
        .collect::<Result<Vec<_>, _>>()?)
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
    fn no_scent_requires_silence() {
        let config = test_config();
        let counts = [0; SNN_OUTPUTS];
        let verdict = evaluate_sample(&Expected::NoScent, &counts, &counts, &config);
        assert!(verdict.passed);
    }

    #[test]
    fn single_note_allows_bounded_spillover() {
        let config = test_config();
        let citrus = label_index("Citrus").expect("citrus");
        let fruity = label_index("Fruity").expect("fruity");
        let mut counts = [0; SNN_OUTPUTS];
        counts[citrus] = 12;
        counts[fruity] = 3;

        let verdict = evaluate_sample(
            &Expected::Single { label: citrus },
            &counts,
            &counts,
            &config,
        );

        assert!(verdict.passed);
    }

    #[test]
    fn single_note_rejects_wrong_dominant_label() {
        let config = test_config();
        let citrus = label_index("Citrus").expect("citrus");
        let fruity = label_index("Fruity").expect("fruity");
        let mut counts = [0; SNN_OUTPUTS];
        counts[citrus] = 4;
        counts[fruity] = 5;

        let verdict = evaluate_sample(
            &Expected::Single { label: citrus },
            &counts,
            &counts,
            &config,
        );

        assert!(!verdict.passed);
        assert_eq!(verdict.kind, FailureKind::WrongDominant);
    }

    #[test]
    fn display_rubric_allows_close_wrong_dominant_label() {
        let mut config = test_config();
        config.rubric = Rubric::Display;
        let green = label_index("Green").expect("green");
        let floral = label_index("Floral").expect("floral");
        let mut counts = [0; SNN_OUTPUTS];
        counts[floral] = 13;
        counts[green] = 12;

        let verdict = evaluate_sample(
            &Expected::Single { label: green },
            &counts,
            &counts,
            &config,
        );

        assert!(verdict.passed);
    }

    #[test]
    fn display_rubric_rejects_large_wrong_dominance() {
        let mut config = test_config();
        config.rubric = Rubric::Display;
        let fruity = label_index("Fruity").expect("fruity");
        let woods = label_index("Woods").expect("woods");
        let mut counts = [0; SNN_OUTPUTS];
        counts[woods] = 18;
        counts[fruity] = 7;

        let verdict = evaluate_sample(
            &Expected::Single { label: fruity },
            &counts,
            &counts,
            &config,
        );

        assert!(!verdict.passed);
        assert_eq!(verdict.kind, FailureKind::WrongDominant);
    }

    #[test]
    fn single_note_distinguishes_raw_from_gate_silence() {
        let config = test_config();
        let citrus = label_index("Citrus").expect("citrus");
        let mut raw_counts = [0; SNN_OUTPUTS];
        let gated_counts = [0; SNN_OUTPUTS];
        raw_counts[citrus] = 2;

        let verdict = evaluate_sample(
            &Expected::Single { label: citrus },
            &raw_counts,
            &gated_counts,
            &config,
        );

        assert!(!verdict.passed);
        assert_eq!(verdict.kind, FailureKind::GateSilent);
    }

    #[test]
    fn confusion_records_silent_predictions() {
        let citrus = label_index("Citrus").expect("citrus");
        let mut confusion = Confusion::default();

        confusion.record(citrus, None, false);

        assert_eq!(confusion.counts[citrus][SILENT_BUCKET], 1);
        assert_eq!(confusion.total[citrus], 1);
        assert_eq!(confusion.pass[citrus], 0);
    }

    fn test_config() -> Config {
        Config {
            data_dir: PathBuf::from(DEFAULT_DATA),
            model: PathBuf::from(DEFAULT_MODEL),
            bins: DEFAULT_BINS,
            subslots: DEFAULT_SUBSLOTS,
            rate_budget: DEFAULT_RATE_BUDGET,
            latency_budget: DEFAULT_LATENCY_BUDGET,
            gate_min_top: DEFAULT_GATE_MIN_TOP,
            gate_margin: DEFAULT_GATE_MARGIN,
            gate_min_activity: DEFAULT_GATE_MIN_ACTIVITY,
            gate_window_samples: DEFAULT_GATE_WINDOW_SAMPLES,
            min_correct_decisions: DEFAULT_MIN_CORRECT_DECISIONS,
            max_spillover_labels: DEFAULT_MAX_SPILLOVER_LABELS,
            max_spillover_decisions: DEFAULT_MAX_SPILLOVER_DECISIONS,
            display_max_dominant_gap: DEFAULT_DISPLAY_MAX_DOMINANT_GAP,
            rubric: Rubric::Strict,
            verbose: false,
        }
    }
}
