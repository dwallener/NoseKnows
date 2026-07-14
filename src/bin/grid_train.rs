use noseknows::grid::{
    is_one_note_or_no_scent, load_samples, normalize_bins, sample_second_grid_windows, save_model,
    GridModel, FEATURES,
};
use noseknows::peak::{top_k, LABELS, OUTPUTS};
use std::env;
use std::path::PathBuf;

const DEFAULT_DATA: &str = "data/views/peak_single_note";
const DEFAULT_OUT: &str = "data/models/grid8_readout.ngm";

struct Config {
    data_dir: PathBuf,
    output_path: PathBuf,
    epochs: usize,
    learning_rate: f32,
    validation_fraction: f32,
    seed: u64,
    lookback_secs: usize,
}

struct EncodedSample {
    id: String,
    labels: [String; 3],
    target: [bool; OUTPUTS],
    second_index: usize,
    bins: [u8; FEATURES],
    features: [f32; FEATURES],
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    active: usize,
    no_scent: usize,
    p_at_1: usize,
    any_at_3: usize,
    silent_active: usize,
    silent_no_scent: usize,
    false_positive: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("grid_train error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let samples = load_samples(&config.data_dir)?
        .into_iter()
        .filter(is_one_note_or_no_scent)
        .collect::<Vec<_>>();
    if samples.len() < 2 {
        return Err("grid training needs at least two no-scent/single-note samples".into());
    }

    let encoded = samples
        .iter()
        .flat_map(|sample| encode_rolling_windows(sample, config.lookback_secs))
        .collect::<Vec<_>>();
    let (train_indexes, validation_indexes) =
        split_indexes(encoded.len(), config.validation_fraction, config.seed);

    println!(
        "Loaded {} rolling grid window(s): {} train, {} validation",
        encoded.len(),
        train_indexes.len(),
        validation_indexes.len()
    );
    println!(
        "Grid readout: 8 active sensors x {} one-second lookback = {} features -> {} labels",
        config.lookback_secs, FEATURES, OUTPUTS
    );

    let mut model = GridModel::new(config.lookback_secs);
    let mut best_model = model.clone();
    let mut best_validation = Metrics::default();
    for epoch in 1..=config.epochs {
        for index in &train_indexes {
            train_one(&mut model, &encoded[*index], config.learning_rate);
        }
        if epoch == 1 || epoch == config.epochs || epoch % 25 == 0 {
            let train = evaluate(&model, &encoded, &train_indexes);
            let validation = evaluate(&model, &encoded, &validation_indexes);
            if validation.score() > best_validation.score() {
                best_validation = validation;
                best_model = model.clone();
            }
            println!(
                "grid epoch {epoch:>4} train p@1 {:.2}% any@3 {:.2}% active-silence {:.2}% no-scent silence {:.2}% fp {:.2}% | val p@1 {:.2}% any@3 {:.2}% active-silence {:.2}% no-scent silence {:.2}% fp {:.2}%",
                train.p_at_1_pct(),
                train.any_at_3_pct(),
                train.active_silence_pct(),
                train.no_scent_silence_pct(),
                train.false_positive_pct(),
                validation.p_at_1_pct(),
                validation.any_at_3_pct(),
                validation.active_silence_pct(),
                validation.no_scent_silence_pct(),
                validation.false_positive_pct()
            );
        }
    }

    println!(
        "best grid validation p@1 {:.2}% any@3 {:.2}% active-silence {:.2}% no-scent silence {:.2}% fp {:.2}%",
        best_validation.p_at_1_pct(),
        best_validation.any_at_3_pct(),
        best_validation.active_silence_pct(),
        best_validation.no_scent_silence_pct(),
        best_validation.false_positive_pct()
    );
    print_examples(&best_model, &encoded, &validation_indexes);
    save_model(&config.output_path, &best_model)?;
    println!("Saved grid model to {}", config.output_path.display());
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut data_dir = PathBuf::from(DEFAULT_DATA);
    let mut output_path = PathBuf::from(DEFAULT_OUT);
    let mut epochs = 250;
    let mut learning_rate = 0.18;
    let mut validation_fraction = 0.2;
    let mut seed = 0x8eed_2026_u64;
    let mut lookback_secs = 8;

    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--data" => {
                index += 1;
                data_dir = PathBuf::from(args.get(index).ok_or("--data requires a path")?);
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
                    .ok_or("--validation requires a value")?
                    .parse()?;
            }
            "--seed" => {
                index += 1;
                seed = args.get(index).ok_or("--seed requires a value")?.parse()?;
            }
            "--lookback-secs" => {
                index += 1;
                lookback_secs = args
                    .get(index)
                    .ok_or("--lookback-secs requires a value")?
                    .parse::<usize>()?
                    .clamp(1, 8);
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin grid_train -- [--data data/views/peak_single_note] [--out data/models/grid8_readout.ngm] [--epochs 250] [--lookback-secs 8]"
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
        data_dir,
        output_path,
        epochs,
        learning_rate,
        validation_fraction,
        seed,
        lookback_secs,
    })
}

fn encode_rolling_windows(
    sample: &noseknows::grid::RawSample,
    lookback_secs: usize,
) -> Vec<EncodedSample> {
    let windows = sample_second_grid_windows(sample, lookback_secs);
    let active_sample = sample.target.iter().any(|active| *active);
    let emit_start = lookback_secs.saturating_sub(1).max(2);
    windows
        .into_iter()
        .map(|(second_index, bins)| {
            let target = if active_sample && second_index >= emit_start {
                sample.target
            } else {
                [false; OUTPUTS]
            };
            EncodedSample {
                id: sample.id.clone(),
                labels: sample.labels.clone(),
                target,
                second_index,
                bins,
                features: normalize_bins(&bins),
            }
        })
        .collect()
}

fn train_one(model: &mut GridModel, sample: &EncodedSample, learning_rate: f32) {
    let logits = model.predict(&sample.features);
    let no_scent = is_no_scent_target(&sample.target);
    for label in 0..OUTPUTS {
        let target = if sample.target[label] { 1.0 } else { 0.0 };
        let weight = if sample.target[label] {
            4.0
        } else if no_scent {
            2.5
        } else {
            1.0
        };
        let error = target - sigmoid(logits[label]);
        let update = learning_rate * weight * error;
        model.bias[label] = (model.bias[label] + update * 0.02).clamp(-16.0, 16.0);
        for feature in 0..FEATURES {
            model.weights[label][feature] = (model.weights[label][feature]
                + update * sample.features[feature])
                .clamp(-32.0, 32.0);
        }
    }
}

fn evaluate(model: &GridModel, samples: &[EncodedSample], indexes: &[usize]) -> Metrics {
    let mut metrics = Metrics::default();
    for index in indexes {
        let sample = &samples[*index];
        let logits = model.predict(&sample.features);
        if is_no_scent_target(&sample.target) {
            metrics.no_scent += 1;
            if logits.iter().all(|value| *value <= 0.0) {
                metrics.silent_no_scent += 1;
            } else {
                metrics.false_positive += 1;
            }
            continue;
        }

        metrics.active += 1;
        if logits.iter().all(|value| *value <= 0.0) {
            metrics.silent_active += 1;
        }
        let top = top_k(&logits, 3);
        if sample.target[top[0].0] {
            metrics.p_at_1 += 1;
        }
        if top.iter().any(|(label, _)| sample.target[*label]) {
            metrics.any_at_3 += 1;
        }
    }
    metrics
}

fn print_examples(model: &GridModel, samples: &[EncodedSample], indexes: &[usize]) {
    println!();
    println!("Validation examples:");
    for index in indexes.iter().take(16) {
        let sample = &samples[*index];
        let logits = model.predict(&sample.features);
        let top = top_k(&logits, 3);
        println!(
            "{} second={} labels=[{}, {}, {}] predicted=[{} {:.3}, {} {:.3}, {} {:.3}] grid={}",
            sample.id,
            sample.second_index,
            sample.labels[0],
            sample.labels[1],
            sample.labels[2],
            LABELS[top[0].0],
            top[0].1,
            LABELS[top[1].0],
            top[1].1,
            LABELS[top[2].0],
            top[2].1,
            compact_grid(&sample.bins),
        );
    }
}

fn compact_grid(bins: &[u8; FEATURES]) -> String {
    (0..8)
        .map(|sensor| {
            bins[sensor * 8..sensor * 8 + 8]
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join("")
        })
        .collect::<Vec<_>>()
        .join("/")
}

impl Metrics {
    fn score(self) -> f32 {
        self.p_at_1_pct() + self.any_at_3_pct() + self.no_scent_silence_pct()
            - self.false_positive_pct()
            - self.active_silence_pct()
    }

    fn p_at_1_pct(self) -> f32 {
        percentage(self.p_at_1, self.active)
    }
    fn any_at_3_pct(self) -> f32 {
        percentage(self.any_at_3, self.active)
    }
    fn active_silence_pct(self) -> f32 {
        percentage(self.silent_active, self.active)
    }
    fn no_scent_silence_pct(self) -> f32 {
        percentage(self.silent_no_scent, self.no_scent)
    }
    fn false_positive_pct(self) -> f32 {
        percentage(self.false_positive, self.no_scent)
    }
}

fn split_indexes(count: usize, validation_fraction: f32, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut indexes = (0..count).collect::<Vec<_>>();
    let mut rng = Lcg::new(seed);
    for index in (1..indexes.len()).rev() {
        let other = rng.range_usize(0, index + 1);
        indexes.swap(index, other);
    }
    let validation_count =
        ((count as f32 * validation_fraction).round() as usize).clamp(1, count.saturating_sub(1));
    (
        indexes[validation_count..].to_vec(),
        indexes[..validation_count].to_vec(),
    )
}

fn is_no_scent_target(target: &[bool; OUTPUTS]) -> bool {
    !target.iter().any(|value| *value)
}

fn percentage(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 * 100.0 / denominator as f32
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value.clamp(-40.0, 40.0)).exp())
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }
    fn range_usize(&mut self, min: usize, max: usize) -> usize {
        min + (self.next_u32() as usize % (max - min))
    }
}
