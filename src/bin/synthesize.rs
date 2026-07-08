use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

const CHANNELS: usize = 9;
const ROWS_PER_SAMPLE: usize = 96;
const LABELS: [&str; 14] = [
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
    out_dir: PathBuf,
    samples: usize,
    seed: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("synthesize error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    fs::create_dir_all(&config.out_dir)?;

    let mut rng = Lcg::new(config.seed);
    for sample_index in 0..config.samples {
        let labels = choose_labels(&mut rng);
        let sample_id = format!("synthetic_{sample_index:04}");
        let sample_name = format!("Synthetic {sample_index:04}");
        let path = config.out_dir.join(format!("{sample_id}.csv"));
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writeln!(
            writer,
            "sample_id,sample_name,label_1,label_2,label_3,host_elapsed_ms,host_unix_ms,device_seq,device_ms,adc0,adc1,adc2,adc3,adc4,adc5,adc6,adc7,adc8"
        )?;

        let mut baseline = [0.0_f32; CHANNELS];
        for value in &mut baseline {
            *value = 420.0 + rng.range_f32(-45.0, 45.0);
        }

        let mut profile = [0.0_f32; CHANNELS];
        add_label_profile(&mut profile, labels[0], 1.0);
        add_label_profile(&mut profile, labels[1], 0.66);
        add_label_profile(&mut profile, labels[2], 0.33);

        let drift = rng.range_f32(-25.0, 25.0);
        for row in 0..ROWS_PER_SAMPLE {
            let elapsed_ms = row as u64 * 100;
            let t = row as f32 / (ROWS_PER_SAMPLE - 1) as f32;
            let rise = 1.0 - (-5.0 * t).exp();
            let washout = if t > 0.72 {
                1.0 - (t - 0.72) * 0.45
            } else {
                1.0
            };
            let envelope = rise * washout.max(0.6);

            write!(
                writer,
                "{},{},{},{},{},{},{},{},{}",
                sample_id,
                sample_name,
                labels[0],
                labels[1],
                labels[2],
                elapsed_ms,
                1_800_000_000_000_u64 + elapsed_ms + sample_index as u64,
                row,
                elapsed_ms
            )?;

            for channel in 0..CHANNELS {
                let periodic = ((row as f32 * 0.19) + channel as f32).sin() * 18.0;
                let noise = rng.range_f32(-22.0, 22.0);
                let value =
                    baseline[channel] + profile[channel] * envelope + drift * t + periodic + noise;
                write!(writer, ",{}", value.clamp(0.0, 4095.0).round() as u16)?;
            }
            writeln!(writer)?;
        }
    }

    println!(
        "Wrote {} synthetic collector-shaped CSV files to {}",
        config.samples,
        config.out_dir.display()
    );
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut out_dir = PathBuf::from("data/raw");
    let mut samples = 100;
    let mut seed = 0x51a7_2026_u64;

    let args: Vec<String> = env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--out" => {
                index += 1;
                out_dir = PathBuf::from(args.get(index).ok_or("--out requires a path")?);
            }
            "--samples" => {
                index += 1;
                samples = args
                    .get(index)
                    .ok_or("--samples requires a value")?
                    .parse()?;
            }
            "--seed" => {
                index += 1;
                seed = args.get(index).ok_or("--seed requires a value")?.parse()?;
            }
            "--help" | "-h" => {
                println!("Usage: cargo run --bin synthesize -- [--out data/raw] [--samples 100]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        index += 1;
    }

    Ok(Config {
        out_dir,
        samples,
        seed,
    })
}

fn choose_labels(rng: &mut Lcg) -> [&'static str; 3] {
    let first = rng.range_usize(0, LABELS.len());
    let mut second = rng.range_usize(0, LABELS.len());
    while second == first {
        second = rng.range_usize(0, LABELS.len());
    }
    let mut third = rng.range_usize(0, LABELS.len());
    while third == first || third == second {
        third = rng.range_usize(0, LABELS.len());
    }
    [LABELS[first], LABELS[second], LABELS[third]]
}

fn add_label_profile(profile: &mut [f32; CHANNELS], label: &str, weight: f32) {
    let label_index = LABELS
        .iter()
        .position(|candidate| *candidate == label)
        .expect("known synthetic label");

    for (channel, value) in profile.iter_mut().enumerate() {
        let angle = (label_index as f32 + 1.0) * (channel as f32 + 1.0) * 0.71;
        let harmonic = ((label_index + channel * 3) % 5) as f32 * 48.0;
        *value += weight * (260.0 + angle.sin() * 170.0 + harmonic);
    }
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

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }

    fn range_usize(&mut self, min: usize, max: usize) -> usize {
        min + (self.next_u32() as usize % (max - min))
    }
}
