use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

const CHANNELS: usize = 9;
const ROWS_PER_SAMPLE: usize = 900;
const SAMPLE_PERIOD_MS: u64 = 100;
const CLEAN_AIR_MIN: f32 = 150.0;
const CLEAN_AIR_MAX: f32 = 300.0;
const LN_10: f32 = 2.302_585_1;
const NO_SCENT_LABEL: &str = "No Scent";
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

// Numeric matrix-to-CSV mapping:
// adc0 IO1  MQ-2
// adc1 IO2  MQ-3
// adc2 IO16 MQ-5
// adc3 IO17 MQ-6
// adc4 IO18 MQ-7
// adc5 IO21 MQ-8
// adc6 IO22 MQ-9
// adc7 IO23 MQ-135
// adc8 MQ-4 placeholder, currently unused by the matrix and held constant
const MQ2: usize = 0;
const MQ3: usize = 1;
const MQ5: usize = 2;
const MQ6: usize = 3;
const MQ7: usize = 4;
#[allow(dead_code)]
const MQ8: usize = 5;
const MQ9: usize = 6;
const MQ135: usize = 7;
const MQ4_UNUSED: usize = 8;

struct Config {
    out_dir: PathBuf,
    samples: usize,
    seed: u64,
    mode: GenerationMode,
    no_scent_ratio: f32,
    single_note_ratio: f32,
}

#[derive(Clone)]
struct Profile {
    peak: [f32; CHANNELS],
    t90: [f32; CHANNELS],
    residual_offset: [f32; CHANNELS],
    rise_tau: f32,
    exposure_secs: f32,
}

#[derive(Clone, Copy)]
enum GenerationMode {
    Matrix,
    Designer,
}

#[derive(Clone, Copy)]
enum MatrixSampleKind {
    NoScent,
    SingleNote,
    MultiLabel,
}

struct PhaseProfile {
    profile: Profile,
    weight: f32,
    start_secs: f32,
}

struct SyntheticSample {
    id: String,
    name: String,
    labels: [&'static str; 3],
    phases: Vec<PhaseProfile>,
}

struct DesignerRecipe {
    slug: &'static str,
    name: &'static str,
    labels: [&'static str; 3],
    phases: &'static [DesignerPhase],
}

#[derive(Clone, Copy)]
struct DesignerPhase {
    label: &'static str,
    start_secs: f32,
    weight: f32,
    peaks: &'static [(usize, f32)],
    t90: &'static [(usize, f32)],
    residuals: &'static [(usize, f32)],
    rise_tau: f32,
    exposure_secs: f32,
}

const EMPTY: [(usize, f32); 0] = [];

const SAUVAGE_TOP_PEAKS: [(usize, f32); 5] = [
    (MQ2, 4000.0),
    (MQ5, 4000.0),
    (MQ6, 4000.0),
    (MQ7, 3200.0),
    (MQ9, 3200.0),
];
const SAUVAGE_TOP_T90: [(usize, f32); 5] =
    [(MQ2, 8.0), (MQ5, 8.0), (MQ6, 8.0), (MQ7, 12.0), (MQ9, 12.0)];
const SAUVAGE_HEART_PEAKS: [(usize, f32); 2] = [(MQ3, 3600.0), (MQ135, 3400.0)];
const SAUVAGE_HEART_T90: [(usize, f32); 2] = [(MQ3, 40.0), (MQ135, 90.0)];
const SAUVAGE_BASE_PEAKS: [(usize, f32); 2] = [(MQ135, 3800.0), (MQ2, 3500.0)];
const SAUVAGE_BASE_T90: [(usize, f32); 2] = [(MQ135, 190.0), (MQ2, 190.0)];
const SAUVAGE_BASE_RESIDUALS: [(usize, f32); 2] = [(MQ135, 260.0), (MQ2, 220.0)];
const SAUVAGE_PHASES: [DesignerPhase; 3] = [
    DesignerPhase {
        label: "Citrus",
        start_secs: 0.0,
        weight: 1.0,
        peaks: &SAUVAGE_TOP_PEAKS,
        t90: &SAUVAGE_TOP_T90,
        residuals: &EMPTY,
        rise_tau: 0.4,
        exposure_secs: 6.0,
    },
    DesignerPhase {
        label: "Aromatic",
        start_secs: 15.0,
        weight: 1.0,
        peaks: &SAUVAGE_HEART_PEAKS,
        t90: &SAUVAGE_HEART_T90,
        residuals: &EMPTY,
        rise_tau: 2.0,
        exposure_secs: 22.0,
    },
    DesignerPhase {
        label: "Woody Amber",
        start_secs: 60.0,
        weight: 1.0,
        peaks: &SAUVAGE_BASE_PEAKS,
        t90: &SAUVAGE_BASE_T90,
        residuals: &SAUVAGE_BASE_RESIDUALS,
        rise_tau: 7.0,
        exposure_secs: 30.0,
    },
];

const SANTAL_TOP_PEAKS: [(usize, f32); 2] = [(MQ2, 2800.0), (MQ135, 2800.0)];
const SANTAL_TOP_T90: [(usize, f32); 2] = [(MQ2, 75.0), (MQ135, 75.0)];
const SANTAL_HEART_PEAKS: [(usize, f32); 3] = [(MQ135, 3200.0), (MQ2, 3100.0), (MQ3, 2200.0)];
const SANTAL_HEART_T90: [(usize, f32); 3] = [(MQ135, 90.0), (MQ2, 90.0), (MQ3, 40.0)];
const SANTAL_BASE_PEAKS: [(usize, f32); 3] = [(MQ7, 3900.0), (MQ9, 3600.0), (MQ135, 3800.0)];
const SANTAL_BASE_T90: [(usize, f32); 3] = [(MQ7, 135.0), (MQ9, 135.0), (MQ135, 135.0)];
const SANTAL_BASE_RESIDUALS: [(usize, f32); 2] = [(MQ7, 280.0), (MQ9, 220.0)];
const SANTAL_33_PHASES: [DesignerPhase; 3] = [
    DesignerPhase {
        label: "Woods",
        start_secs: 0.0,
        weight: 0.9,
        peaks: &SANTAL_TOP_PEAKS,
        t90: &SANTAL_TOP_T90,
        residuals: &EMPTY,
        rise_tau: 3.0,
        exposure_secs: 20.0,
    },
    DesignerPhase {
        label: "Woods",
        start_secs: 15.0,
        weight: 1.0,
        peaks: &SANTAL_HEART_PEAKS,
        t90: &SANTAL_HEART_T90,
        residuals: &EMPTY,
        rise_tau: 4.0,
        exposure_secs: 40.0,
    },
    DesignerPhase {
        label: "Dry Woods",
        start_secs: 60.0,
        weight: 1.0,
        peaks: &SANTAL_BASE_PEAKS,
        t90: &SANTAL_BASE_T90,
        residuals: &SANTAL_BASE_RESIDUALS,
        rise_tau: 5.0,
        exposure_secs: 35.0,
    },
];

const BLACK_TOP_PEAKS: [(usize, f32); 3] = [(MQ3, 3800.0), (MQ135, 2800.0), (MQ2, 2800.0)];
const BLACK_TOP_T90: [(usize, f32); 3] = [(MQ3, 14.0), (MQ135, 120.0), (MQ2, 120.0)];
const BLACK_HEART_PEAKS: [(usize, f32); 2] = [(MQ3, 4095.0), (MQ7, 3400.0)];
const BLACK_HEART_T90: [(usize, f32); 2] = [(MQ3, 60.0), (MQ7, 90.0)];
const BLACK_BASE_PEAKS: [(usize, f32); 4] =
    [(MQ135, 4095.0), (MQ7, 3900.0), (MQ9, 3800.0), (MQ2, 3500.0)];
const BLACK_BASE_T90: [(usize, f32); 4] =
    [(MQ135, 300.0), (MQ7, 300.0), (MQ9, 300.0), (MQ2, 300.0)];
const BLACK_BASE_RESIDUALS: [(usize, f32); 4] =
    [(MQ135, 400.0), (MQ7, 400.0), (MQ9, 400.0), (MQ2, 400.0)];
const BLACK_ORCHID_PHASES: [DesignerPhase; 3] = [
    DesignerPhase {
        label: "Fruity",
        start_secs: 0.0,
        weight: 1.0,
        peaks: &BLACK_TOP_PEAKS,
        t90: &BLACK_TOP_T90,
        residuals: &EMPTY,
        rise_tau: 2.0,
        exposure_secs: 18.0,
    },
    DesignerPhase {
        label: "Floral Amber",
        start_secs: 15.0,
        weight: 1.0,
        peaks: &BLACK_HEART_PEAKS,
        t90: &BLACK_HEART_T90,
        residuals: &EMPTY,
        rise_tau: 3.0,
        exposure_secs: 45.0,
    },
    DesignerPhase {
        label: "Amber",
        start_secs: 60.0,
        weight: 1.0,
        peaks: &BLACK_BASE_PEAKS,
        t90: &BLACK_BASE_T90,
        residuals: &BLACK_BASE_RESIDUALS,
        rise_tau: 7.0,
        exposure_secs: 40.0,
    },
];

const ACQUA_TOP_PEAKS: [(usize, f32); 3] = [(MQ5, 3000.0), (MQ6, 3000.0), (MQ135, 2500.0)];
const ACQUA_TOP_T90: [(usize, f32); 3] = [(MQ5, 6.0), (MQ6, 6.0), (MQ135, 8.0)];
const ACQUA_HEART_PEAKS: [(usize, f32); 2] = [(MQ3, 4095.0), (MQ135, 1800.0)];
const ACQUA_HEART_T90: [(usize, f32); 2] = [(MQ3, 38.0), (MQ135, 15.0)];
const ACQUA_BASE_PEAKS: [(usize, f32); 4] =
    [(MQ2, 1700.0), (MQ3, 2200.0), (MQ7, 1600.0), (MQ135, 2200.0)];
const ACQUA_BASE_T90: [(usize, f32); 4] = [(MQ2, 18.0), (MQ3, 18.0), (MQ7, 18.0), (MQ135, 18.0)];
const ACQUA_DI_GIO_PHASES: [DesignerPhase; 3] = [
    DesignerPhase {
        label: "Citrus",
        start_secs: 0.0,
        weight: 0.8,
        peaks: &ACQUA_TOP_PEAKS,
        t90: &ACQUA_TOP_T90,
        residuals: &EMPTY,
        rise_tau: 0.7,
        exposure_secs: 6.0,
    },
    DesignerPhase {
        label: "Water",
        start_secs: 15.0,
        weight: 1.0,
        peaks: &ACQUA_HEART_PEAKS,
        t90: &ACQUA_HEART_T90,
        residuals: &EMPTY,
        rise_tau: 2.0,
        exposure_secs: 35.0,
    },
    DesignerPhase {
        label: "Soft Floral",
        start_secs: 60.0,
        weight: 0.55,
        peaks: &ACQUA_BASE_PEAKS,
        t90: &ACQUA_BASE_T90,
        residuals: &EMPTY,
        rise_tau: 1.2,
        exposure_secs: 20.0,
    },
];

const FLOWER_TOP_PEAKS: [(usize, f32); 2] = [(MQ135, 2500.0), (MQ3, 2500.0)];
const FLOWER_TOP_T90: [(usize, f32); 2] = [(MQ135, 7.0), (MQ3, 7.0)];
const FLOWER_HEART_PEAKS: [(usize, f32); 2] = [(MQ3, 4095.0), (MQ135, 3000.0)];
const FLOWER_HEART_T90: [(usize, f32); 2] = [(MQ3, 90.0), (MQ135, 90.0)];
const FLOWER_BASE_PEAKS: [(usize, f32); 2] = [(MQ135, 4095.0), (MQ7, 3000.0)];
const FLOWER_BASE_T90: [(usize, f32); 2] = [(MQ135, 120.0), (MQ7, 120.0)];
const FLOWERBOMB_PHASES: [DesignerPhase; 3] = [
    DesignerPhase {
        label: "Green",
        start_secs: 0.0,
        weight: 0.75,
        peaks: &FLOWER_TOP_PEAKS,
        t90: &FLOWER_TOP_T90,
        residuals: &EMPTY,
        rise_tau: 0.7,
        exposure_secs: 8.0,
    },
    DesignerPhase {
        label: "Floral",
        start_secs: 15.0,
        weight: 1.0,
        peaks: &FLOWER_HEART_PEAKS,
        t90: &FLOWER_HEART_T90,
        residuals: &EMPTY,
        rise_tau: 2.0,
        exposure_secs: 45.0,
    },
    DesignerPhase {
        label: "Amber",
        start_secs: 60.0,
        weight: 1.0,
        peaks: &FLOWER_BASE_PEAKS,
        t90: &FLOWER_BASE_T90,
        residuals: &EMPTY,
        rise_tau: 5.0,
        exposure_secs: 35.0,
    },
];

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
    let no_scent_target = if matches!(config.mode, GenerationMode::Matrix) {
        ((config.samples as f32 * config.no_scent_ratio).round() as usize).min(config.samples)
    } else {
        0
    };
    let single_note_target = if matches!(config.mode, GenerationMode::Matrix) {
        ((config.samples as f32 * config.single_note_ratio).round() as usize)
            .min(config.samples - no_scent_target)
    } else {
        0
    };
    for sample_index in 0..config.samples {
        let sample = match config.mode {
            GenerationMode::Matrix => matrix_sample(
                sample_index,
                &mut rng,
                matrix_sample_kind(sample_index, no_scent_target, single_note_target),
            ),
            GenerationMode::Designer => designer_sample(sample_index, &mut rng),
        };
        let path = config.out_dir.join(format!("{}.csv", sample.id));
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writeln!(
            writer,
            "sample_id,sample_name,label_1,label_2,label_3,host_elapsed_ms,host_unix_ms,device_seq,device_ms,adc0,adc1,adc2,adc3,adc4,adc5,adc6,adc7,adc8"
        )?;

        let mut baseline = [0.0_f32; CHANNELS];
        for value in &mut baseline {
            *value = rng.range_f32(CLEAN_AIR_MIN, CLEAN_AIR_MAX);
        }
        baseline[MQ4_UNUSED] = 220.0;

        let drift = rng.range_f32(-8.0, 8.0);

        for row in 0..ROWS_PER_SAMPLE {
            let elapsed_ms = row as u64 * SAMPLE_PERIOD_MS;
            let seconds = elapsed_ms as f32 / 1000.0;

            write!(
                writer,
                "{},{},{},{},{},{},{},{},{}",
                sample.id,
                sample.name,
                sample.labels[0],
                sample.labels[1],
                sample.labels[2],
                elapsed_ms,
                1_800_000_000_000_u64 + elapsed_ms + sample_index as u64,
                row,
                elapsed_ms
            )?;

            for channel in 0..CHANNELS {
                let value = if channel == MQ4_UNUSED {
                    baseline[channel]
                } else {
                    let mut signal = 0.0_f32;
                    for phase in &sample.phases {
                        if seconds >= phase.start_secs {
                            signal += phase.weight
                                * profile_value(
                                    &phase.profile,
                                    channel,
                                    baseline[channel],
                                    seconds - phase.start_secs,
                                );
                        }
                    }

                    let periodic = ((row as f32 * 0.07) + channel as f32).sin() * 3.5;
                    let noise = rng.range_f32(-4.0, 4.0);
                    let drift_value = drift * (seconds / total_secs());
                    baseline[channel] + signal + drift_value + periodic + noise
                };
                write!(writer, ",{}", value.clamp(0.0, 4095.0).round() as u16)?;
            }
            writeln!(writer)?;
        }
    }

    println!(
        "Wrote {} {} synthetic CSV files to {}",
        config.samples,
        match config.mode {
            GenerationMode::Matrix => "numeric-matrix",
            GenerationMode::Designer => "designer-phase",
        },
        config.out_dir.display()
    );
    if matches!(config.mode, GenerationMode::Matrix) {
        println!(
            "Matrix mix: no_scent={} single_note={} multi_label={}",
            no_scent_target,
            single_note_target,
            config.samples - no_scent_target - single_note_target
        );
    }
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut out_dir = PathBuf::from("data/raw");
    let mut samples = 100;
    let mut seed = 0x51a7_2026_u64;
    let mut mode = GenerationMode::Matrix;
    let mut no_scent_ratio: f32 = 0.25;
    let mut single_note_ratio: f32 = 0.0;

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
            "--designer" => {
                mode = GenerationMode::Designer;
            }
            "--no-scent-ratio" => {
                index += 1;
                no_scent_ratio = args
                    .get(index)
                    .ok_or("--no-scent-ratio requires a value")?
                    .parse()?;
            }
            "--single-note-ratio" => {
                index += 1;
                single_note_ratio = args
                    .get(index)
                    .ok_or("--single-note-ratio requires a value")?
                    .parse()?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin synthesize -- [--out data/raw] [--samples 100] [--designer] [--no-scent-ratio 0.25] [--single-note-ratio 0.0]"
                );
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
        mode,
        no_scent_ratio: no_scent_ratio.clamp(0.0, 1.0),
        single_note_ratio: single_note_ratio.clamp(0.0, 1.0),
    })
}

fn matrix_sample(sample_index: usize, rng: &mut Lcg, kind: MatrixSampleKind) -> SyntheticSample {
    match kind {
        MatrixSampleKind::NoScent => return no_scent_sample(sample_index),
        MatrixSampleKind::SingleNote => return single_note_sample(sample_index, rng),
        MatrixSampleKind::MultiLabel => {}
    }

    let labels = choose_labels(rng);
    SyntheticSample {
        id: format!("synthetic_{sample_index:04}"),
        name: format!("Synthetic {sample_index:04}"),
        labels,
        phases: vec![
            PhaseProfile {
                profile: varied_profile(profile_for_label(labels[0]), rng),
                weight: 1.0,
                start_secs: 0.0,
            },
            PhaseProfile {
                profile: varied_profile(profile_for_label(labels[1]), rng),
                weight: 0.66,
                start_secs: 0.0,
            },
            PhaseProfile {
                profile: varied_profile(profile_for_label(labels[2]), rng),
                weight: 0.33,
                start_secs: 0.0,
            },
        ],
    }
}

fn matrix_sample_kind(
    sample_index: usize,
    no_scent_target: usize,
    single_note_target: usize,
) -> MatrixSampleKind {
    if sample_index < no_scent_target {
        MatrixSampleKind::NoScent
    } else if sample_index < no_scent_target + single_note_target {
        MatrixSampleKind::SingleNote
    } else {
        MatrixSampleKind::MultiLabel
    }
}

fn no_scent_sample(sample_index: usize) -> SyntheticSample {
    SyntheticSample {
        id: format!("no_scent_{sample_index:04}"),
        name: format!("No Scent {sample_index:04}"),
        labels: [NO_SCENT_LABEL, NO_SCENT_LABEL, NO_SCENT_LABEL],
        phases: Vec::new(),
    }
}

fn single_note_sample(sample_index: usize, rng: &mut Lcg) -> SyntheticSample {
    let label = LABELS[rng.range_usize(0, LABELS.len())];
    SyntheticSample {
        id: format!(
            "single_{}_{sample_index:04}",
            label
                .to_ascii_lowercase()
                .replace(' ', "_")
                .replace('-', "_")
        ),
        name: format!("Single {label} {sample_index:04}"),
        labels: [label, NO_SCENT_LABEL, NO_SCENT_LABEL],
        phases: vec![PhaseProfile {
            profile: varied_profile(profile_for_label(label), rng),
            weight: 1.0,
            start_secs: 0.0,
        }],
    }
}

fn designer_sample(sample_index: usize, rng: &mut Lcg) -> SyntheticSample {
    let recipes = designer_recipes();
    let recipe = &recipes[rng.range_usize(0, recipes.len())];
    let variant = sample_index;
    let phases = recipe
        .phases
        .iter()
        .map(|phase| PhaseProfile {
            profile: varied_profile(profile_from_designer_phase(phase), rng),
            weight: phase.weight * rng.range_f32(0.9, 1.1),
            start_secs: (phase.start_secs + rng.range_f32(-1.0, 1.0)).max(0.0),
        })
        .collect();

    SyntheticSample {
        id: format!("designer_{}_{variant:04}", recipe.slug),
        name: format!("{} Variant {variant:04}", recipe.name),
        labels: recipe.labels,
        phases,
    }
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

fn designer_recipes() -> Vec<DesignerRecipe> {
    vec![
        DesignerRecipe {
            slug: "sauvage",
            name: "Sauvage Type",
            labels: ["Aromatic", "Citrus", "Woody Amber"],
            phases: &SAUVAGE_PHASES,
        },
        DesignerRecipe {
            slug: "santal_33",
            name: "Santal 33 Type",
            labels: ["Dry Woods", "Woods", "Soft Floral"],
            phases: &SANTAL_33_PHASES,
        },
        DesignerRecipe {
            slug: "black_orchid",
            name: "Black Orchid Type",
            labels: ["Amber", "Woody Amber", "Floral Amber"],
            phases: &BLACK_ORCHID_PHASES,
        },
        DesignerRecipe {
            slug: "acqua_di_gio",
            name: "Acqua di Gio Type",
            labels: ["Water", "Citrus", "Floral"],
            phases: &ACQUA_DI_GIO_PHASES,
        },
        DesignerRecipe {
            slug: "flowerbomb",
            name: "Flowerbomb Type",
            labels: ["Floral", "Amber", "Green"],
            phases: &FLOWERBOMB_PHASES,
        },
    ]
}

fn profile_from_designer_phase(phase: &DesignerPhase) -> Profile {
    let mut profile = profile_for_label(phase.label);
    profile.peak = [0.0; CHANNELS];
    profile.t90 = [30.0; CHANNELS];
    profile.residual_offset = [0.0; CHANNELS];
    profile.rise_tau = phase.rise_tau;
    profile.exposure_secs = phase.exposure_secs;
    set_peaks(&mut profile, phase.peaks);
    for (channel, t90) in phase.t90 {
        profile.t90[*channel] = *t90;
    }
    for (channel, residual) in phase.residuals {
        profile.residual_offset[*channel] = *residual;
    }
    profile
}

fn varied_profile(mut profile: Profile, rng: &mut Lcg) -> Profile {
    for channel in 0..CHANNELS {
        if profile.peak[channel] > 0.0 {
            profile.peak[channel] *= rng.range_f32(0.9, 1.1);
            profile.t90[channel] *= rng.range_f32(0.9, 1.1);
            profile.residual_offset[channel] *= rng.range_f32(0.9, 1.1);
        }
    }
    profile.rise_tau *= rng.range_f32(0.9, 1.1);
    profile.exposure_secs *= rng.range_f32(0.9, 1.1);
    profile
}

fn profile_for_label(label: &str) -> Profile {
    let mut profile = Profile {
        peak: [0.0; CHANNELS],
        t90: [30.0; CHANNELS],
        residual_offset: [0.0; CHANNELS],
        rise_tau: 1.2,
        exposure_secs: 8.0,
    };

    match label {
        "Citrus" => {
            set_peaks(
                &mut profile,
                &[
                    (MQ2, 3800.0),
                    (MQ3, 2000.0),
                    (MQ5, 3500.0),
                    (MQ6, 3700.0),
                    (MQ135, 2200.0),
                ],
            );
            set_t90_all_active(&mut profile, 7.0);
            profile.rise_tau = 0.45;
            profile.exposure_secs = 5.0;
        }
        "Water" => {
            set_peaks(&mut profile, &[(MQ3, 800.0), (MQ135, 1800.0)]);
            profile.t90[MQ3] = 5.0;
            profile.t90[MQ135] = 15.0;
            profile.residual_offset[MQ135] = 25.0;
            profile.rise_tau = 0.8;
            profile.exposure_secs = 6.0;
        }
        "Green" => {
            set_peaks(
                &mut profile,
                &[(MQ3, 2500.0), (MQ6, 600.0), (MQ135, 3000.0)],
            );
            set_t90_all_active(&mut profile, 6.5);
            profile.rise_tau = 0.6;
            profile.exposure_secs = 5.0;
        }
        "Fruity" => {
            set_peaks(
                &mut profile,
                &[(MQ2, 1200.0), (MQ3, 3800.0), (MQ135, 2200.0)],
            );
            set_t90_all_active(&mut profile, 13.5);
            profile.rise_tau = 1.0;
            profile.exposure_secs = 7.0;
        }
        "Floral" => {
            set_peaks(&mut profile, &[(MQ3, 4095.0), (MQ135, 1500.0)]);
            profile.t90[MQ3] = 38.0;
            profile.t90[MQ135] = 38.0;
            profile.rise_tau = 2.0;
            profile.exposure_secs = 9.0;
        }
        "Soft Floral" => {
            set_peaks(
                &mut profile,
                &[(MQ3, 2200.0), (MQ7, 2100.0), (MQ135, 2200.0)],
            );
            set_t90_all_active(&mut profile, 40.0);
            profile.rise_tau = 2.0;
            profile.exposure_secs = 9.0;
        }
        "Floral Amber" => {
            set_peaks(
                &mut profile,
                &[
                    (MQ2, 1100.0),
                    (MQ3, 4095.0),
                    (MQ7, 3400.0),
                    (MQ9, 3000.0),
                    (MQ135, 3800.0),
                ],
            );
            profile.t90[MQ2] = 30.0;
            profile.t90[MQ3] = 8.0;
            profile.t90[MQ7] = 60.0;
            profile.t90[MQ9] = 55.0;
            profile.t90[MQ135] = 60.0;
            profile.rise_tau = 1.8;
            profile.exposure_secs = 9.0;
        }
        "Soft Amber" => {
            set_peaks(
                &mut profile,
                &[(MQ2, 900.0), (MQ3, 1200.0), (MQ7, 3700.0), (MQ135, 3500.0)],
            );
            profile.t90[MQ2] = 70.0;
            profile.t90[MQ3] = 70.0;
            profile.t90[MQ7] = 95.0;
            profile.t90[MQ135] = 85.0;
            profile.residual_offset[MQ7] = 180.0;
            profile.rise_tau = 4.0;
            profile.exposure_secs = 11.0;
        }
        "Amber" => {
            set_peaks(
                &mut profile,
                &[
                    (MQ2, 2200.0),
                    (MQ3, 2500.0),
                    (MQ7, 3900.0),
                    (MQ9, 3800.0),
                    (MQ135, 4095.0),
                ],
            );
            set_t90_all_active(&mut profile, 180.0);
            for channel in 0..CHANNELS {
                profile.residual_offset[channel] = 300.0;
            }
            profile.rise_tau = 4.8;
            profile.exposure_secs = 12.0;
        }
        "Woody Amber" => {
            set_peaks(
                &mut profile,
                &[
                    (MQ2, 3500.0),
                    (MQ3, 1800.0),
                    (MQ7, 3000.0),
                    (MQ9, 3400.0),
                    (MQ135, 4000.0),
                ],
            );
            set_t90_all_active(&mut profile, 190.0);
            profile.residual_offset[MQ2] = 220.0;
            profile.residual_offset[MQ135] = 260.0;
            profile.rise_tau = 5.0;
            profile.exposure_secs = 12.0;
        }
        "Woods" => {
            set_peaks(
                &mut profile,
                &[(MQ2, 3100.0), (MQ3, 900.0), (MQ135, 3200.0)],
            );
            set_t90_all_active(&mut profile, 75.0);
            profile.rise_tau = 3.6;
            profile.exposure_secs = 10.0;
        }
        "Mossy Woods" => {
            set_peaks(
                &mut profile,
                &[(MQ2, 2800.0), (MQ3, 2900.0), (MQ7, 3100.0), (MQ135, 3000.0)],
            );
            set_t90_all_active(&mut profile, 120.0);
            profile.rise_tau = 4.0;
            profile.exposure_secs = 11.0;
        }
        "Dry Woods" => {
            set_peaks(
                &mut profile,
                &[
                    (MQ2, 2000.0),
                    (MQ3, 1100.0),
                    (MQ7, 3900.0),
                    (MQ9, 3600.0),
                    (MQ135, 3800.0),
                ],
            );
            set_t90_all_active(&mut profile, 135.0);
            profile.residual_offset[MQ7] = 280.0;
            profile.residual_offset[MQ9] = 220.0;
            profile.rise_tau = 4.2;
            profile.exposure_secs = 11.0;
        }
        "Aromatic" => {
            set_peaks(
                &mut profile,
                &[
                    (MQ2, 2400.0),
                    (MQ3, 3600.0),
                    (MQ6, 2900.0),
                    (MQ7, 2900.0),
                    (MQ135, 3400.0),
                ],
            );
            profile.t90[MQ6] = 10.0;
            profile.t90[MQ3] = 40.0;
            profile.t90[MQ2] = 35.0;
            profile.t90[MQ7] = 90.0;
            profile.t90[MQ135] = 90.0;
            profile.rise_tau = 1.2;
            profile.exposure_secs = 9.0;
        }
        _ => unreachable!("unknown label"),
    }

    profile
}

fn profile_value(profile: &Profile, channel: usize, baseline: f32, seconds: f32) -> f32 {
    let peak = profile.peak[channel];
    if peak <= 0.0 {
        return 0.0;
    }

    let amplitude = (peak - baseline).max(0.0);
    let residual = profile.residual_offset[channel].min(amplitude * 0.8);

    if seconds <= profile.exposure_secs {
        let numerator = 1.0 - (-seconds / profile.rise_tau).exp();
        let denominator = (1.0 - (-profile.exposure_secs / profile.rise_tau).exp()).max(0.001);
        amplitude * numerator / denominator
    } else {
        let elapsed = seconds - profile.exposure_secs;
        let tau = (profile.t90[channel] / LN_10).max(0.1);
        residual + (amplitude - residual) * (-elapsed / tau).exp()
    }
}

fn set_peaks(profile: &mut Profile, pairs: &[(usize, f32)]) {
    for (channel, peak) in pairs {
        profile.peak[*channel] = *peak;
    }
}

fn set_t90_all_active(profile: &mut Profile, t90: f32) {
    for channel in 0..CHANNELS {
        if profile.peak[channel] > 0.0 {
            profile.t90[channel] = t90;
        }
    }
}

fn total_secs() -> f32 {
    (ROWS_PER_SAMPLE - 1) as f32 * SAMPLE_PERIOD_MS as f32 / 1000.0
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn citrus_uses_expected_numeric_peaks() {
        let profile = profile_for_label("Citrus");
        assert_eq!(profile.peak[MQ2], 3800.0);
        assert_eq!(profile.peak[MQ3], 2000.0);
        assert_eq!(profile.peak[MQ5], 3500.0);
        assert_eq!(profile.peak[MQ6], 3700.0);
        assert_eq!(profile.peak[MQ7], 0.0);
        assert_eq!(profile.peak[MQ8], 0.0);
        assert_eq!(profile.peak[MQ9], 0.0);
        assert_eq!(profile.peak[MQ135], 2200.0);
    }

    #[test]
    fn amber_leaves_fixed_offset() {
        let profile = profile_for_label("Amber");
        assert_eq!(profile.peak[MQ135], 4095.0);
        assert_eq!(profile.residual_offset[MQ135], 300.0);
        assert!(profile_value(&profile, MQ135, 200.0, 60.0) > 300.0);
    }

    #[test]
    fn mq4_placeholder_has_no_profile_signal() {
        for label in LABELS {
            assert_eq!(profile_for_label(label).peak[MQ4_UNUSED], 0.0);
        }
    }

    #[test]
    fn no_scent_sample_has_no_phases() {
        let sample = no_scent_sample(7);

        assert_eq!(
            sample.labels,
            [NO_SCENT_LABEL, NO_SCENT_LABEL, NO_SCENT_LABEL]
        );
        assert!(sample.phases.is_empty());
    }

    #[test]
    fn single_note_sample_has_one_fragrance_phase() {
        let mut rng = Lcg::new(123);
        let sample = single_note_sample(4, &mut rng);

        assert_ne!(sample.labels[0], NO_SCENT_LABEL);
        assert_eq!(sample.labels[1], NO_SCENT_LABEL);
        assert_eq!(sample.labels[2], NO_SCENT_LABEL);
        assert_eq!(sample.phases.len(), 1);
    }
}
