use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const CHANNELS: usize = 9;
const NO_SCENT_LABEL: &str = "No Scent";
const DEFAULT_INPUT: &str = "data/training/snn_comprehensive";
const DEFAULT_OUTPUT: &str = "data/streams/snn_comprehensive_stream.csv";
const DEFAULT_INITIAL_NO_SCENT_CAPTURES: usize = 3;
const DEFAULT_MAX_GAP_NO_SCENT_CAPTURES: usize = 3;
const SAMPLE_PERIOD_MS: u64 = 100;

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
    input_dir: PathBuf,
    output_path: PathBuf,
    initial_no_scent_captures: usize,
    max_gap_no_scent_captures: usize,
    seed: u64,
    limit: Option<usize>,
}

#[derive(Clone)]
struct Capture {
    id: String,
    name: String,
    labels: [String; 3],
    rows: Vec<[u16; CHANNELS]>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("stitch_stream error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let captures = load_captures(&config.input_dir)?;
    let scent = captures
        .iter()
        .filter(|capture| !is_no_scent(&capture.labels))
        .cloned()
        .collect::<Vec<_>>();
    let mut no_scent = captures
        .iter()
        .filter(|capture| is_no_scent(&capture.labels))
        .cloned()
        .collect::<Vec<_>>();

    if scent.is_empty() {
        return Err("stream stitching needs at least one fragrance capture".into());
    }
    if no_scent.is_empty() {
        return Err("stream stitching needs at least one no-scent capture".into());
    }

    let mut rng = Lcg::new(config.seed);
    shuffle(&mut no_scent, &mut rng);
    let (mut scent, balance) = balanced_scent_captures(scent, config.limit, &mut rng)?;
    shuffle(&mut scent, &mut rng);

    let mut gap_source = GapSource::new(no_scent);

    if let Some(parent) = config.output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(&config.output_path)?;
    writeln!(
        file,
        "sample_id,sample_name,label_1,label_2,label_3,host_elapsed_ms,host_unix_ms,device_seq,device_ms,adc0,adc1,adc2,adc3,adc4,adc5,adc6,adc7,adc8,stream_segment,stream_kind,source_sample_id"
    )?;

    let mut stream_row = 0_u64;
    let mut no_scent_rows_written = 0_usize;
    let mut scent_rows_written = 0_usize;
    let mut stream_segment = 0_usize;
    let no_scent_labels = [
        NO_SCENT_LABEL.to_string(),
        NO_SCENT_LABEL.to_string(),
        NO_SCENT_LABEL.to_string(),
    ];

    for _ in 0..config.initial_no_scent_captures {
        let gap = gap_source.next_capture();
        no_scent_rows_written += write_capture_rows(
            &mut file,
            &mut stream_row,
            stream_segment,
            "no_scent",
            &gap.id,
            "Stream No Scent Prelude",
            &no_scent_labels,
            &gap.rows,
        )?;
        stream_segment += 1;
    }

    for capture in &scent {
        scent_rows_written += write_capture_rows(
            &mut file,
            &mut stream_row,
            stream_segment,
            "scent",
            &capture.id,
            &capture.name,
            &capture.labels,
            &capture.rows,
        )?;
        stream_segment += 1;

        let gap_count = rng.range_usize(0, config.max_gap_no_scent_captures + 1);
        for _ in 0..gap_count {
            let gap = gap_source.next_capture();
            no_scent_rows_written += write_capture_rows(
                &mut file,
                &mut stream_row,
                stream_segment,
                "no_scent",
                &gap.id,
                "Stream No Scent Gap",
                &no_scent_labels,
                &gap.rows,
            )?;
            stream_segment += 1;
        }
    }

    let total = scent_rows_written + no_scent_rows_written;
    let actual_no_scent_ratio = if total == 0 {
        0.0
    } else {
        no_scent_rows_written as f32 / total as f32
    };
    println!(
        "Wrote stream CSV to {}",
        config.output_path.display()
    );
    println!(
        "Stream scent_segments={} single={} two={} three={} target_per_bucket={} total_segments={} rows={} scent_rows={} no_scent_rows={} no_scent_ratio={:.3} initial_no_scent={} max_gap_no_scent={}",
        scent.len(),
        balance.single,
        balance.two,
        balance.three,
        balance.target_per_bucket,
        stream_segment,
        total,
        scent_rows_written,
        no_scent_rows_written,
        actual_no_scent_ratio,
        config.initial_no_scent_captures,
        config.max_gap_no_scent_captures
    );
    Ok(())
}

struct BalanceSummary {
    single: usize,
    two: usize,
    three: usize,
    target_per_bucket: usize,
}

fn balanced_scent_captures(
    captures: Vec<Capture>,
    limit_per_bucket: Option<usize>,
    rng: &mut Lcg,
) -> Result<(Vec<Capture>, BalanceSummary), Box<dyn std::error::Error>> {
    let mut buckets = [Vec::new(), Vec::new(), Vec::new()];
    for capture in captures {
        let count = active_label_count(&capture.labels);
        if !(1..=3).contains(&count) {
            return Err(format!("{} has unsupported active label count {count}", capture.id).into());
        }
        buckets[count - 1].push(capture);
    }

    for (index, bucket) in buckets.iter().enumerate() {
        if bucket.is_empty() {
            return Err(format!("missing {}-note captures for stream balancing", index + 1).into());
        }
    }
    for bucket in &mut buckets {
        shuffle(bucket, rng);
    }

    let natural_target = buckets.iter().map(Vec::len).max().unwrap_or(0);
    let target_per_bucket = limit_per_bucket.unwrap_or(natural_target).max(1);
    let mut balanced = Vec::with_capacity(target_per_bucket * buckets.len());
    let mut counts = [0_usize; 3];
    for (bucket_index, bucket) in buckets.iter().enumerate() {
        for index in 0..target_per_bucket {
            balanced.push(bucket[index % bucket.len()].clone());
            counts[bucket_index] += 1;
        }
    }

    Ok((
        balanced,
        BalanceSummary {
            single: counts[0],
            two: counts[1],
            three: counts[2],
            target_per_bucket,
        },
    ))
}

struct GapSource {
    captures: Vec<Capture>,
    capture_index: usize,
}

impl GapSource {
    fn new(captures: Vec<Capture>) -> Self {
        Self {
            captures,
            capture_index: 0,
        }
    }

    fn next_capture(&mut self) -> Capture {
        let capture = self.captures[self.capture_index].clone();
        self.capture_index = (self.capture_index + 1) % self.captures.len();
        capture
    }
}

fn write_capture_rows(
    file: &mut fs::File,
    stream_row: &mut u64,
    segment_index: usize,
    kind: &str,
    source_id: &str,
    sample_name: &str,
    labels: &[String; 3],
    rows: &[[u16; CHANNELS]],
) -> Result<usize, Box<dyn std::error::Error>> {
    for row in rows {
        write_row(
            file,
            *stream_row,
            segment_index,
            kind,
            source_id,
            sample_name,
            labels,
            *row,
        )?;
        *stream_row += 1;
    }
    Ok(rows.len())
}

fn write_row(
    file: &mut fs::File,
    stream_row: u64,
    segment_index: usize,
    kind: &str,
    source_id: &str,
    sample_name: &str,
    labels: &[String; 3],
    adc: [u16; CHANNELS],
) -> Result<(), Box<dyn std::error::Error>> {
    let sample_id = format!("stream_{stream_row:010}");
    let host_elapsed_ms = stream_row * SAMPLE_PERIOD_MS;
    let device_seq = stream_row;
    let device_ms = host_elapsed_ms;
    write!(
        file,
        "{},{},{},{},{},{},{},{},{},",
        escape_csv(&sample_id),
        escape_csv(sample_name),
        escape_csv(&labels[0]),
        escape_csv(&labels[1]),
        escape_csv(&labels[2]),
        host_elapsed_ms,
        0,
        device_seq,
        device_ms
    )?;
    for (index, value) in adc.iter().enumerate() {
        if index > 0 {
            write!(file, ",")?;
        }
        write!(file, "{value}")?;
    }
    writeln!(
        file,
        ",seg{segment_index:06},{},{}",
        escape_csv(kind),
        escape_csv(source_id)
    )?;
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut input_dir = PathBuf::from(DEFAULT_INPUT);
    let mut output_path = PathBuf::from(DEFAULT_OUTPUT);
    let mut initial_no_scent_captures = DEFAULT_INITIAL_NO_SCENT_CAPTURES;
    let mut max_gap_no_scent_captures = DEFAULT_MAX_GAP_NO_SCENT_CAPTURES;
    let mut seed = 0x57_ea_3_u64;
    let mut limit = None;

    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--input" => {
                index += 1;
                input_dir = PathBuf::from(args.get(index).ok_or("--input requires a path")?);
            }
            "--out" => {
                index += 1;
                output_path = PathBuf::from(args.get(index).ok_or("--out requires a path")?);
            }
            "--initial-no-scent-captures" => {
                index += 1;
                initial_no_scent_captures = args
                    .get(index)
                    .ok_or("--initial-no-scent-captures requires a value")?
                    .parse()?;
            }
            "--max-gap-no-scent-captures" => {
                index += 1;
                max_gap_no_scent_captures = args
                    .get(index)
                    .ok_or("--max-gap-no-scent-captures requires a value")?
                    .parse()?;
            }
            "--seed" => {
                index += 1;
                seed = args.get(index).ok_or("--seed requires a value")?.parse()?;
            }
            "--limit" => {
                index += 1;
                limit = Some(args.get(index).ok_or("--limit requires a value")?.parse()?);
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --bin stitch_stream -- [--input data/training/snn_comprehensive] [--out data/streams/snn_comprehensive_stream.csv] [--initial-no-scent-captures 3] [--max-gap-no-scent-captures 3] [--limit per-note-bucket]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        index += 1;
    }

    if initial_no_scent_captures == 0 {
        return Err("--initial-no-scent-captures must be at least 1".into());
    }

    Ok(Config {
        input_dir,
        output_path,
        initial_no_scent_captures,
        max_gap_no_scent_captures,
        seed,
        limit,
    })
}

fn load_captures(data_dir: &Path) -> Result<Vec<Capture>, Box<dyn std::error::Error>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(data_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) == Some("csv") {
            paths.push(path);
        }
    }
    paths.sort();

    let mut captures = Vec::new();
    for path in paths {
        if let Some(capture) = load_capture(&path)? {
            captures.push(capture);
        }
    }
    Ok(captures)
}

fn load_capture(path: &Path) -> Result<Option<Capture>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = match lines.next() {
        Some(header) => header,
        None => return Ok(None),
    };
    let header_fields = parse_csv_line(header);
    let index = |name: &str| -> Result<usize, Box<dyn std::error::Error>> {
        header_fields
            .iter()
            .position(|field| field == name)
            .ok_or_else(|| format!("{} missing column {name}", path.display()).into())
    };

    let sample_id_index = index("sample_id")?;
    let sample_name_index = index("sample_name")?;
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

    let mut id = String::new();
    let mut name = String::new();
    let mut labels = [String::new(), String::new(), String::new()];
    let mut rows = Vec::new();

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_line(line);
        if fields.len() <= *adc_indexes.iter().max().expect("adc indexes") {
            continue;
        }
        if id.is_empty() {
            id = fields[sample_id_index].clone();
            name = fields[sample_name_index].clone();
            labels = [
                fields[label_indexes[0]].clone(),
                fields[label_indexes[1]].clone(),
                fields[label_indexes[2]].clone(),
            ];
        }
        let mut row = [0_u16; CHANNELS];
        for (channel, field_index) in adc_indexes.iter().enumerate() {
            row[channel] = fields[*field_index].parse::<u16>().unwrap_or(0);
        }
        rows.push(row);
    }

    if id.is_empty() || rows.is_empty() {
        return Ok(None);
    }

    Ok(Some(Capture {
        id,
        name,
        labels,
        rows,
    }))
}

fn is_no_scent(labels: &[String; 3]) -> bool {
    labels.iter().all(|label| {
        label.eq_ignore_ascii_case(NO_SCENT_LABEL)
            || !LABELS.iter().any(|known| known.eq_ignore_ascii_case(label))
    })
}

fn active_label_count(labels: &[String; 3]) -> usize {
    labels
        .iter()
        .filter(|label| LABELS.iter().any(|known| known.eq_ignore_ascii_case(label)))
        .count()
}

fn shuffle<T>(items: &mut [T], rng: &mut Lcg) {
    for index in (1..items.len()).rev() {
        let other = rng.range_usize(0, index + 1);
        items.swap(index, other);
    }
}

fn escape_csv(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
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

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn range_usize(&mut self, min: usize, max: usize) -> usize {
        min + (self.next_u32() as usize % (max - min).max(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture(id: &str, labels: [&str; 3]) -> Capture {
        Capture {
            id: id.to_string(),
            name: id.to_string(),
            labels: labels.map(str::to_string),
            rows: vec![[0; CHANNELS]],
        }
    }

    #[test]
    fn no_scent_requires_no_known_labels() {
        assert!(is_no_scent(&[
            NO_SCENT_LABEL.to_string(),
            NO_SCENT_LABEL.to_string(),
            NO_SCENT_LABEL.to_string(),
        ]));
        assert!(!is_no_scent(&[
            "Citrus".to_string(),
            NO_SCENT_LABEL.to_string(),
            NO_SCENT_LABEL.to_string(),
        ]));
    }

    #[test]
    fn active_label_count_ignores_no_scent_slots() {
        assert_eq!(
            active_label_count(&[
                "Citrus".to_string(),
                NO_SCENT_LABEL.to_string(),
                NO_SCENT_LABEL.to_string(),
            ]),
            1
        );
        assert_eq!(
            active_label_count(&["Citrus".to_string(), "Water".to_string(), "Green".to_string()]),
            3
        );
    }

    #[test]
    fn balancing_repeats_smaller_buckets_to_largest_bucket() {
        let captures = vec![
            capture("single_0", ["Citrus", NO_SCENT_LABEL, NO_SCENT_LABEL]),
            capture("two_0", ["Citrus", "Water", NO_SCENT_LABEL]),
            capture("two_1", ["Citrus", "Green", NO_SCENT_LABEL]),
            capture("three_0", ["Citrus", "Water", "Green"]),
            capture("three_1", ["Citrus", "Water", "Fruity"]),
            capture("three_2", ["Citrus", "Water", "Floral"]),
        ];
        let mut rng = Lcg::new(123);
        let (balanced, summary) =
            balanced_scent_captures(captures, None, &mut rng).expect("balanced");

        assert_eq!(summary.single, 3);
        assert_eq!(summary.two, 3);
        assert_eq!(summary.three, 3);
        assert_eq!(summary.target_per_bucket, 3);
        assert_eq!(balanced.len(), 9);
    }

    #[test]
    fn gap_source_preserves_whole_capture_order() {
        let captures = vec![Capture {
            id: "gap_a".to_string(),
            name: "Gap A".to_string(),
            labels: [
                NO_SCENT_LABEL.to_string(),
                NO_SCENT_LABEL.to_string(),
                NO_SCENT_LABEL.to_string(),
            ],
            rows: vec![[1; CHANNELS], [2; CHANNELS], [3; CHANNELS]],
        }];
        let mut source = GapSource::new(captures);
        let first = source.next_capture();
        let second = source.next_capture();

        assert_eq!(first.rows.iter().map(|row| row[0]).collect::<Vec<_>>(), [1, 2, 3]);
        assert_eq!(second.rows.iter().map(|row| row[0]).collect::<Vec<_>>(), [1, 2, 3]);
    }
}
