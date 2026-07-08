use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_PORT: &str = "/dev/cu.usbmodem21401";
const DEFAULT_BAUD: u32 = 115_200;
const DEFAULT_DURATION_SECS: u64 = 30;
const CHANNEL_COUNT: usize = 9;
const WHEEL_LABELS: [&str; 14] = [
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

#[derive(Debug, PartialEq, Eq)]
struct AdcFrame {
    seq: u64,
    device_ms: u64,
    adc: [u16; CHANNEL_COUNT],
}

#[derive(Debug, PartialEq, Eq)]
struct SampleLabels {
    primary: &'static str,
    secondary: &'static str,
    tertiary: &'static str,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("collector error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("NoseKnows serial collector");
    println!("Close any active PlatformIO serial monitor before starting collection.");

    let sample_name = prompt_required("Sample name")?;
    print_wheel_labels();
    let labels = prompt_sample_labels()?;
    let duration_secs = prompt_u64("Collection seconds", DEFAULT_DURATION_SECS)?;
    let port_name = prompt_with_default("Serial port", DEFAULT_PORT)?;
    let baud = prompt_u32("Baud", DEFAULT_BAUD)?;

    fs::create_dir_all("data/raw")?;
    let (sample_id, output_path) = output_path(Path::new("data/raw"), &sample_name)?;

    println!("Collecting '{sample_name}' from {port_name} at {baud} baud for {duration_secs}s");
    println!(
        "Labels: {}, {}, {}",
        labels.primary, labels.secondary, labels.tertiary
    );
    println!("Writing {}", output_path.display());

    let port = serialport::new(&port_name, baud)
        .timeout(Duration::from_millis(200))
        .open()?;
    let mut reader = BufReader::new(port);
    let output = File::create(&output_path)?;
    let mut writer = BufWriter::new(output);

    writeln!(
        writer,
        "sample_id,sample_name,label_1,label_2,label_3,host_elapsed_ms,host_unix_ms,device_seq,device_ms,adc0,adc1,adc2,adc3,adc4,adc5,adc6,adc7,adc8"
    )?;

    let started = Instant::now();
    let deadline = started + Duration::from_secs(duration_secs);
    let mut line = String::new();
    let mut rows = 0_u64;
    let mut ignored = 0_u64;
    let mut last_progress_second = 0_u64;

    while Instant::now() < deadline {
        line.clear();

        match reader.read_line(&mut line) {
            Ok(0) => continue,
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if let Some(frame) = parse_adc_frame(trimmed) {
                    rows += 1;
                    write_frame(
                        &mut writer,
                        &sample_id,
                        &sample_name,
                        &labels,
                        started,
                        &frame,
                    )?;
                } else {
                    ignored += 1;
                }
            }
            Err(error) if error.kind() == io::ErrorKind::TimedOut => continue,
            Err(error) => return Err(Box::new(error)),
        }

        let elapsed_second = started.elapsed().as_secs();
        if elapsed_second != last_progress_second {
            last_progress_second = elapsed_second;
            print!(
                "\r{:>3}s / {:>3}s, {:>5} rows",
                elapsed_second, duration_secs, rows
            );
            io::stdout().flush()?;
        }
    }

    writer.flush()?;
    println!(
        "\nDone. Wrote {rows} rows to {} ({ignored} non-data lines ignored).",
        output_path.display()
    );

    Ok(())
}

fn prompt_required(label: &str) -> io::Result<String> {
    loop {
        let value = prompt(label)?;
        if !value.trim().is_empty() {
            return Ok(value.trim().to_string());
        }
        println!("{label} is required.");
    }
}

fn print_wheel_labels() {
    println!("Wheel labels:");
    for (index, label) in WHEEL_LABELS.iter().enumerate() {
        println!("  {:>2}. {label}", index + 1);
    }
}

fn prompt_sample_labels() -> io::Result<SampleLabels> {
    Ok(SampleLabels {
        primary: prompt_wheel_label("Primary label")?,
        secondary: prompt_wheel_label("Secondary label")?,
        tertiary: prompt_wheel_label("Tertiary label")?,
    })
}

fn prompt_wheel_label(label: &str) -> io::Result<&'static str> {
    loop {
        let value = prompt(label)?;
        let trimmed = value.trim();

        match parse_wheel_label(trimmed) {
            Some(parsed) => return Ok(parsed),
            None => println!("Enter a label name or number from the wheel list."),
        }
    }
}

fn prompt_with_default(label: &str, default: &str) -> io::Result<String> {
    let value = prompt(&format!("{label} [{default}]"))?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

fn prompt_u64(label: &str, default: u64) -> io::Result<u64> {
    loop {
        let value = prompt(&format!("{label} [{default}]"))?;
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Ok(default);
        }
        match trimmed.parse() {
            Ok(parsed) if parsed > 0 => return Ok(parsed),
            _ => println!("Enter a positive integer."),
        }
    }
}

fn prompt_u32(label: &str, default: u32) -> io::Result<u32> {
    loop {
        let value = prompt(&format!("{label} [{default}]"))?;
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Ok(default);
        }
        match trimmed.parse() {
            Ok(parsed) if parsed > 0 => return Ok(parsed),
            _ => println!("Enter a positive integer."),
        }
    }
}

fn prompt(label: &str) -> io::Result<String> {
    print!("{label}: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input)
}

fn parse_wheel_label(value: &str) -> Option<&'static str> {
    if let Ok(index) = value.parse::<usize>() {
        return WHEEL_LABELS.get(index.checked_sub(1)?).copied();
    }

    let normalized = normalize_label(value);
    WHEEL_LABELS
        .iter()
        .copied()
        .find(|label| normalize_label(label) == normalized)
}

fn parse_adc_frame(line: &str) -> Option<AdcFrame> {
    let mut fields = line.split(',');

    if fields.next()? != "NK_ADC" {
        return None;
    }

    let seq = fields.next()?.parse().ok()?;
    let device_ms = fields.next()?.parse().ok()?;
    let mut adc = [0_u16; CHANNEL_COUNT];

    for value in &mut adc {
        *value = fields.next()?.parse().ok()?;
    }

    if fields.next().is_some() {
        return None;
    }

    Some(AdcFrame {
        seq,
        device_ms,
        adc,
    })
}

fn write_frame<W: Write>(
    writer: &mut W,
    sample_id: &str,
    sample_name: &str,
    labels: &SampleLabels,
    started: Instant,
    frame: &AdcFrame,
) -> io::Result<()> {
    write!(
        writer,
        "{},{},{},{},{},{},{},{},{}",
        csv_escape(sample_id),
        csv_escape(sample_name),
        csv_escape(labels.primary),
        csv_escape(labels.secondary),
        csv_escape(labels.tertiary),
        started.elapsed().as_millis(),
        unix_ms(),
        frame.seq,
        frame.device_ms
    )?;

    for value in frame.adc {
        write!(writer, ",{value}")?;
    }

    writeln!(writer)
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn output_path(base_dir: &Path, sample_name: &str) -> io::Result<(String, PathBuf)> {
    let unix_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let sample_slug = sanitize_filename(sample_name);
    let mut sample_id = format!("{unix_secs}_{sample_slug}");
    let mut candidate = base_dir.join(format!("{sample_id}.csv"));
    let mut suffix = 1_u32;

    while candidate.exists() {
        sample_id = format!("{unix_secs}_{sample_slug}_{suffix}");
        candidate = base_dir.join(format!("{sample_id}.csv"));
        suffix += 1;
    }

    Ok((sample_id, candidate))
}

fn sanitize_filename(value: &str) -> String {
    let mut output = String::new();

    for character in value.trim().chars() {
        if character.is_ascii_alphanumeric() {
            output.push(character.to_ascii_lowercase());
        } else if character == '-' || character == '_' || character.is_whitespace() {
            if !output.ends_with('_') {
                output.push('_');
            }
        }
    }

    let output = output.trim_matches('_').to_string();
    if output.is_empty() {
        "sample".to_string()
    } else {
        output
    }
}

fn normalize_label(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace() && *character != '-' && *character != '_')
        .flat_map(char::to_lowercase)
        .collect()
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_adc_frame() {
        let frame = parse_adc_frame("NK_ADC,2399,240510,2687,745,730,536,565,671,647,670,148")
            .expect("valid frame");

        assert_eq!(frame.seq, 2399);
        assert_eq!(frame.device_ms, 240510);
        assert_eq!(frame.adc, [2687, 745, 730, 536, 565, 671, 647, 670, 148]);
    }

    #[test]
    fn ignores_non_adc_lines() {
        assert_eq!(parse_adc_frame("NK_HEADER,seq,ms,adc0"), None);
    }

    #[test]
    fn rejects_incomplete_frames() {
        assert_eq!(parse_adc_frame("NK_ADC,1,100,10,20"), None);
    }

    #[test]
    fn parses_wheel_labels_by_name_and_number() {
        assert_eq!(parse_wheel_label("Soft Amber"), Some("Soft Amber"));
        assert_eq!(parse_wheel_label("soft amber"), Some("Soft Amber"));
        assert_eq!(parse_wheel_label("soft-amber"), Some("Soft Amber"));
        assert_eq!(parse_wheel_label("9"), Some("Dry Woods"));
        assert_eq!(parse_wheel_label("15"), None);
    }

    #[test]
    fn sanitizes_sample_names_for_filenames() {
        assert_eq!(
            sanitize_filename(" Soft Amber / test #1 "),
            "soft_amber_test_1"
        );
    }

    #[test]
    fn escapes_csv_values() {
        assert_eq!(csv_escape("plain"), "plain");
        assert_eq!(csv_escape("a,b \"c\""), "\"a,b \"\"c\"\"\"");
    }
}
