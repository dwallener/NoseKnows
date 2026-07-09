use std::env;
use std::fs;
use std::path::PathBuf;

const DEFAULT_OUT: &str = "data/plots/scent_sensor_response.svg";
const MAX_ADC: f32 = 4095.0;

const SENSORS: [&str; 8] = [
    "adc0 MQ-2",
    "adc1 MQ-3",
    "adc2 MQ-5",
    "adc3 MQ-6",
    "adc4 MQ-7",
    "adc5 MQ-8",
    "adc6 MQ-9",
    "adc7 MQ-135",
];

const RESPONSES: [(&str, [f32; 8]); 14] = [
    ("Floral", [0.0, 4095.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1500.0]),
    (
        "Soft Floral",
        [0.0, 2200.0, 0.0, 0.0, 2100.0, 0.0, 0.0, 2200.0],
    ),
    (
        "Floral Amber",
        [1100.0, 4095.0, 0.0, 0.0, 3400.0, 0.0, 3000.0, 3800.0],
    ),
    (
        "Amber",
        [2200.0, 2500.0, 0.0, 0.0, 3900.0, 0.0, 3800.0, 4095.0],
    ),
    (
        "Soft Amber",
        [900.0, 1200.0, 0.0, 0.0, 3700.0, 0.0, 0.0, 3500.0],
    ),
    (
        "Woody Amber",
        [3500.0, 1800.0, 0.0, 0.0, 3000.0, 0.0, 3400.0, 4000.0],
    ),
    ("Woods", [3100.0, 900.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3200.0]),
    (
        "Mossy Woods",
        [2800.0, 2900.0, 0.0, 0.0, 3100.0, 0.0, 0.0, 3000.0],
    ),
    (
        "Dry Woods",
        [2000.0, 1100.0, 0.0, 0.0, 3900.0, 0.0, 3600.0, 3800.0],
    ),
    (
        "Aromatic",
        [2400.0, 3600.0, 0.0, 2900.0, 2900.0, 0.0, 0.0, 3400.0],
    ),
    (
        "Citrus",
        [3800.0, 2000.0, 3500.0, 3700.0, 0.0, 0.0, 0.0, 2200.0],
    ),
    ("Water", [0.0, 800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1800.0]),
    ("Green", [0.0, 2500.0, 0.0, 600.0, 0.0, 0.0, 0.0, 3000.0]),
    ("Fruity", [1200.0, 3800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2200.0]),
];

const PEAK_SECONDS: [f32; 14] = [
    9.0, 9.0, 9.0, 12.0, 11.0, 12.0, 10.0, 11.0, 11.0, 9.0, 5.0, 6.0, 5.0, 7.0,
];

const DECAY_T90: [[f32; 8]; 14] = [
    [0.0, 38.0, 0.0, 0.0, 0.0, 0.0, 0.0, 38.0],
    [0.0, 40.0, 0.0, 0.0, 40.0, 0.0, 0.0, 40.0],
    [30.0, 8.0, 0.0, 0.0, 60.0, 0.0, 55.0, 60.0],
    [180.0, 180.0, 0.0, 0.0, 180.0, 0.0, 180.0, 180.0],
    [70.0, 70.0, 0.0, 0.0, 95.0, 0.0, 0.0, 85.0],
    [190.0, 190.0, 0.0, 0.0, 190.0, 0.0, 190.0, 190.0],
    [75.0, 75.0, 0.0, 0.0, 0.0, 0.0, 0.0, 75.0],
    [120.0, 120.0, 0.0, 0.0, 120.0, 0.0, 0.0, 120.0],
    [135.0, 135.0, 0.0, 0.0, 135.0, 0.0, 135.0, 135.0],
    [35.0, 40.0, 0.0, 10.0, 90.0, 0.0, 0.0, 90.0],
    [7.0, 7.0, 7.0, 7.0, 0.0, 0.0, 0.0, 7.0],
    [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0],
    [0.0, 6.5, 0.0, 6.5, 0.0, 0.0, 0.0, 6.5],
    [13.5, 13.5, 0.0, 0.0, 0.0, 0.0, 0.0, 13.5],
];

fn main() {
    if let Err(error) = run() {
        eprintln!("response_plot error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let output_path = parse_args()?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, render_svg())?;
    println!("Wrote response plot to {}", output_path.display());
    Ok(())
}

fn parse_args() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut output_path = PathBuf::from(DEFAULT_OUT);
    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--out" => {
                index += 1;
                output_path = PathBuf::from(args.get(index).ok_or("--out requires a path")?);
            }
            "--help" | "-h" => {
                println!("Usage: cargo run --bin response_plot -- [--out data/plots/scent_sensor_response.svg]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        index += 1;
    }
    Ok(output_path)
}

fn render_svg() -> String {
    let left = 150.0;
    let top = 82.0;
    let cell_w = 92.0;
    let cell_h = 30.0;
    let grid_h = cell_h * RESPONSES.len() as f32;
    let grid_gap = 76.0;
    let width = left + cell_w * SENSORS.len() as f32 + 34.0;
    let height = top + grid_h * 3.0 + grid_gap * 2.0 + 96.0;

    let mut svg = String::new();
    push_line(
        &mut svg,
        &format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0}" height="{height:.0}" viewBox="0 0 {width:.0} {height:.0}">"##
        ),
    );
    push_line(
        &mut svg,
        r##"<rect width="100%" height="100%" fill="#f6f8f8"/>"##,
    );
    push_line(
        &mut svg,
        r##"<text x="20" y="32" font-family="system-ui,-apple-system,sans-serif" font-size="22" font-weight="700" fill="#263235">Synthetic scent response model</text>"##,
    );
    push_line(
        &mut svg,
        r##"<text x="20" y="54" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="12" fill="#657073">Peak ADC targets, 8-bin peak code, and timing model by fragrance label and active sensor. MQ-4 placeholder omitted.</text>"##,
    );

    render_headers(&mut svg, left, top - 10.0, cell_w);
    render_peak_grid(&mut svg, left, top, cell_w, cell_h);

    let quantized_top = top + grid_h + grid_gap;
    render_headers(&mut svg, left, quantized_top - 10.0, cell_w);
    render_quantized_grid(&mut svg, left, quantized_top, cell_w, cell_h);

    let timing_top = quantized_top + grid_h + grid_gap;
    render_headers(&mut svg, left, timing_top - 10.0, cell_w);
    render_timing_grid(&mut svg, left, timing_top, cell_w, cell_h);

    let legend_x = left;
    let legend_y = height - 34.0;
    for step in 0..=10 {
        let strength = step as f32 / 10.0;
        push_line(
            &mut svg,
            &format!(
                r##"<rect x="{:.1}" y="{legend_y:.1}" width="28" height="10" fill="{}"/>"##,
                legend_x + step as f32 * 28.0,
                heat_color(strength)
            ),
        );
    }
    push_line(
        &mut svg,
        &format!(
            r##"<text x="{legend_x:.1}" y="{:.1}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11" fill="#657073">0</text>"##,
            legend_y + 26.0
        ),
    );
    push_line(
        &mut svg,
        &format!(
            r##"<text x="{:.1}" y="{:.1}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11" text-anchor="end" fill="#657073">4095</text>"##,
            legend_x + 308.0,
            legend_y + 26.0
        ),
    );
    push_line(&mut svg, "</svg>");
    svg
}

fn render_headers(svg: &mut String, left: f32, y: f32, cell_w: f32) {
    for (sensor, name) in SENSORS.iter().enumerate() {
        let x = left + sensor as f32 * cell_w + cell_w / 2.0;
        push_line(
            svg,
            &format!(
                r##"<text x="{x:.1}" y="{y:.1}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11" text-anchor="middle" fill="#526064">{}</text>"##,
                xml_escape(name)
            ),
        );
    }
}

fn render_peak_grid(svg: &mut String, left: f32, top: f32, cell_w: f32, cell_h: f32) {
    render_title(
        svg,
        top - 38.0,
        "Peak target ADC",
        "12-bit synthetic peak value, clipped at 4095",
    );
    render_grid(svg, left, top, cell_w, cell_h, |row, sensor| {
        let value = RESPONSES[row].1[sensor];
        if value <= 0.0 {
            Cell::empty()
        } else {
            Cell::new(
                format!("{value:.0}"),
                heat_color((value / MAX_ADC).clamp(0.0, 1.0)),
                value / MAX_ADC,
            )
        }
    });
}

fn render_quantized_grid(svg: &mut String, left: f32, top: f32, cell_w: f32, cell_h: f32) {
    render_title(
        svg,
        top - 38.0,
        "Quantized peak code",
        "8 bins over 0..4095; inactive sensors shown as dash",
    );
    render_grid(svg, left, top, cell_w, cell_h, |row, sensor| {
        let value = RESPONSES[row].1[sensor];
        if value <= 0.0 {
            Cell::empty()
        } else {
            let bin = quantize_8(value);
            Cell::new(
                bin.to_string(),
                heat_color(bin as f32 / 7.0),
                bin as f32 / 7.0,
            )
        }
    });
}

fn render_timing_grid(svg: &mut String, left: f32, top: f32, cell_w: f32, cell_h: f32) {
    render_title(
        svg,
        top - 38.0,
        "Timing model",
        "cell text is peak seconds from t0 / decay T90 seconds",
    );
    render_grid(svg, left, top, cell_w, cell_h, |row, sensor| {
        if RESPONSES[row].1[sensor] <= 0.0 {
            Cell::empty()
        } else {
            let peak = PEAK_SECONDS[row];
            let decay = DECAY_T90[row][sensor];
            let timing_strength = (peak / 12.0 * 0.35 + decay / 190.0 * 0.65).clamp(0.0, 1.0);
            Cell::new(
                format!("{peak:.0}/{decay:.0}s"),
                time_color(timing_strength),
                timing_strength,
            )
        }
    });
}

fn render_title(svg: &mut String, y: f32, title: &str, subtitle: &str) {
    push_line(
        svg,
        &format!(
            r##"<text x="20" y="{y:.1}" font-family="system-ui,-apple-system,sans-serif" font-size="17" font-weight="700" fill="#263235">{}</text>"##,
            xml_escape(title)
        ),
    );
    push_line(
        svg,
        &format!(
            r##"<text x="20" y="{:.1}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11" fill="#657073">{}</text>"##,
            y + 17.0,
            xml_escape(subtitle)
        ),
    );
}

fn render_grid<F>(svg: &mut String, left: f32, top: f32, cell_w: f32, cell_h: f32, mut cell: F)
where
    F: FnMut(usize, usize) -> Cell,
{
    for (row, (label, _)) in RESPONSES.iter().enumerate() {
        let y = top + row as f32 * cell_h;
        push_line(
            svg,
            &format!(
                r##"<text x="20" y="{:.1}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="12" fill="#263235">{}</text>"##,
                y + 20.0,
                xml_escape(label)
            ),
        );
        for sensor in 0..SENSORS.len() {
            let x = left + sensor as f32 * cell_w;
            let cell = cell(row, sensor);
            let text_fill = if cell.strength > 0.62 {
                "#ffffff"
            } else {
                "#263235"
            };
            push_line(
                svg,
                &format!(
                    r##"<rect x="{x:.1}" y="{y:.1}" width="{:.1}" height="{:.1}" fill="{}" stroke="#d9e0e0" stroke-width="1"/>"##,
                    cell_w - 2.0,
                    cell_h - 2.0,
                    cell.fill
                ),
            );
            push_line(
                svg,
                &format!(
                    r##"<text x="{:.1}" y="{:.1}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11" text-anchor="middle" fill="{text_fill}">{}</text>"##,
                    x + cell_w / 2.0 - 1.0,
                    y + 19.0,
                    cell.text
                ),
            );
        }
    }
}

struct Cell {
    text: String,
    fill: String,
    strength: f32,
}

impl Cell {
    fn new(text: String, fill: String, strength: f32) -> Self {
        Self {
            text,
            fill,
            strength,
        }
    }

    fn empty() -> Self {
        Self::new(String::from("-"), String::from("#eef2f2"), 0.0)
    }
}

fn quantize_8(value: f32) -> usize {
    ((value / MAX_ADC).clamp(0.0, 1.0) * 8.0).floor().min(7.0) as usize
}

fn heat_color(strength: f32) -> String {
    let strength = strength.clamp(0.0, 1.0);
    let low = (236.0, 242.0, 242.0);
    let mid = (41.0, 148.0, 139.0);
    let high = (160.0, 75.0, 170.0);
    let (a, b, t) = if strength < 0.55 {
        (low, mid, strength / 0.55)
    } else {
        (mid, high, (strength - 0.55) / 0.45)
    };
    let blend = |left: f32, right: f32| (left + (right - left) * t).round() as u8;
    format!(
        "#{:02x}{:02x}{:02x}",
        blend(a.0, b.0),
        blend(a.1, b.1),
        blend(a.2, b.2)
    )
}

fn time_color(strength: f32) -> String {
    let strength = strength.clamp(0.0, 1.0);
    let low = (236.0, 242.0, 242.0);
    let mid = (217.0, 154.0, 43.0);
    let high = (139.0, 86.0, 62.0);
    let (a, b, t) = if strength < 0.5 {
        (low, mid, strength / 0.5)
    } else {
        (mid, high, (strength - 0.5) / 0.5)
    };
    let blend = |left: f32, right: f32| (left + (right - left) * t).round() as u8;
    format!(
        "#{:02x}{:02x}{:02x}",
        blend(a.0, b.0),
        blend(a.1, b.1),
        blend(a.2, b.2)
    )
}

fn push_line(output: &mut String, line: &str) {
    output.push_str(line);
    output.push('\n');
}

fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
