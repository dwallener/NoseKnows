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
    let width = left + cell_w * SENSORS.len() as f32 + 34.0;
    let height = top + cell_h * RESPONSES.len() as f32 + 72.0;

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
        r##"<text x="20" y="54" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="12" fill="#657073">Peak 12-bit ADC target by fragrance label and active sensor; darker cells are stronger responses. MQ-4 placeholder omitted.</text>"##,
    );

    for (sensor, name) in SENSORS.iter().enumerate() {
        let x = left + sensor as f32 * cell_w + cell_w / 2.0;
        push_line(
            &mut svg,
            &format!(
                r##"<text x="{x:.1}" y="72" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11" text-anchor="middle" fill="#526064">{}</text>"##,
                xml_escape(name)
            ),
        );
    }

    for (row, (label, values)) in RESPONSES.iter().enumerate() {
        let y = top + row as f32 * cell_h;
        push_line(
            &mut svg,
            &format!(
                r##"<text x="20" y="{:.1}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="12" fill="#263235">{}</text>"##,
                y + 20.0,
                xml_escape(label)
            ),
        );
        for (sensor, value) in values.iter().enumerate() {
            let x = left + sensor as f32 * cell_w;
            let strength = (*value / MAX_ADC).clamp(0.0, 1.0);
            let fill = heat_color(strength);
            let text_fill = if strength > 0.62 {
                "#ffffff"
            } else {
                "#263235"
            };
            push_line(
                &mut svg,
                &format!(
                    r##"<rect x="{x:.1}" y="{y:.1}" width="{:.1}" height="{:.1}" fill="{fill}" stroke="#d9e0e0" stroke-width="1"/>"##,
                    cell_w - 2.0,
                    cell_h - 2.0
                ),
            );
            let text = if *value <= 0.0 {
                String::from("-")
            } else {
                format!("{value:.0}")
            };
            push_line(
                &mut svg,
                &format!(
                    r##"<text x="{:.1}" y="{:.1}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11" text-anchor="middle" fill="{text_fill}">{text}</text>"##,
                    x + cell_w / 2.0 - 1.0,
                    y + 19.0
                ),
            );
        }
    }

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
