use std::env;
use std::fs;
use std::path::PathBuf;

const DEFAULT_OUT: &str = "data/plots/scent_sensor_response_curves.svg";
const MAX_ADC: f32 = 4095.0;
const BASELINE: f32 = 220.0;
const LN_10: f32 = 2.302_585_1;
const TOTAL_SECS: f32 = 90.0;
const STEP_SECS: f32 = 0.5;

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

const SENSOR_COLORS: [&str; 8] = [
    "#2f77b4", "#d17a22", "#4b9b57", "#c34f6a", "#8467a9", "#7b6b56", "#2a9d8f", "#c6a336",
];

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

const PEAKS: [[f32; 8]; 14] = [
    [0.0, 4095.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1500.0],
    [0.0, 2200.0, 0.0, 0.0, 2100.0, 0.0, 0.0, 2200.0],
    [1100.0, 4095.0, 0.0, 0.0, 3400.0, 0.0, 3000.0, 3800.0],
    [2200.0, 2500.0, 0.0, 0.0, 3900.0, 0.0, 3800.0, 4095.0],
    [900.0, 1200.0, 0.0, 0.0, 3700.0, 0.0, 0.0, 3500.0],
    [3500.0, 1800.0, 0.0, 0.0, 3000.0, 0.0, 3400.0, 4000.0],
    [3100.0, 900.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3200.0],
    [2800.0, 2900.0, 0.0, 0.0, 3100.0, 0.0, 0.0, 3000.0],
    [2000.0, 1100.0, 0.0, 0.0, 3900.0, 0.0, 3600.0, 3800.0],
    [2400.0, 3600.0, 0.0, 2900.0, 2900.0, 0.0, 0.0, 3400.0],
    [3800.0, 2000.0, 3500.0, 3700.0, 0.0, 0.0, 0.0, 2200.0],
    [0.0, 800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1800.0],
    [0.0, 2500.0, 0.0, 600.0, 0.0, 0.0, 0.0, 3000.0],
    [1200.0, 3800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2200.0],
];

const T90: [[f32; 8]; 14] = [
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

const RISE_TAU: [f32; 14] = [
    2.0, 2.0, 1.8, 4.8, 4.0, 5.0, 3.6, 4.0, 4.2, 1.2, 0.45, 0.8, 0.6, 1.0,
];

const EXPOSURE_SECS: [f32; 14] = [
    9.0, 9.0, 9.0, 12.0, 11.0, 12.0, 10.0, 11.0, 11.0, 9.0, 5.0, 6.0, 5.0, 7.0,
];

const RESIDUAL: [[f32; 8]; 14] = [
    [0.0; 8],
    [0.0; 8],
    [0.0; 8],
    [300.0; 8],
    [0.0, 0.0, 0.0, 0.0, 180.0, 0.0, 0.0, 0.0],
    [220.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 260.0],
    [0.0; 8],
    [0.0; 8],
    [0.0, 0.0, 0.0, 0.0, 280.0, 0.0, 220.0, 0.0],
    [0.0; 8],
    [0.0; 8],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0],
    [0.0; 8],
    [0.0; 8],
];

fn main() {
    if let Err(error) = run() {
        eprintln!("response_curves error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let output_path = parse_args()?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, render_svg())?;
    println!("Wrote response curves to {}", output_path.display());
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
                println!(
                    "Usage: cargo run --bin response_curves -- [--out data/plots/scent_sensor_response_curves.svg]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        index += 1;
    }
    Ok(output_path)
}

fn render_svg() -> String {
    let panel_w = 430.0;
    let panel_h = 160.0;
    let gap_x = 28.0;
    let gap_y = 38.0;
    let margin_x = 34.0;
    let margin_top = 96.0;
    let width = margin_x * 2.0 + panel_w * 2.0 + gap_x;
    let rows = 7.0;
    let height = margin_top + rows * panel_h + (rows - 1.0) * gap_y + 74.0;

    let mut svg = String::new();
    line(
        &mut svg,
        &format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0}" height="{height:.0}" viewBox="0 0 {width:.0} {height:.0}">"##
        ),
    );
    line(
        &mut svg,
        r##"<rect width="100%" height="100%" fill="#f6f8f8"/>"##,
    );
    line(
        &mut svg,
        r##"<text x="28" y="34" font-family="system-ui,-apple-system,sans-serif" font-size="22" font-weight="700" fill="#263235">Synthetic scent response curves</text>"##,
    );
    line(
        &mut svg,
        r##"<text x="28" y="56" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="12" fill="#657073">One panel per fragrance note. Lines show idealized ADC response over 90s for active sensors; MQ-4 placeholder omitted.</text>"##,
    );
    render_legend(&mut svg, 28.0, 72.0);

    for label_index in 0..LABELS.len() {
        let col = label_index % 2;
        let row = label_index / 2;
        let x = margin_x + col as f32 * (panel_w + gap_x);
        let y = margin_top + row as f32 * (panel_h + gap_y);
        render_panel(&mut svg, label_index, x, y, panel_w, panel_h);
    }

    line(&mut svg, "</svg>");
    svg
}

fn render_legend(svg: &mut String, x: f32, y: f32) {
    let mut cursor = x;
    for (sensor, name) in SENSORS.iter().enumerate() {
        line(
            svg,
            &format!(
                r##"<line x1="{cursor:.1}" y1="{y:.1}" x2="{:.1}" y2="{y:.1}" stroke="{}" stroke-width="2.4"/>"##,
                cursor + 18.0,
                SENSOR_COLORS[sensor]
            ),
        );
        line(
            svg,
            &format!(
                r##"<text x="{:.1}" y="{:.1}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="10" fill="#526064">{}</text>"##,
                cursor + 22.0,
                y + 3.5,
                xml_escape(name)
            ),
        );
        cursor += 103.0;
        if sensor == 3 {
            cursor = x;
        }
    }
}

fn render_panel(svg: &mut String, label_index: usize, x: f32, y: f32, w: f32, h: f32) {
    let plot_x = x + 48.0;
    let plot_y = y + 24.0;
    let plot_w = w - 62.0;
    let plot_h = h - 44.0;

    line(
        svg,
        &format!(
            r##"<rect x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}" rx="7" fill="#ffffff" stroke="#d7e0e3"/>"##
        ),
    );
    line(
        svg,
        &format!(
            r##"<text x="{:.1}" y="{:.1}" font-family="system-ui,-apple-system,sans-serif" font-size="14" font-weight="700" fill="#263235">{}</text>"##,
            x + 12.0,
            y + 17.0,
            xml_escape(LABELS[label_index])
        ),
    );

    for tick in [0.0, 1024.0, 2048.0, 3072.0, 4095.0] {
        let ty = plot_y + plot_h - tick / MAX_ADC * plot_h;
        line(
            svg,
            &format!(
                r##"<line x1="{plot_x:.1}" y1="{ty:.1}" x2="{:.1}" y2="{ty:.1}" stroke="#e8edef" stroke-width="1"/>"##,
                plot_x + plot_w
            ),
        );
    }
    for second in [0.0, 30.0, 60.0, 90.0] {
        let tx = plot_x + second / TOTAL_SECS * plot_w;
        line(
            svg,
            &format!(
                r##"<line x1="{tx:.1}" y1="{plot_y:.1}" x2="{tx:.1}" y2="{:.1}" stroke="#eef2f3" stroke-width="1"/>"##,
                plot_y + plot_h
            ),
        );
        line(
            svg,
            &format!(
                r##"<text x="{tx:.1}" y="{:.1}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="9" text-anchor="middle" fill="#7a8589">{second:.0}</text>"##,
                plot_y + plot_h + 13.0
            ),
        );
    }

    line(
        svg,
        &format!(
            r##"<polyline points="{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}" fill="none" stroke="#97a3a8" stroke-width="1.2"/>"##,
            plot_x,
            plot_y,
            plot_x,
            plot_y + plot_h,
            plot_x + plot_w,
            plot_y + plot_h
        ),
    );
    line(
        svg,
        &format!(
            r##"<text x="{:.1}" y="{:.1}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="9" text-anchor="end" fill="#7a8589">4095</text>"##,
            plot_x - 5.0,
            plot_y + 4.0
        ),
    );
    line(
        svg,
        &format!(
            r##"<text x="{:.1}" y="{:.1}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="9" text-anchor="end" fill="#7a8589">0</text>"##,
            plot_x - 5.0,
            plot_y + plot_h + 3.0
        ),
    );

    for sensor in 0..SENSORS.len() {
        if PEAKS[label_index][sensor] <= 0.0 {
            continue;
        }
        let points = curve_points(label_index, sensor, plot_x, plot_y, plot_w, plot_h);
        line(
            svg,
            &format!(
                r##"<polyline points="{points}" fill="none" stroke="{}" stroke-width="2.0" stroke-linejoin="round" stroke-linecap="round" opacity="0.9"><title>{}: {}</title></polyline>"##,
                SENSOR_COLORS[sensor],
                xml_escape(LABELS[label_index]),
                xml_escape(SENSORS[sensor])
            ),
        );
    }
}

fn curve_points(label_index: usize, sensor: usize, x: f32, y: f32, w: f32, h: f32) -> String {
    let mut points = Vec::new();
    let steps = (TOTAL_SECS / STEP_SECS).round() as usize;
    for step in 0..=steps {
        let seconds = step as f32 * STEP_SECS;
        let value = response_value(label_index, sensor, seconds);
        let px = x + seconds / TOTAL_SECS * w;
        let py = y + h - value / MAX_ADC * h;
        points.push(format!("{px:.1},{py:.1}"));
    }
    points.join(" ")
}

fn response_value(label_index: usize, sensor: usize, seconds: f32) -> f32 {
    let peak = PEAKS[label_index][sensor];
    if peak <= 0.0 {
        return BASELINE;
    }
    let amplitude = (peak - BASELINE).max(0.0);
    let residual = RESIDUAL[label_index][sensor].min(amplitude * 0.8);
    let exposure = EXPOSURE_SECS[label_index];
    let signal = if seconds <= exposure {
        let tau = RISE_TAU[label_index].max(0.1);
        let numerator = 1.0 - (-seconds / tau).exp();
        let denominator = (1.0 - (-exposure / tau).exp()).max(0.001);
        amplitude * numerator / denominator
    } else {
        let elapsed = seconds - exposure;
        let tau = (T90[label_index][sensor] / LN_10).max(0.1);
        residual + (amplitude - residual) * (-elapsed / tau).exp()
    };
    (BASELINE + signal).clamp(0.0, MAX_ADC)
}

fn line(svg: &mut String, text: &str) {
    svg.push_str(text);
    svg.push('\n');
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
