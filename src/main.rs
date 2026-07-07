use std::cmp::Ordering;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{SystemTime, UNIX_EPOCH};

const CATEGORIES: [&str; 14] = [
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

#[derive(Clone, Copy)]
struct Activation {
    name: &'static str,
    logit: f64,
    strength: f64,
}

fn main() -> std::io::Result<()> {
    let listener = bind_first_available(7878, 7899)?;
    let address = listener.local_addr()?;
    println!("NoseKnows demo running at http://{address}");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(err) = handle_connection(stream) {
                    eprintln!("request failed: {err}");
                }
            }
            Err(err) => eprintln!("connection failed: {err}"),
        }
    }

    Ok(())
}

fn bind_first_available(start_port: u16, end_port: u16) -> std::io::Result<TcpListener> {
    let mut last_error = None;

    for port in start_port..=end_port {
        match TcpListener::bind(("127.0.0.1", port)) {
            Ok(listener) => return Ok(listener),
            Err(err) => last_error = Some(err),
        }
    }

    Err(last_error.unwrap_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::AddrNotAvailable, "no candidate ports")
    }))
}

fn handle_connection(mut stream: TcpStream) -> std::io::Result<()> {
    let mut buffer = [0; 2048];
    let bytes_read = stream.read(&mut buffer)?;
    let request = String::from_utf8_lossy(&buffer[..bytes_read]);
    let path = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/");

    match path {
        "/" => respond(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            APP_HTML.as_bytes(),
        ),
        "/state" => {
            let body = random_state_json();
            respond(
                &mut stream,
                "200 OK",
                "application/json; charset=utf-8",
                body.as_bytes(),
            )
        }
        _ => respond(
            &mut stream,
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"not found",
        ),
    }
}

fn respond(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let headers = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes())?;
    stream.write_all(body)
}

fn random_state_json() -> String {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let mut rng = XorShift64::new(seed ^ 0x9e37_79b9_7f4a_7c15);
    let mut logits: Vec<Activation> = CATEGORIES
        .iter()
        .map(|name| Activation {
            name,
            logit: rng.normalish_logit(),
            strength: 0.0,
        })
        .collect();

    logits.sort_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap_or(Ordering::Equal));
    let mut top: Vec<Activation> = logits.into_iter().take(3).collect();
    let max_logit = top
        .iter()
        .map(|item| item.logit)
        .fold(f64::NEG_INFINITY, f64::max);
    let temperature = 0.85;
    let exp_sum: f64 = top
        .iter()
        .map(|item| ((item.logit - max_logit) / temperature).exp())
        .sum();

    for item in &mut top {
        let probability = ((item.logit - max_logit) / temperature).exp() / exp_sum;
        item.strength = probability.sqrt().clamp(0.08, 1.0);
    }

    let items = top
        .iter()
        .map(|item| {
            format!(
                "{{\"name\":\"{}\",\"logit\":{:.3},\"strength\":{:.3}}}",
                item.name, item.logit, item.strength
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "{{\"generated_at_ms\":{},\"activations\":[{}]}}",
        seed / 1_000_000,
        items
    )
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_f64(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64) / (u64::MAX as f64)
    }

    fn normalish_logit(&mut self) -> f64 {
        let a = self.next_f64().clamp(0.000_001, 0.999_999);
        let b = self.next_f64().clamp(0.000_001, 0.999_999);
        ((a / (1.0 - a)).ln() * 0.65) + (b * 1.8)
    }
}

const APP_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NoseKnows Wheel Demo</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1f2428;
      --muted: #68727d;
      --panel: #ffffff;
      --line: #d8dee5;
      --background: #f3f5f7;
      --accent: #0e7c86;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--background);
      color: var(--ink);
    }

    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0;
      display: grid;
      grid-template-columns: minmax(340px, 1fr) 340px;
      gap: 28px;
      align-items: start;
    }

    .stage {
      min-width: 0;
    }

    .appbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      letter-spacing: 0;
    }

    .status {
      color: var(--muted);
      font-size: 14px;
      margin-top: 5px;
    }

    button {
      border: 1px solid #0b6971;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 700;
      min-width: 104px;
      height: 40px;
      padding: 0 16px;
      cursor: pointer;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.12);
    }

    button:hover {
      background: #0c717a;
    }

    .wheel-wrap {
      position: relative;
      width: min(620px, 100%);
      aspect-ratio: 1 / 1;
      margin: 0 auto;
      border-radius: 50%;
      background: radial-gradient(circle at 50% 48%, #ffffff 0 17%, #eef2f4 17.4% 18.8%, #ffffff 19.2% 100%);
      box-shadow: 0 18px 42px rgba(31, 36, 40, 0.12);
      overflow: visible;
    }

    .wheel-overlay {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }

    .base-sector {
      opacity: 0.24;
      stroke: rgba(255, 255, 255, 0.96);
      stroke-width: 1.35;
    }

    .sector {
      opacity: 0;
      stroke: rgba(255, 255, 255, 0.92);
      stroke-width: 1.55;
      transition: opacity 520ms ease, filter 520ms ease, stroke-width 520ms ease, transform 520ms ease;
      transform-box: fill-box;
      transform-origin: center;
    }

    .sector.active {
      opacity: var(--sector-opacity, 0.85);
      filter: drop-shadow(0 0 calc(var(--sector-strength, 0.5) * 13px) rgba(255, 255, 255, 0.86));
      stroke-width: 1.9;
      transform: scale(calc(1 + var(--sector-strength, 0.5) * 0.018));
    }

    .wheel-rim {
      fill: none;
      stroke: #cbd3d9;
      stroke-width: 0.8;
      opacity: 0.78;
    }

    .wheel-tick {
      stroke: #cbd3d9;
      stroke-width: 0.62;
      stroke-linecap: round;
      opacity: 0.86;
    }

    .wheel-center {
      fill: rgba(255, 255, 255, 0.9);
      stroke: #e2e7eb;
      stroke-width: 0.5;
    }

    .center-title {
      font-size: 2.35px;
      fill: #9da7af;
      font-weight: 800;
      letter-spacing: 0.16px;
      text-anchor: middle;
    }

    .center-subtitle {
      font-size: 3.35px;
      fill: #c0c7cd;
      letter-spacing: 0;
      text-anchor: middle;
    }

    .wheel-label {
      fill: #7f8991;
      font-size: 2.08px;
      font-weight: 700;
      letter-spacing: 0;
      text-anchor: middle;
      dominant-baseline: middle;
      pointer-events: none;
    }

    .readout {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 12px 28px rgba(31, 36, 40, 0.08);
    }

    .readout h2 {
      margin: 0 0 14px;
      font-size: 16px;
      letter-spacing: 0;
    }

    .sensor-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-bottom: 18px;
    }

    .sensor {
      min-height: 50px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #f8fafb;
    }

    .sensor span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 5px;
    }

    .sensor strong {
      font-size: 16px;
      font-variant-numeric: tabular-nums;
    }

    .rank {
      display: grid;
      grid-template-columns: 28px 1fr 54px;
      gap: 10px;
      align-items: center;
      padding: 12px 0;
      border-top: 1px solid var(--line);
    }

    .rank:first-of-type {
      border-top: 0;
    }

    .swatch {
      width: 18px;
      height: 18px;
      border-radius: 50%;
      border: 1px solid rgba(0, 0, 0, 0.16);
    }

    .bar {
      height: 8px;
      border-radius: 999px;
      background: #e4e9ed;
      overflow: hidden;
      margin-top: 6px;
    }

    .bar > div {
      height: 100%;
      width: calc(var(--value) * 100%);
      background: currentColor;
      transition: width 520ms ease;
    }

    .logit {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      text-align: right;
    }

    @media (max-width: 860px) {
      main {
        grid-template-columns: 1fr;
      }

      .readout {
        order: -1;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="stage">
      <div class="appbar">
        <div>
          <h1>NoseKnows fragrance wheel</h1>
          <div id="status" class="status">Generating random NoseLLM vectors every 3 seconds</div>
        </div>
        <button id="toggle" type="button">Stop</button>
      </div>

      <div class="wheel-wrap" aria-label="Fragrance wheel">
        <svg id="wheel" class="wheel-overlay" viewBox="0 0 100 100" role="img" aria-label="Active fragrance sectors"></svg>
      </div>
    </section>

    <aside class="readout">
      <h2>Sensor vector</h2>
      <div id="sensors" class="sensor-grid"></div>
      <h2>Top categories</h2>
      <div id="ranks"></div>
    </aside>
  </main>

  <script>
    const categories = [
      { name: "Floral", color: "#d85a8a" },
      { name: "Soft Floral", color: "#e994b0" },
      { name: "Floral Amber", color: "#c89184" },
      { name: "Amber", color: "#c56630" },
      { name: "Soft Amber", color: "#d99053" },
      { name: "Woody Amber", color: "#a66a3f" },
      { name: "Woods", color: "#81613c" },
      { name: "Mossy Woods", color: "#6f7a4b" },
      { name: "Dry Woods", color: "#9a875c" },
      { name: "Aromatic", color: "#6d91bd" },
      { name: "Citrus", color: "#e7bd35" },
      { name: "Water", color: "#63aac6" },
      { name: "Green", color: "#65a85e" },
      { name: "Fruity", color: "#dc6b66" }
    ];

    const wheel = document.querySelector("#wheel");
    const ranks = document.querySelector("#ranks");
    const sensors = document.querySelector("#sensors");
    const status = document.querySelector("#status");
    const toggle = document.querySelector("#toggle");
    const sectors = new Map();
    let running = true;
    let timer = null;
    const SVG_NS = "http://www.w3.org/2000/svg";

    function polar(cx, cy, r, degrees) {
      const radians = (degrees - 90) * Math.PI / 180;
      return [cx + r * Math.cos(radians), cy + r * Math.sin(radians)];
    }

    function sectorPath(start, end, inner = 17.5, outer = 44.8) {
      const [x1, y1] = polar(50, 50, outer, start);
      const [x2, y2] = polar(50, 50, outer, end);
      const [x3, y3] = polar(50, 50, inner, end);
      const [x4, y4] = polar(50, 50, inner, start);
      const largeArc = end - start > 180 ? 1 : 0;
      return [
        `M ${x1.toFixed(3)} ${y1.toFixed(3)}`,
        `A ${outer} ${outer} 0 ${largeArc} 1 ${x2.toFixed(3)} ${y2.toFixed(3)}`,
        `L ${x3.toFixed(3)} ${y3.toFixed(3)}`,
        `A ${inner} ${inner} 0 ${largeArc} 0 ${x4.toFixed(3)} ${y4.toFixed(3)}`,
        "Z"
      ].join(" ");
    }

    function svgEl(name, attributes = {}) {
      const node = document.createElementNS(SVG_NS, name);
      Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
      return node;
    }

    function labelText(name) {
      return name.includes(" ") ? name.replace(" ", "\n") : name;
    }

    function renderWheel() {
      wheel.innerHTML = "";
      sectors.clear();

      wheel.appendChild(svgEl("circle", { class: "wheel-rim", cx: 50, cy: 50, r: 47 }));
      wheel.appendChild(svgEl("circle", { class: "wheel-rim", cx: 50, cy: 50, r: 36.8 }));
      wheel.appendChild(svgEl("circle", { class: "wheel-rim", cx: 50, cy: 50, r: 24.6 }));

      const sweep = 360 / categories.length;
      const startOffset = -sweep * 1.5;
      const baseLayer = svgEl("g");
      const activeLayer = svgEl("g");
      const labelLayer = svgEl("g");
      wheel.append(baseLayer, activeLayer, labelLayer);

      categories.forEach((category, index) => {
        const start = startOffset + index * sweep + 0.62;
        const end = startOffset + (index + 1) * sweep - 0.62;
        const mid = (start + end) / 2;
        const path = sectorPath(start, end);

        baseLayer.appendChild(svgEl("path", {
          class: "base-sector",
          d: path,
          fill: category.color
        }));

        const active = svgEl("path", {
          class: "sector",
          d: path,
          fill: category.color
        });
        activeLayer.appendChild(active);
        sectors.set(category.name, active);

        const [tx1, ty1] = polar(50, 50, 45.8, start - 0.62);
        const [tx2, ty2] = polar(50, 50, 48, start - 0.62);
        labelLayer.appendChild(svgEl("line", {
          class: "wheel-tick",
          x1: tx1.toFixed(2),
          y1: ty1.toFixed(2),
          x2: tx2.toFixed(2),
          y2: ty2.toFixed(2)
        }));

        const [lx, ly] = polar(50, 50, 32.2, mid);
        const label = svgEl("text", {
          class: "wheel-label",
          x: lx.toFixed(2),
          y: ly.toFixed(2)
        });

        const lines = labelText(category.name).split("\n");
        lines.forEach((line, lineIndex) => {
          const tspan = svgEl("tspan", {
            x: lx.toFixed(2),
            dy: lineIndex === 0 ? (lines.length > 1 ? "-0.58em" : "0") : "1.16em"
          });
          tspan.textContent = line;
          label.appendChild(tspan);
        });
        labelLayer.appendChild(label);
      });

      wheel.appendChild(svgEl("circle", { class: "wheel-center", cx: 50, cy: 50, r: 16.2 }));
      const title = svgEl("text", { class: "center-title", x: 50, y: 46.9 });
      title.textContent = "NOSELLM";
      wheel.appendChild(title);
      const subtitleA = svgEl("text", { class: "center-subtitle", x: 50, y: 51.2 });
      subtitleA.textContent = "FRAGRANCE";
      wheel.appendChild(subtitleA);
      const subtitleB = svgEl("text", { class: "center-subtitle", x: 50, y: 55.4 });
      subtitleB.textContent = "WHEEL";
      wheel.appendChild(subtitleB);
    }

    function renderSensors() {
      sensors.innerHTML = "";
      Array.from({ length: 9 }, (_, index) => {
        const value = (Math.random() * 2 - 1).toFixed(3);
        const node = document.createElement("div");
        node.className = "sensor";
        node.innerHTML = `<span>Gas ${index + 1}</span><strong>${value}</strong>`;
        sensors.appendChild(node);
      });
    }

    function categoryColor(name) {
      return categories.find((category) => category.name === name)?.color ?? "#77808a";
    }

    function applyState(state) {
      for (const sector of sectors.values()) {
        sector.classList.remove("active");
        sector.style.removeProperty("--sector-opacity");
        sector.style.removeProperty("--sector-strength");
      }

      ranks.innerHTML = "";
      state.activations.forEach((activation, index) => {
        const sector = sectors.get(activation.name);
        if (sector) {
          const opacity = 0.32 + activation.strength * 0.66;
          sector.classList.add("active");
          sector.style.setProperty("--sector-opacity", opacity.toFixed(3));
          sector.style.setProperty("--sector-strength", activation.strength.toFixed(3));
        }

        const color = categoryColor(activation.name);
        const row = document.createElement("div");
        row.className = "rank";
        row.style.color = color;
        row.innerHTML = `
          <div class="swatch" style="background: ${color}"></div>
          <div>
            <strong>${index + 1}. ${activation.name}</strong>
            <div class="bar" aria-hidden="true"><div style="--value: ${activation.strength}"></div></div>
          </div>
          <div class="logit">${activation.logit.toFixed(2)}</div>
        `;
        ranks.appendChild(row);
      });

      renderSensors();
      status.textContent = running
        ? "Generating random NoseLLM vectors every 3 seconds"
        : "Paused";
    }

    async function refresh() {
      const response = await fetch("/state", { cache: "no-store" });
      applyState(await response.json());
    }

    function schedule() {
      clearInterval(timer);
      if (running) {
        refresh();
        timer = setInterval(refresh, 3000);
      }
    }

    toggle.addEventListener("click", () => {
      running = !running;
      toggle.textContent = running ? "Stop" : "Start";
      status.textContent = running ? "Generating random NoseLLM vectors every 3 seconds" : "Paused";
      schedule();
    });

    renderWheel();
    schedule();
  </script>
</body>
</html>
"##;
