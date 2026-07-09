use std::cmp::Ordering;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const CHANNELS: usize = 9;
const SEQ_LEN: usize = 32;
const D_MODEL: usize = 16;
const D_FF: usize = 32;
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
const NO_SCENT_LABEL: &str = "No Scent";

#[derive(Clone)]
struct Sample {
    id: String,
    name: String,
    labels: [String; 3],
    target: [f32; LABELS.len()],
    sequence: Vec<[f32; CHANNELS]>,
}

struct TinyTransformer {
    input_proj: Matrix,
    input_bias: Vec<f32>,
    pos_embedding: Matrix,
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix,
    ff_1: Matrix,
    ff_1_bias: Vec<f32>,
    ff_2: Matrix,
    ff_2_bias: Vec<f32>,
    classifier: Matrix,
    classifier_bias: Vec<f32>,
}

#[derive(Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

struct TrainConfig {
    data_dir: PathBuf,
    output_path: PathBuf,
    model_path: PathBuf,
    predict_path: Option<PathBuf>,
    epochs: usize,
    learning_rate: f32,
    full_model: bool,
    full_learning_rate: f32,
    perturbation: f32,
    seed: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("train error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;

    if let Some(predict_path) = &config.predict_path {
        return run_prediction(&config.model_path, predict_path, config.seed);
    }

    let samples = load_samples(&config.data_dir)?;

    if samples.is_empty() {
        return Err(format!(
            "no usable CSV samples found in {}",
            config.data_dir.display()
        )
        .into());
    }

    println!(
        "Loaded {} sample(s) from {}",
        samples.len(),
        config.data_dir.display()
    );

    let mut rng = Lcg::new(config.seed);
    let mut model = TinyTransformer::new(&mut rng);
    let mut params = model.to_params();

    if config.full_model {
        println!(
            "Training mode: full model SPSA over {} parameters",
            params.len()
        );
    } else {
        println!(
            "Training mode: output head only over {} parameters",
            head_param_count()
        );
    }

    for epoch in 1..=config.epochs {
        if config.full_model {
            train_output_head_epoch(&mut model, &samples, config.learning_rate);
            params = model.to_params();
            train_full_model_spsa_epoch(
                &mut model,
                &mut params,
                &samples,
                &mut rng,
                config.full_learning_rate,
                config.perturbation,
            );
        } else {
            train_output_head_epoch(&mut model, &samples, config.learning_rate);
        }

        let loss = dataset_loss(&model, &samples);
        if epoch == 1 || epoch == config.epochs || epoch % 10 == 0 {
            println!("epoch {epoch:>4} loss {loss:.5}");
        }
    }

    println!();
    println!("Training-set predictions:");
    for sample in &samples {
        let logits = model.forward(&sample.sequence);
        let top = top_k(&logits, 3);
        println!(
            "{} ({}) labels=[{}, {}, {}] predicted=[{}, {}, {}]",
            sample.id,
            sample.name,
            sample.labels[0],
            sample.labels[1],
            sample.labels[2],
            LABELS[top[0].0],
            LABELS[top[1].0],
            LABELS[top[2].0]
        );
    }

    let params = model.to_params();
    save_model(&config.output_path, &params)?;
    println!();
    println!("Saved model parameters to {}", config.output_path.display());

    Ok(())
}

fn parse_args() -> Result<TrainConfig, Box<dyn std::error::Error>> {
    let mut data_dir = PathBuf::from("data/raw");
    let mut output_path = PathBuf::from("data/models/tiny_transformer.ntm");
    let mut model_path = PathBuf::from("data/models/tiny_transformer.ntm");
    let mut predict_path = None;
    let mut epochs = 100;
    let mut learning_rate = 0.08;
    let mut full_model = false;
    let mut full_learning_rate = 0.004;
    let mut perturbation = 0.02;
    let mut seed = 0x5eed_1234_u64;

    let args: Vec<String> = env::args().skip(1).collect();
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
            "--model" => {
                index += 1;
                model_path = PathBuf::from(args.get(index).ok_or("--model requires a path")?);
            }
            "--predict" => {
                index += 1;
                predict_path = Some(PathBuf::from(
                    args.get(index).ok_or("--predict requires a CSV path")?,
                ));
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
            "--full-model" => {
                full_model = true;
            }
            "--full-lr" => {
                index += 1;
                full_learning_rate = args
                    .get(index)
                    .ok_or("--full-lr requires a value")?
                    .parse()?;
            }
            "--perturbation" => {
                index += 1;
                perturbation = args
                    .get(index)
                    .ok_or("--perturbation requires a value")?
                    .parse()?;
            }
            "--seed" => {
                index += 1;
                seed = args.get(index).ok_or("--seed requires a value")?.parse()?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        index += 1;
    }

    Ok(TrainConfig {
        data_dir,
        output_path,
        model_path,
        predict_path,
        epochs,
        learning_rate,
        full_model,
        full_learning_rate,
        perturbation,
        seed,
    })
}

fn print_usage() {
    println!(
        "Usage: cargo run --bin train -- [--data data/raw] [--out data/models/tiny_transformer.ntm] [--epochs 100] [--lr 0.08]"
    );
    println!(
        "       cargo run --bin train -- --data data/raw --out data/models/full_transformer.ntm --full-model --epochs 200"
    );
    println!(
        "       cargo run --bin train -- --model data/models/tiny_transformer.ntm --predict data/raw/sample.csv"
    );
}

fn run_prediction(
    model_path: &Path,
    predict_path: &Path,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let sample = load_sample(predict_path)?
        .ok_or_else(|| format!("no usable sample rows in {}", predict_path.display()))?;
    let params = load_model(model_path)?;

    let mut rng = Lcg::new(seed);
    let mut model = TinyTransformer::new(&mut rng);
    model.load_params_checked(&params)?;

    let logits = model.forward(&sample.sequence);
    let top = top_k(&logits, 3);

    println!("Inference for {} ({})", sample.id, sample.name);
    println!(
        "stored labels=[{}, {}, {}]",
        sample.labels[0], sample.labels[1], sample.labels[2]
    );
    for (rank, (index, logit)) in top.iter().enumerate() {
        println!("{}. {} logit={:.5}", rank + 1, LABELS[*index], logit);
    }

    Ok(())
}

fn load_samples(data_dir: &Path) -> Result<Vec<Sample>, Box<dyn std::error::Error>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(data_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) == Some("csv") {
            paths.push(path);
        }
    }
    paths.sort();

    let mut samples = Vec::new();
    for path in paths {
        if let Some(sample) = load_sample(&path)? {
            samples.push(sample);
        }
    }

    Ok(samples)
}

fn load_sample(path: &Path) -> Result<Option<Sample>, Box<dyn std::error::Error>> {
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
    let label_1_index = index("label_1")?;
    let label_2_index = index("label_2")?;
    let label_3_index = index("label_3")?;
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

    let mut sample_id = String::new();
    let mut sample_name = String::new();
    let mut labels = [String::new(), String::new(), String::new()];
    let mut raw_rows = Vec::new();

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }

        let fields = parse_csv_line(line);
        if fields.len() <= *adc_indexes.iter().max().expect("adc indexes") {
            continue;
        }

        if sample_id.is_empty() {
            sample_id = fields[sample_id_index].clone();
            sample_name = fields[sample_name_index].clone();
            labels = [
                fields[label_1_index].clone(),
                fields[label_2_index].clone(),
                fields[label_3_index].clone(),
            ];
        }

        let mut row = [0.0_f32; CHANNELS];
        for (channel, field_index) in adc_indexes.iter().enumerate() {
            row[channel] = fields[*field_index].parse::<f32>()? / 4095.0;
        }
        raw_rows.push(row);
    }

    if raw_rows.is_empty() {
        return Ok(None);
    }

    let target = target_from_labels(&labels)?;
    let sequence = resample_sequence(&raw_rows, SEQ_LEN);
    Ok(Some(Sample {
        id: sample_id,
        name: sample_name,
        labels,
        target,
        sequence,
    }))
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;

    while let Some(character) = chars.next() {
        match character {
            '"' if in_quotes && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                fields.push(field);
                field = String::new();
            }
            _ => field.push(character),
        }
    }

    fields.push(field);
    fields
}

fn target_from_labels(
    labels: &[String; 3],
) -> Result<[f32; LABELS.len()], Box<dyn std::error::Error>> {
    let mut target = [0.0_f32; LABELS.len()];
    let weights = [1.0_f32, 0.66, 0.33];

    for (label, weight) in labels.iter().zip(weights) {
        if is_no_scent_label(label) {
            continue;
        }
        let index = label_index(label).ok_or_else(|| format!("unknown label: {label}"))?;
        target[index] = weight;
    }

    Ok(target)
}

fn is_no_scent_label(label: &str) -> bool {
    normalize_label(label) == normalize_label(NO_SCENT_LABEL)
}

fn label_index(label: &str) -> Option<usize> {
    let normalized = normalize_label(label);
    LABELS
        .iter()
        .position(|candidate| normalize_label(candidate) == normalized)
}

fn normalize_label(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace() && *character != '-' && *character != '_')
        .flat_map(char::to_lowercase)
        .collect()
}

fn resample_sequence(rows: &[[f32; CHANNELS]], seq_len: usize) -> Vec<[f32; CHANNELS]> {
    if rows.len() == seq_len {
        return rows.to_vec();
    }

    let mut output = Vec::with_capacity(seq_len);
    for index in 0..seq_len {
        let source = if seq_len == 1 {
            0
        } else {
            index * (rows.len() - 1) / (seq_len - 1)
        };
        output.push(rows[source]);
    }
    output
}

impl TinyTransformer {
    fn new(rng: &mut Lcg) -> Self {
        Self {
            input_proj: Matrix::random(CHANNELS, D_MODEL, 0.25, rng),
            input_bias: vec![0.0; D_MODEL],
            pos_embedding: Matrix::random(SEQ_LEN, D_MODEL, 0.05, rng),
            w_q: Matrix::random(D_MODEL, D_MODEL, 0.15, rng),
            w_k: Matrix::random(D_MODEL, D_MODEL, 0.15, rng),
            w_v: Matrix::random(D_MODEL, D_MODEL, 0.15, rng),
            w_o: Matrix::random(D_MODEL, D_MODEL, 0.15, rng),
            ff_1: Matrix::random(D_MODEL, D_FF, 0.15, rng),
            ff_1_bias: vec![0.0; D_FF],
            ff_2: Matrix::random(D_FF, D_MODEL, 0.15, rng),
            ff_2_bias: vec![0.0; D_MODEL],
            classifier: Matrix::random(D_MODEL, LABELS.len(), 0.15, rng),
            classifier_bias: vec![0.0; LABELS.len()],
        }
    }

    fn forward(&self, sequence: &[[f32; CHANNELS]]) -> Vec<f32> {
        let pooled = self.encode(sequence);
        self.classify(&pooled)
    }

    fn encode(&self, sequence: &[[f32; CHANNELS]]) -> [f32; D_MODEL] {
        let mut hidden = vec![[0.0_f32; D_MODEL]; SEQ_LEN];
        for (position, row) in sequence.iter().enumerate().take(SEQ_LEN) {
            for dim in 0..D_MODEL {
                let mut value = self.input_bias[dim] + self.pos_embedding.get(position, dim);
                for (channel, input) in row.iter().enumerate() {
                    value += input * self.input_proj.get(channel, dim);
                }
                hidden[position][dim] = gelu(value);
            }
        }

        let attention = self.self_attention(&hidden);
        let mut transformed = vec![[0.0_f32; D_MODEL]; SEQ_LEN];
        for position in 0..SEQ_LEN {
            for dim in 0..D_MODEL {
                transformed[position][dim] = hidden[position][dim] + attention[position][dim];
            }
            layer_norm(&mut transformed[position]);
        }

        let mut pooled = [0.0_f32; D_MODEL];
        for token in &transformed {
            let mut token = *token;
            let ff = self.feed_forward(&token);
            for dim in 0..D_MODEL {
                token[dim] += ff[dim];
            }
            layer_norm(&mut token);
            for dim in 0..D_MODEL {
                pooled[dim] += token[dim] / SEQ_LEN as f32;
            }
        }

        pooled
    }

    fn classify(&self, pooled: &[f32; D_MODEL]) -> Vec<f32> {
        let mut logits = vec![0.0; LABELS.len()];
        for (class, logit) in logits.iter_mut().enumerate() {
            *logit = self.classifier_bias[class];
            for (dim, feature) in pooled.iter().enumerate() {
                *logit += feature * self.classifier.get(dim, class);
            }
        }

        logits
    }

    fn self_attention(&self, hidden: &[[f32; D_MODEL]]) -> Vec<[f32; D_MODEL]> {
        let mut q = vec![[0.0_f32; D_MODEL]; SEQ_LEN];
        let mut k = vec![[0.0_f32; D_MODEL]; SEQ_LEN];
        let mut v = vec![[0.0_f32; D_MODEL]; SEQ_LEN];

        for position in 0..SEQ_LEN {
            q[position] = mat_vec_d_model(&self.w_q, &hidden[position]);
            k[position] = mat_vec_d_model(&self.w_k, &hidden[position]);
            v[position] = mat_vec_d_model(&self.w_v, &hidden[position]);
        }

        let mut output = vec![[0.0_f32; D_MODEL]; SEQ_LEN];
        let scale = (D_MODEL as f32).sqrt();

        for position in 0..SEQ_LEN {
            let mut scores = [0.0_f32; SEQ_LEN];
            for other in 0..SEQ_LEN {
                scores[other] = dot(&q[position], &k[other]) / scale;
            }
            softmax_in_place(&mut scores);

            let mut context = [0.0_f32; D_MODEL];
            for other in 0..SEQ_LEN {
                for (dim, value) in context.iter_mut().enumerate() {
                    *value += scores[other] * v[other][dim];
                }
            }
            output[position] = mat_vec_d_model(&self.w_o, &context);
        }

        output
    }

    fn feed_forward(&self, token: &[f32; D_MODEL]) -> [f32; D_MODEL] {
        let mut hidden = [0.0_f32; D_FF];
        for (dim, value) in hidden.iter_mut().enumerate() {
            *value = self.ff_1_bias[dim];
            for (input_dim, token_value) in token.iter().enumerate() {
                *value += token_value * self.ff_1.get(input_dim, dim);
            }
            *value = gelu(*value);
        }

        let mut output = [0.0_f32; D_MODEL];
        for (dim, value) in output.iter_mut().enumerate() {
            *value = self.ff_2_bias[dim];
            for (hidden_dim, hidden_value) in hidden.iter().enumerate() {
                *value += hidden_value * self.ff_2.get(hidden_dim, dim);
            }
        }
        output
    }

    fn to_params(&self) -> Vec<f32> {
        let mut params = Vec::new();
        self.push_params(&mut params);
        params
    }

    fn push_params(&self, params: &mut Vec<f32>) {
        self.input_proj.push_params(params);
        params.extend(&self.input_bias);
        self.pos_embedding.push_params(params);
        self.w_q.push_params(params);
        self.w_k.push_params(params);
        self.w_v.push_params(params);
        self.w_o.push_params(params);
        self.ff_1.push_params(params);
        params.extend(&self.ff_1_bias);
        self.ff_2.push_params(params);
        params.extend(&self.ff_2_bias);
        self.classifier.push_params(params);
        params.extend(&self.classifier_bias);
    }

    fn load_params(&mut self, params: &[f32]) {
        let mut offset = 0;
        self.input_proj.load_params(params, &mut offset);
        load_vec(&mut self.input_bias, params, &mut offset);
        self.pos_embedding.load_params(params, &mut offset);
        self.w_q.load_params(params, &mut offset);
        self.w_k.load_params(params, &mut offset);
        self.w_v.load_params(params, &mut offset);
        self.w_o.load_params(params, &mut offset);
        self.ff_1.load_params(params, &mut offset);
        load_vec(&mut self.ff_1_bias, params, &mut offset);
        self.ff_2.load_params(params, &mut offset);
        load_vec(&mut self.ff_2_bias, params, &mut offset);
        self.classifier.load_params(params, &mut offset);
        load_vec(&mut self.classifier_bias, params, &mut offset);
        debug_assert_eq!(offset, params.len());
    }

    fn load_params_checked(&mut self, params: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        let expected = self.to_params().len();
        if params.len() != expected {
            return Err(format!(
                "model parameter count mismatch: expected {expected}, got {}",
                params.len()
            )
            .into());
        }
        self.load_params(params);
        Ok(())
    }
}

impl Matrix {
    fn random(rows: usize, cols: usize, scale: f32, rng: &mut Lcg) -> Self {
        let values = (0..rows * cols)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        Self { rows, cols, values }
    }

    fn get(&self, row: usize, col: usize) -> f32 {
        self.values[row * self.cols + col]
    }

    fn push_params(&self, params: &mut Vec<f32>) {
        params.extend(&self.values);
    }

    fn load_params(&mut self, params: &[f32], offset: &mut usize) {
        let end = *offset + self.values.len();
        self.values.copy_from_slice(&params[*offset..end]);
        *offset = end;
    }
}

fn load_vec(values: &mut [f32], params: &[f32], offset: &mut usize) {
    let end = *offset + values.len();
    values.copy_from_slice(&params[*offset..end]);
    *offset = end;
}

fn mat_vec_d_model(matrix: &Matrix, input: &[f32; D_MODEL]) -> [f32; D_MODEL] {
    debug_assert_eq!(matrix.rows, D_MODEL);
    debug_assert_eq!(matrix.cols, D_MODEL);

    let mut output = [0.0_f32; D_MODEL];
    for (col, value) in output.iter_mut().enumerate() {
        for (row, input_value) in input.iter().enumerate() {
            *value += input_value * matrix.get(row, col);
        }
    }
    output
}

fn dataset_loss(model: &TinyTransformer, samples: &[Sample]) -> f32 {
    let mut loss = 0.0;
    for sample in samples {
        let logits = model.forward(&sample.sequence);
        loss += binary_cross_entropy_with_logits(&logits, &sample.target);
    }
    loss / samples.len() as f32
}

fn train_output_head_epoch(model: &mut TinyTransformer, samples: &[Sample], learning_rate: f32) {
    for sample in samples {
        let features = model.encode(&sample.sequence);
        let logits = model.classify(&features);

        for (class, logit) in logits.iter().enumerate() {
            let gradient = (sigmoid(*logit) - sample.target[class]) / LABELS.len() as f32;
            model.classifier_bias[class] -= learning_rate * gradient;
            for (dim, feature) in features.iter().enumerate() {
                let index = dim * model.classifier.cols + class;
                model.classifier.values[index] -= learning_rate * gradient * feature;
                model.classifier.values[index] = model.classifier.values[index].clamp(-4.0, 4.0);
            }
        }
    }
}

fn train_full_model_spsa_epoch(
    model: &mut TinyTransformer,
    params: &mut [f32],
    samples: &[Sample],
    rng: &mut Lcg,
    learning_rate: f32,
    perturbation: f32,
) {
    let delta = (0..params.len())
        .map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 })
        .collect::<Vec<_>>();

    let mut plus = params.to_vec();
    let mut minus = params.to_vec();
    for ((plus_value, minus_value), delta_value) in
        plus.iter_mut().zip(minus.iter_mut()).zip(delta.iter())
    {
        *plus_value += perturbation * delta_value;
        *minus_value -= perturbation * delta_value;
    }

    model.load_params(&plus);
    let plus_loss = dataset_loss(model, samples);
    model.load_params(&minus);
    let minus_loss = dataset_loss(model, samples);
    let gradient_scale = (plus_loss - minus_loss) / (2.0 * perturbation);

    for (param, delta_value) in params.iter_mut().zip(delta.iter()) {
        *param -= learning_rate * gradient_scale * delta_value;
        *param = param.clamp(-4.0, 4.0);
    }

    model.load_params(params);
}

fn head_param_count() -> usize {
    D_MODEL * LABELS.len() + LABELS.len()
}

fn save_model(path: &Path, params: &[f32]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut output = String::new();
    output.push_str("format=noseknows-tiny-transformer-v1\n");
    output.push_str(&format!("channels={CHANNELS}\n"));
    output.push_str(&format!("seq_len={SEQ_LEN}\n"));
    output.push_str(&format!("d_model={D_MODEL}\n"));
    output.push_str(&format!("d_ff={D_FF}\n"));
    output.push_str(&format!("labels={}\n", LABELS.join("|")));
    output.push_str("params=");
    for (index, value) in params.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&format!("{value:.8}"));
    }
    output.push('\n');

    fs::write(path, output)
}

fn load_model(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    for line in text.lines() {
        if let Some(params) = line.strip_prefix("params=") {
            return params
                .split(',')
                .filter(|value| !value.trim().is_empty())
                .map(|value| value.parse::<f32>().map_err(Into::into))
                .collect();
        }
    }
    Err(format!("{} does not contain params=", path.display()).into())
}

fn binary_cross_entropy_with_logits(logits: &[f32], target: &[f32; LABELS.len()]) -> f32 {
    let mut loss = 0.0;
    for (logit, target) in logits.iter().zip(target.iter()) {
        let max = logit.max(0.0);
        loss += max - logit * target + (1.0 + (-logit.abs()).exp()).ln();
    }
    loss / logits.len() as f32
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh())
}

fn layer_norm(values: &mut [f32]) {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f32>()
        / values.len() as f32;
    let scale = (variance + 1e-5).sqrt();

    for value in values {
        *value = ((*value - mean) / scale).clamp(-6.0, 6.0);
    }
}

fn dot(left: &[f32; D_MODEL], right: &[f32; D_MODEL]) -> f32 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum.max(f32::MIN_POSITIVE);
    }
}

fn top_k(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    indexed.truncate(k);
    indexed
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_csv_quotes() {
        assert_eq!(
            parse_csv_line("a,\"b,c\",\"d\"\"e\""),
            vec!["a", "b,c", "d\"e"]
        );
    }

    #[test]
    fn maps_labels_to_weighted_targets() {
        let target = target_from_labels(&[
            "Aromatic".to_string(),
            "Soft Amber".to_string(),
            "Dry Woods".to_string(),
        ])
        .expect("target");

        assert_eq!(target[label_index("Aromatic").unwrap()], 1.0);
        assert_eq!(target[label_index("Soft Amber").unwrap()], 0.66);
        assert_eq!(target[label_index("Dry Woods").unwrap()], 0.33);
    }

    #[test]
    fn no_scent_labels_map_to_empty_target() {
        let target = target_from_labels(&[
            NO_SCENT_LABEL.to_string(),
            NO_SCENT_LABEL.to_string(),
            NO_SCENT_LABEL.to_string(),
        ])
        .expect("target");

        assert!(target.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn model_forward_returns_fourteen_logits() {
        let mut rng = Lcg::new(1);
        let model = TinyTransformer::new(&mut rng);
        let sequence = vec![[0.5; CHANNELS]; SEQ_LEN];
        assert_eq!(model.forward(&sequence).len(), LABELS.len());
    }
}
