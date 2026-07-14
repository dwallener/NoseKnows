use crate::peak::OUTPUTS;
use std::collections::VecDeque;

pub const EMBEDDING_VERSION: &str = "scent_embedding_v1";
pub const EMBEDDING_DIMS: usize = 1024;
pub const LABEL_PAIRS: usize = OUTPUTS * (OUTPUTS - 1) / 2;
const SHORT_WINDOW: usize = 8;
const LONG_WINDOW: usize = 32;
const FEATURE_PREFIX_DIMS: usize = 256;

pub struct EmbeddingRuntime {
    logits_history: VecDeque<[f32; OUTPUTS]>,
    top3_history: VecDeque<[bool; OUTPUTS]>,
    top1_history: VecDeque<[bool; OUTPUTS]>,
    rows_since_seen: [usize; OUTPUTS],
}

impl EmbeddingRuntime {
    pub fn new() -> Self {
        Self {
            logits_history: VecDeque::with_capacity(LONG_WINDOW),
            top3_history: VecDeque::with_capacity(LONG_WINDOW),
            top1_history: VecDeque::with_capacity(LONG_WINDOW),
            rows_since_seen: [LONG_WINDOW; OUTPUTS],
        }
    }

    pub fn step(
        &mut self,
        logits: &[f32; OUTPUTS],
        predicted: &[usize],
        model_features: &[f32],
    ) -> [f32; EMBEDDING_DIMS] {
        let mut top3 = [false; OUTPUTS];
        let mut top1 = [false; OUTPUTS];
        for label in predicted.iter().take(3) {
            top3[*label] = true;
        }
        if let Some(label) = predicted.first() {
            top1[*label] = true;
        }

        self.logits_history.push_back(*logits);
        self.top3_history.push_back(top3);
        self.top1_history.push_back(top1);
        while self.logits_history.len() > LONG_WINDOW {
            self.logits_history.pop_front();
        }
        while self.top3_history.len() > LONG_WINDOW {
            self.top3_history.pop_front();
        }
        while self.top1_history.len() > LONG_WINDOW {
            self.top1_history.pop_front();
        }

        for label in 0..OUTPUTS {
            self.rows_since_seen[label] = if top3[label] {
                0
            } else {
                self.rows_since_seen[label]
                    .saturating_add(1)
                    .min(LONG_WINDOW)
            };
        }

        self.embedding(logits, &top3, &top1, model_features)
    }

    fn embedding(
        &self,
        logits: &[f32; OUTPUTS],
        top3: &[bool; OUTPUTS],
        top1: &[bool; OUTPUTS],
        model_features: &[f32],
    ) -> [f32; EMBEDDING_DIMS] {
        let mut vector = [0.0_f32; EMBEDDING_DIMS];

        write_label_block(&mut vector, 0, *logits, squash_logit);
        write_label_block(&mut vector, 14, positive_logits(logits), |value| value);
        write_label_block(
            &mut vector,
            28,
            self.mean_logits(SHORT_WINDOW),
            squash_logit,
        );
        write_label_block(&mut vector, 42, self.max_logits(SHORT_WINDOW), squash_logit);
        write_label_block(&mut vector, 56, self.mean_logits(LONG_WINDOW), squash_logit);
        write_label_block(&mut vector, 70, self.max_logits(LONG_WINDOW), squash_logit);
        write_label_block(&mut vector, 84, self.top3_frequency(), |value| value);
        write_label_block(&mut vector, 98, self.top1_frequency(), |value| value);
        write_label_block(&mut vector, 112, self.recency(), |value| value);
        write_label_block(&mut vector, 126, bools_to_f32(top3), |value| value);
        write_label_block(&mut vector, 140, bools_to_f32(top1), |value| value);

        let pairwise = self.pairwise_coactivation();
        for (offset, value) in pairwise.iter().enumerate() {
            vector[154 + offset] = *value;
        }

        for (offset, value) in model_features.iter().take(FEATURE_PREFIX_DIMS).enumerate() {
            vector[256 + offset] = *value;
        }

        vector
    }

    fn mean_logits(&self, window: usize) -> [f32; OUTPUTS] {
        let mut means = [0.0_f32; OUTPUTS];
        let rows = self
            .logits_history
            .iter()
            .rev()
            .take(window)
            .collect::<Vec<_>>();
        if rows.is_empty() {
            return means;
        }
        for row in &rows {
            for label in 0..OUTPUTS {
                means[label] += row[label];
            }
        }
        for value in &mut means {
            *value /= rows.len() as f32;
        }
        means
    }

    fn max_logits(&self, window: usize) -> [f32; OUTPUTS] {
        let mut maxima = [f32::NEG_INFINITY; OUTPUTS];
        let mut seen = false;
        for row in self.logits_history.iter().rev().take(window) {
            seen = true;
            for label in 0..OUTPUTS {
                maxima[label] = maxima[label].max(row[label]);
            }
        }
        if seen {
            maxima
        } else {
            [0.0; OUTPUTS]
        }
    }

    fn top3_frequency(&self) -> [f32; OUTPUTS] {
        label_frequency(&self.top3_history)
    }

    fn top1_frequency(&self) -> [f32; OUTPUTS] {
        label_frequency(&self.top1_history)
    }

    fn recency(&self) -> [f32; OUTPUTS] {
        let mut recency = [0.0_f32; OUTPUTS];
        for (label, value) in recency.iter_mut().enumerate() {
            *value =
                1.0 - (self.rows_since_seen[label].min(LONG_WINDOW) as f32 / LONG_WINDOW as f32);
        }
        recency
    }

    fn pairwise_coactivation(&self) -> [f32; LABEL_PAIRS] {
        let rows = self
            .top3_history
            .iter()
            .rev()
            .take(LONG_WINDOW)
            .collect::<Vec<_>>();
        let mut values = [0.0_f32; LABEL_PAIRS];
        if rows.is_empty() {
            return values;
        }
        let mut index = 0;
        for left in 0..OUTPUTS {
            for right in left + 1..OUTPUTS {
                let count = rows.iter().filter(|row| row[left] && row[right]).count();
                values[index] = count as f32 / rows.len() as f32;
                index += 1;
            }
        }
        values
    }
}

fn label_frequency(history: &VecDeque<[bool; OUTPUTS]>) -> [f32; OUTPUTS] {
    let mut frequency = [0.0_f32; OUTPUTS];
    let rows = history.iter().rev().take(LONG_WINDOW).collect::<Vec<_>>();
    if rows.is_empty() {
        return frequency;
    }
    for row in &rows {
        for label in 0..OUTPUTS {
            if row[label] {
                frequency[label] += 1.0;
            }
        }
    }
    for value in &mut frequency {
        *value /= rows.len() as f32;
    }
    frequency
}

fn positive_logits(logits: &[f32; OUTPUTS]) -> [f32; OUTPUTS] {
    let mut positive = [0.0_f32; OUTPUTS];
    for label in 0..OUTPUTS {
        positive[label] = logits[label].max(0.0).min(16.0) / 16.0;
    }
    positive
}

fn bools_to_f32(flags: &[bool; OUTPUTS]) -> [f32; OUTPUTS] {
    let mut values = [0.0_f32; OUTPUTS];
    for label in 0..OUTPUTS {
        values[label] = if flags[label] { 1.0 } else { 0.0 };
    }
    values
}

fn write_label_block(
    vector: &mut [f32; EMBEDDING_DIMS],
    start: usize,
    values: [f32; OUTPUTS],
    transform: fn(f32) -> f32,
) {
    for (offset, value) in values.into_iter().enumerate() {
        vector[start + offset] = transform(value);
    }
}

fn squash_logit(value: f32) -> f32 {
    (value / 8.0).tanh()
}

pub fn format_embedding(vector: &[f32; EMBEDDING_DIMS]) -> String {
    vector
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join("|")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_has_fixed_width() {
        let mut runtime = EmbeddingRuntime::new();
        let logits = [0.0; OUTPUTS];
        let vector = runtime.step(&logits, &[], &[]);
        assert_eq!(vector.len(), EMBEDDING_DIMS);
    }

    #[test]
    fn persistent_prediction_raises_frequency_and_recency() {
        let mut runtime = EmbeddingRuntime::new();
        let mut logits = [0.0; OUTPUTS];
        logits[3] = 10.0;
        let vector = runtime.step(&logits, &[3], &[]);
        assert!(vector[84 + 3] > 0.0);
        assert_eq!(vector[98 + 3], 1.0);
        assert_eq!(vector[112 + 3], 1.0);
    }
}
