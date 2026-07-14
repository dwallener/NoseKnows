use crate::peak::{label_index, CHANNELS, LABELS, MAX_ADC};

pub const ACTIVE_SENSORS: usize = 8;

#[derive(Clone)]
pub struct GainStage {
    focus_label: Option<String>,
    gains: [f32; ACTIVE_SENSORS],
    mask_id: String,
}

#[derive(Clone)]
pub struct GainOutput {
    pub adc: [f32; CHANNELS],
    pub gains: [f32; ACTIVE_SENSORS],
    pub mask_id: String,
    pub focus_label: Option<String>,
    pub clip_count: usize,
}

impl GainStage {
    pub fn identity() -> Self {
        Self {
            focus_label: None,
            gains: [1.0; ACTIVE_SENSORS],
            mask_id: "identity".to_string(),
        }
    }

    pub fn for_focus(label: Option<&str>) -> Result<Self, Box<dyn std::error::Error>> {
        let Some(label) = label.filter(|value| !value.trim().is_empty()) else {
            return Ok(Self::identity());
        };
        let normalized = normalize_label(label);
        let index = label_index(&normalized)
            .ok_or_else(|| format!("unknown focus label: {label}. Expected one of {:?}", LABELS))?;
        Ok(Self {
            focus_label: Some(LABELS[index].to_string()),
            gains: focus_mask(index),
            mask_id: format!("focus_{}", LABELS[index].to_lowercase().replace(' ', "_")),
        })
    }

    pub fn has_focus(&self) -> bool {
        self.focus_label.is_some()
    }

    pub fn apply(&self, adc: &[f32; CHANNELS]) -> GainOutput {
        let mut masked = *adc;
        let mut clip_count = 0_usize;
        for sensor in 0..ACTIVE_SENSORS {
            let value = adc[sensor] * self.gains[sensor];
            if value >= MAX_ADC {
                clip_count += 1;
            }
            masked[sensor] = value.clamp(0.0, MAX_ADC);
        }
        GainOutput {
            adc: masked,
            gains: self.gains,
            mask_id: self.mask_id.clone(),
            focus_label: self.focus_label.clone(),
            clip_count,
        }
    }
}

fn normalize_label(label: &str) -> String {
    label
        .trim()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn focus_mask(label_index: usize) -> [f32; ACTIVE_SENSORS] {
    let mut gains = [0.65_f32; ACTIVE_SENSORS];
    for sensor in relevant_sensors(label_index) {
        gains[*sensor] = 1.0;
    }
    gains
}

fn relevant_sensors(label_index: usize) -> &'static [usize] {
    match LABELS[label_index] {
        "Floral" => &[1, 7],
        "Soft Floral" => &[1, 4, 7],
        "Floral Amber" => &[0, 1, 4, 6, 7],
        "Amber" => &[0, 1, 4, 6, 7],
        "Soft Amber" => &[0, 1, 4, 7],
        "Woody Amber" => &[0, 1, 4, 6, 7],
        "Woods" => &[0, 1, 7],
        "Mossy Woods" => &[0, 1, 4, 7],
        "Dry Woods" => &[0, 1, 4, 6, 7],
        "Aromatic" => &[0, 1, 3, 4, 7],
        "Citrus" => &[0, 1, 2, 3, 7],
        "Water" => &[1, 7],
        "Green" => &[1, 3, 7],
        "Fruity" => &[0, 1, 7],
        _ => &[],
    }
}

pub fn format_adc(adc: &[f32; CHANNELS]) -> String {
    adc.iter()
        .take(CHANNELS)
        .map(|value| format!("{value:.1}"))
        .collect::<Vec<_>>()
        .join("|")
}

pub fn format_gains(gains: &[f32; ACTIVE_SENSORS]) -> String {
    gains
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join("|")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_leaves_adc_unchanged() {
        let adc = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let output = GainStage::identity().apply(&adc);
        assert_eq!(output.adc, adc);
        assert_eq!(output.clip_count, 0);
    }

    #[test]
    fn focus_attenuates_non_relevant_sensors() {
        let adc = [1000.0; CHANNELS];
        let stage = GainStage::for_focus(Some("Citrus")).unwrap();
        let output = stage.apply(&adc);
        assert_eq!(output.adc[0], 1000.0);
        assert_eq!(output.adc[2], 1000.0);
        assert!(output.adc[4] < 1000.0);
    }

    #[test]
    fn unknown_focus_is_rejected() {
        assert!(GainStage::for_focus(Some("Banana Peel")).is_err());
    }
}
