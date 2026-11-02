//! (`whisper.model` x `whisper.quantization`) -> Hugging Face repo + ggml file name.
//!
//! Checked against the Hugging Face file listings of `ggerganov/whisper.cpp` and
//! `distil-whisper/distil-large-v3-ggml` (2026-11-01). Not every model has every variant:
//! `large-v3` has no q8_0, `small` has q5_1 instead of q5_0, and `distil-large-v3` only ships f16.

use stt_core::config::WhisperQuant;

/// Repo holding most ggml models.
pub const MAIN_REPO: &str = "ggerganov/whisper.cpp";
/// Repo holding the distilled model.
pub const DISTIL_REPO: &str = "distil-whisper/distil-large-v3-ggml";
/// Repo holding the Silero VAD model used for silence filtering.
pub const VAD_REPO: &str = "ggml-org/whisper-vad";
/// File name of the Silero VAD model (885 KB).
pub const VAD_FILE: &str = "ggml-silero-v5.1.2.bin";

/// A resolved model file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFile {
    /// Hugging Face repo id.
    pub repo: &'static str,
    /// File name inside the repo (also the local file name).
    pub file: &'static str,
    /// Variant actually chosen: "f16", "q8_0", "q5_0" or "q5_1".
    pub variant: &'static str,
    /// `Some(wanted)` when the requested variant does not exist and `variant` was used instead.
    pub substituted_for: Option<WhisperQuant>,
}

impl ModelFile {
    /// Text for `LoadInfo.detail`, e.g. `q5_0` or `q5_1 (q5_0 not available)`.
    pub fn detail(&self) -> String {
        match self.substituted_for {
            Some(want) => format!("{} ({} not available)", self.variant, want.as_str()),
            None => self.variant.to_string(),
        }
    }
}

type Variants = &'static [(&'static str, &'static str)];

/// Repo and available (variant, file name) pairs for a model name.
fn variants(model: &str) -> Option<(&'static str, Variants)> {
    Some(match model {
        "large-v3" => (
            MAIN_REPO,
            &[
                ("f16", "ggml-large-v3.bin"),
                ("q5_0", "ggml-large-v3-q5_0.bin"),
            ],
        ),
        "large-v3-turbo" => (
            MAIN_REPO,
            &[
                ("f16", "ggml-large-v3-turbo.bin"),
                ("q8_0", "ggml-large-v3-turbo-q8_0.bin"),
                ("q5_0", "ggml-large-v3-turbo-q5_0.bin"),
            ],
        ),
        "distil-large-v3" => (DISTIL_REPO, &[("f16", "ggml-distil-large-v3.bin")]),
        "medium" => (
            MAIN_REPO,
            &[
                ("f16", "ggml-medium.bin"),
                ("q8_0", "ggml-medium-q8_0.bin"),
                ("q5_0", "ggml-medium-q5_0.bin"),
            ],
        ),
        "small" => (
            MAIN_REPO,
            &[
                ("f16", "ggml-small.bin"),
                ("q8_0", "ggml-small-q8_0.bin"),
                ("q5_1", "ggml-small-q5_1.bin"),
            ],
        ),
        _ => return None,
    })
}

/// Resolve `model` and `quant` to a file. `None` for an unknown model name. When the exact variant is
/// missing, the nearest one is used: q8_0 falls back to q5, then f16; q5_0 to q5_1, q8_0, then f16;
/// f16 to q8_0, then q5.
pub fn resolve(model: &str, quant: WhisperQuant) -> Option<ModelFile> {
    let (repo, list) = variants(model)?;
    let order: &[&str] = match quant {
        WhisperQuant::F16 => &["f16", "q8_0", "q5_0", "q5_1"],
        WhisperQuant::Q8_0 => &["q8_0", "q5_0", "q5_1", "f16"],
        WhisperQuant::Q5_0 => &["q5_0", "q5_1", "q8_0", "f16"],
    };
    for want in order {
        if let Some((variant, file)) = list.iter().find(|(v, _)| v == want) {
            let exact = *variant == quant.as_str();
            return Some(ModelFile {
                repo,
                file,
                variant,
                substituted_for: (!exact).then_some(quant),
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use stt_core::transcribe::WHISPER_MODELS;

    const QUANTS: [WhisperQuant; 3] = [WhisperQuant::F16, WhisperQuant::Q8_0, WhisperQuant::Q5_0];

    #[test]
    fn every_model_resolves_for_every_quant() {
        for m in WHISPER_MODELS {
            for q in QUANTS {
                let f = resolve(m, q).unwrap_or_else(|| panic!("{m} {q:?}"));
                assert!(f.file.starts_with("ggml-") && f.file.ends_with(".bin"));
            }
        }
        assert!(resolve("nope", WhisperQuant::F16).is_none());
    }

    #[test]
    fn exact_matches() {
        let f = resolve("large-v3-turbo", WhisperQuant::Q5_0).unwrap();
        assert_eq!(f.file, "ggml-large-v3-turbo-q5_0.bin");
        assert_eq!(f.repo, MAIN_REPO);
        assert_eq!(f.substituted_for, None);
        assert_eq!(f.detail(), "q5_0");
        let f = resolve("medium", WhisperQuant::F16).unwrap();
        assert_eq!(f.file, "ggml-medium.bin");
    }

    #[test]
    fn substitutions() {
        let f = resolve("large-v3", WhisperQuant::Q8_0).unwrap();
        assert_eq!(f.file, "ggml-large-v3-q5_0.bin");
        assert_eq!(f.detail(), "q5_0 (q8_0 not available)");
        let f = resolve("small", WhisperQuant::Q5_0).unwrap();
        assert_eq!(f.file, "ggml-small-q5_1.bin");
        assert_eq!(f.detail(), "q5_1 (q5_0 not available)");
        let f = resolve("distil-large-v3", WhisperQuant::Q5_0).unwrap();
        assert_eq!((f.repo, f.file), (DISTIL_REPO, "ggml-distil-large-v3.bin"));
        assert_eq!(f.detail(), "f16 (q5_0 not available)");
    }
}
