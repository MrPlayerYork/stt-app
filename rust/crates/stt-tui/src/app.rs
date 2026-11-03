//! Shared application state ([`App`]) and the services it uses ([`Services`]).
//!
//! Rule for every screen: **derive what you draw from `App` on each render**; keep only screen-local UI
//! state (selection, text being edited, a running [`crate::task::Task`]) in the screen itself. Settings
//! changes go straight into `app.cfg` followed by [`App::save`], so every tab sees them immediately.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use stt_core::secrets::get_api_key;
use stt_core::{
    AppConfig, AudioSource, Clipboard, ConfigError, LlmFactory, LlmResult, Provider, SecretStore,
    TranscriberFactory,
};

use crate::mic::Mic;
use crate::model::ModelLoader;

/// Where the config is saved. The binary uses [`FileConfigStore`]; tests use [`MemoryConfigStore`].
pub trait ConfigStore: Send + Sync {
    /// Persist `cfg` (atomically for the real file).
    fn save(&self, cfg: &AppConfig) -> Result<(), ConfigError>;
    /// The file path, for display in settings; `None` if not file-backed.
    fn path(&self) -> Option<PathBuf>;
}

/// The real config file.
pub struct FileConfigStore {
    /// Path written by [`ConfigStore::save`].
    pub path: PathBuf,
}

impl ConfigStore for FileConfigStore {
    fn save(&self, cfg: &AppConfig) -> Result<(), ConfigError> {
        stt_core::config::save_config_to(cfg, &self.path)
    }
    fn path(&self) -> Option<PathBuf> {
        Some(self.path.clone())
    }
}

/// In-memory config store that records every save (tests).
#[derive(Default, Clone)]
pub struct MemoryConfigStore {
    saves: Arc<Mutex<Vec<AppConfig>>>,
}

impl MemoryConfigStore {
    /// Every config saved so far, oldest first.
    pub fn saves(&self) -> Vec<AppConfig> {
        self.saves.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl ConfigStore for MemoryConfigStore {
    fn save(&self, cfg: &AppConfig) -> Result<(), ConfigError> {
        self.saves
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(cfg.clone());
        Ok(())
    }
    fn path(&self) -> Option<PathBuf> {
        None
    }
}

/// Everything concrete the UI talks to. The binary wires the real ones; tests use `stt_core::fakes`.
#[derive(Clone)]
pub struct Services {
    /// Microphones.
    pub audio: Arc<dyn AudioSource>,
    /// Builds the Whisper transcriber from settings.
    pub transcribers: Arc<dyn TranscriberFactory>,
    /// Builds LLM backends from settings.
    pub llm: Arc<dyn LlmFactory>,
    /// System clipboard.
    pub clipboard: Arc<dyn Clipboard>,
    /// Keyring.
    pub secrets: Arc<dyn SecretStore>,
    /// Config persistence.
    pub config: Arc<dyn ConfigStore>,
}

/// The three tabs, in display order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tab {
    /// Status, record, last result, recent dictations.
    Main,
    /// Full history with preview and re-copy.
    History,
    /// All settings.
    Settings,
}

impl Tab {
    /// All tabs, in order.
    pub const ALL: [Tab; 3] = [Tab::Main, Tab::History, Tab::Settings];

    /// Tab title.
    pub fn title(self) -> &'static str {
        match self {
            Self::Main => "Main",
            Self::History => "History",
            Self::Settings => "Settings",
        }
    }

    /// Next tab (wraps).
    pub fn next(self) -> Self {
        Self::ALL[(self.index() + 1) % 3]
    }

    /// Previous tab (wraps).
    pub fn prev(self) -> Self {
        Self::ALL[(self.index() + 2) % 3]
    }

    /// Position in [`Self::ALL`].
    pub fn index(self) -> usize {
        Self::ALL.iter().position(|t| *t == self).unwrap_or(0)
    }
}

/// The three modes, in [m]ode cycling order (Python `MODE_ORDER`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeKey {
    /// No LLM.
    Raw,
    /// LLM cleanup.
    Cleanup,
    /// LLM Prompt Mode.
    Prompt,
}

/// One dictation (Python `HistoryItem`, plus the time).
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryItem {
    /// Local time it finished.
    pub at: chrono::DateTime<chrono::Local>,
    /// Whisper's transcript.
    pub raw: String,
    /// What was copied (LLM output, or `raw`).
    pub final_text: String,
    /// Mode label at the time, e.g. "Prompt: Deramble", "Raw (LLM failed)".
    pub mode: String,
    /// LLM reasoning text, `""` if none.
    pub reasoning: String,
}

/// Severity of a [`Flash`] message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashKind {
    /// Green.
    Ok,
    /// Yellow.
    Warn,
    /// Red.
    Error,
}

/// A short status message shown in the frame for a few seconds ("Copied to clipboard", errors).
#[derive(Debug, Clone, PartialEq)]
pub struct Flash {
    /// Text.
    pub text: String,
    /// Colour.
    pub kind: FlashKind,
    /// When it was set.
    pub at: Instant,
}

/// How long a flash message stays visible.
pub const FLASH_FOR: Duration = Duration::from_secs(4);

/// Shared state of the running app.
pub struct App {
    /// Current settings (save with [`App::save`] after changing).
    pub cfg: AppConfig,
    /// Concrete services.
    pub services: Services,
    /// Whisper model + status.
    pub model: ModelLoader,
    /// Hot mic.
    pub mic: Mic,
    /// Newest first, at most `max(1, cfg.history_size)` entries.
    pub history: VecDeque<HistoryItem>,
    /// Every dictation this session, oldest first (printed to the terminal on quit).
    pub session_log: Vec<HistoryItem>,
    /// Full LLM result of the newest dictation (`None` for Raw or when the LLM failed).
    pub last_result: Option<LlmResult>,
    /// Set once this session has sent text to the LLM (decides unload-on-exit).
    pub llm_used: bool,
    /// Active tab.
    pub tab: Tab,
    /// Transient message.
    pub flash: Option<Flash>,
    /// Set to leave the event loop.
    pub quit: bool,
}

impl App {
    /// New app state. Does not start the model or the mic; call [`App::start_loading`] and
    /// [`App::open_mic`] (the binary does, after first-run).
    pub fn new(cfg: AppConfig, services: Services) -> Self {
        Self {
            cfg,
            services,
            model: ModelLoader::default(),
            mic: Mic::default(),
            history: VecDeque::new(),
            session_log: Vec::new(),
            last_result: None,
            llm_used: false,
            tab: Tab::Main,
            flash: None,
            quit: false,
        }
    }

    /// Save the config; on failure show a red flash (Python: "Could not save config: ...").
    pub fn save(&mut self) {
        if let Err(e) = self.services.config.save(&self.cfg) {
            self.flash_error(format!("Could not save config: {e}"));
        }
    }

    /// (Re)create and load the Whisper model in the background from the current settings.
    pub fn start_loading(&mut self) {
        self.model.start(
            self.services.transcribers.as_ref(),
            &self.cfg,
            self.services.secrets.as_ref(),
        );
    }

    /// (Re)open the configured microphone in the background (hot mic).
    pub fn open_mic(&mut self) {
        let name = self.cfg.audio.device_name.clone();
        self.mic.set_gain(self.cfg.audio.gain);
        self.mic.open(self.services.audio.clone(), name.as_deref());
    }

    /// Advance background work (model load, mic open, flash expiry). The runner calls it every tick.
    pub fn tick(&mut self) {
        self.model.tick();
        self.mic.tick();
        if self
            .flash
            .as_ref()
            .is_some_and(|f| f.at.elapsed() > FLASH_FOR)
        {
            self.flash = None;
        }
    }

    /// Show a message for a few seconds.
    pub fn flash(&mut self, kind: FlashKind, text: impl Into<String>) {
        self.flash = Some(Flash {
            text: text.into(),
            kind,
            at: Instant::now(),
        });
    }

    /// Green flash.
    pub fn flash_ok(&mut self, text: impl Into<String>) {
        self.flash(FlashKind::Ok, text);
    }

    /// Red flash.
    pub fn flash_error(&mut self, text: impl Into<String>) {
        self.flash(FlashKind::Error, text);
    }

    /// Current mode (Python `mode_key`).
    pub fn mode_key(&self) -> ModeKey {
        let llm = &self.cfg.llm;
        if !llm.enabled {
            ModeKey::Raw
        } else if llm.mode == stt_core::LlmMode::Prompt {
            ModeKey::Prompt
        } else {
            ModeKey::Cleanup
        }
    }

    /// "Raw", "Cleanup" or "Prompt: <preset label>" (Python `mode_label`).
    pub fn mode_label(&self) -> String {
        match self.mode_key() {
            ModeKey::Raw => "Raw".into(),
            ModeKey::Cleanup => "Cleanup".into(),
            ModeKey::Prompt => {
                let presets = stt_llm::all_presets(&self.cfg.llm);
                let label = presets
                    .iter()
                    .find(|(k, _)| *k == self.cfg.llm.preset)
                    .map_or(self.cfg.llm.preset.clone(), |(_, l)| l.clone());
                format!("Prompt: {label}")
            }
        }
    }

    /// " · reasoning: low · word choice" (Python `mode_extras`); `""` in Raw mode.
    pub fn mode_extras(&self) -> String {
        let llm = &self.cfg.llm;
        if !llm.enabled {
            return String::new();
        }
        let mut out = String::new();
        if llm.reasoning != stt_core::ReasoningLevel::Default {
            out.push_str(&format!(" · reasoning: {}", llm.reasoning));
        }
        if llm.word_choice && self.mode_key() == ModeKey::Prompt {
            out.push_str(" · word choice");
        }
        out
    }

    /// A model is chosen and, for `openai_compat`, a base URL (Python `llm_configured`).
    pub fn llm_configured(&self) -> bool {
        let llm = &self.cfg.llm;
        !llm.model.is_empty()
            && (llm.provider != Provider::OpenAiCompat || !llm.base_url.is_empty())
    }

    /// API key for the current provider: keyring, then `llm.api_key_env`.
    pub fn api_key(&self) -> Option<String> {
        get_api_key(
            self.services.secrets.as_ref(),
            self.cfg.llm.provider.keyring_user(),
            &self.cfg.llm.api_key_env,
        )
    }

    /// Record a finished dictation (newest first, trimmed to `history_size`).
    pub fn push_history(&mut self, item: HistoryItem) {
        self.session_log.push(item.clone());
        self.history.push_front(item);
        self.history.truncate(self.cfg.history_size.max(1));
    }

    /// On quit: free the model, close the mic, and unload the local LLM if `unload_on_exit` is set and
    /// the LLM was used this session (Python `unload_llm`). Returns a warning to print, if any. Blocking.
    pub fn shutdown(&mut self) -> Option<String> {
        self.mic.close();
        self.model.unload();
        let llm = &self.cfg.llm;
        if !(llm.unload_on_exit
            && self.llm_used
            && llm.provider == Provider::OpenAiCompat
            && !llm.model.is_empty())
        {
            return None;
        }
        let key = self.api_key();
        let res = self
            .services
            .llm
            .create(llm, key.as_deref())
            .and_then(|b| b.unload());
        res.err().map(|e| format!("Could not unload the LLM: {e}"))
    }
}
