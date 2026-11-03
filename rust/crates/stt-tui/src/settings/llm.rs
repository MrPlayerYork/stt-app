//! Settings › LLM (A8).

use ratatui::style::Style;
use ratatui::text::Line;
use stt_core::config::LOCAL_PRESETS;
use stt_core::llm::TEST_SAMPLE_TEXT;
use stt_core::secrets::set_api_key;
use stt_core::{Provider, ReasoningLevel};

use crate::app::App;
use crate::screen::Nav;
use crate::theme;
use crate::widgets::{PickList, TextPrompt};

use super::popups::{Busy, Info};
use super::prompts::PromptsScreen;
use super::section::{Row, Section};
use super::{commit, cycle_in, parse_duration, ttl_label};

pub(super) struct LlmSection;

const DEFAULT_URL: &str = "http://localhost:1234/v1";

fn yes_no(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

fn reasoning_desc(l: ReasoningLevel) -> &'static str {
    match l {
        ReasoningLevel::Default => "don't send anything; the model decides",
        ReasoningLevel::Off => "ask the model not to reason (if supported)",
        _ => "more reasoning = slower, possibly better",
    }
}

fn server_label(url: &str) -> String {
    match LOCAL_PRESETS.iter().find(|(_, u)| *u == url) {
        Some((n, u)) => format!("{n} ({u})"),
        None if url.is_empty() => "not set".into(),
        None => url.to_string(),
    }
}

impl Section for LlmSection {
    fn title(&self) -> String {
        "LLM".into()
    }

    fn rows(&self, app: &App) -> Vec<Row> {
        let l = &app.cfg.llm;
        let key = match (app.api_key().is_some(), l.provider.needs_api_key()) {
            (true, _) => "set".to_string(),
            (false, true) => "not set".to_string(),
            (false, false) => "not set (optional for local servers)".to_string(),
        } + &if l.api_key_env.is_empty() {
            String::new()
        } else {
            format!(" (env {})", l.api_key_env)
        };
        let server = if l.provider == Provider::OpenAiCompat {
            server_label(&l.base_url)
        } else {
            "(only for local servers)".into()
        };
        let prompts = l.custom_prompts.len();
        vec![
            Row::new('1', "Enabled", yes_no(l.enabled)).choice(),
            Row::new(
                '2',
                "Provider",
                l.provider.label().split(" (").next().unwrap_or_default(),
            ),
            Row::new('3', "Server", server),
            Row::new('4', "API key", key),
            Row::new(
                '5',
                "Key env var",
                if l.api_key_env.is_empty() {
                    "-"
                } else {
                    &l.api_key_env
                },
            ),
            Row::new(
                '6',
                "Model",
                if l.model.is_empty() { "-" } else { &l.model },
            ),
            Row::new('7', "Reasoning", l.reasoning.as_str()).choice(),
            Row::new(
                '8',
                "Word choice in Prompt Mode",
                if l.word_choice { "on" } else { "off" },
            )
            .choice(),
            Row::new('9', "Idle unload (TTL)", ttl_label(l.idle_ttl_seconds)),
            Row::new('u', "Unload model when quitting", yes_no(l.unload_on_exit)).choice(),
            Row::new('c', "Custom prompts", format!("{prompts} defined")),
            Row::new('t', "Test connection", ""),
        ]
    }

    fn help(&self, app: &App, i: usize) -> String {
        match i {
            0 => "Turn LLM post-processing on or off. It needs a provider, a server (local) and a model first.".into(),
            1 => "Local servers (Ollama, LM Studio, vLLM, llama-server) or a cloud provider. Switching resets the model to the provider's default.".into(),
            2 => "Where the local server listens. Only used by the local / OpenAI-compatible provider.".into(),
            3 => "Stored in the system keyring and never shown. Enter opens a masked input.".into(),
            4 => "Optional environment variable holding the key, used when the keyring has none. Blank clears it.".into(),
            5 => "Pick from the server's model list, or type a name.".into(),
            6 => format!(
                "{}. Local models only honor this if they support it.",
                reasoning_desc(app.cfg.llm.reasoning)
            ),
            7 => "Prompt Mode may replace words that clearly don't fit the context; Cleanup always stays word-for-word.".into(),
            8 => "TTL: LM Studio applies it only to models it loads on demand (JIT); models loaded manually or via lms keep no TTL. Seconds, 90s, 2m, 1h; 0 = server default.".into(),
            9 => "Ask the local server to unload the model when the app quits (only if the LLM was used this session).".into(),
            10 => "Your own Prompt Mode prompts. Enter opens the list.".into(),
            _ => "Sends a sample sentence through the current settings and shows the result.".into(),
        }
    }

    fn activate(&mut self, app: &mut App, i: usize) -> Nav {
        match i {
            0 | 6 | 7 | 9 => {
                self.cycle(app, i, 1);
                Nav::Stay
            }
            1 => provider_picker(app),
            2 => {
                if app.cfg.llm.provider == Provider::OpenAiCompat {
                    server_picker(app)
                } else {
                    app.flash(
                        crate::app::FlashKind::Warn,
                        "Only used by the local / OpenAI-compatible provider.",
                    );
                    Nav::Stay
                }
            }
            3 => {
                let provider = app.cfg.llm.provider;
                Nav::Push(Box::new(
                    TextPrompt::secret(
                        format!("API key for {}", provider.label()),
                        move |app, key| {
                            if key.is_empty() {
                                return Nav::Stay;
                            }
                            let store = app.services.secrets.clone();
                            Nav::Push(Box::new(Busy::new(
                                "Storing the key…",
                                move || set_api_key(store.as_ref(), provider.keyring_user(), &key),
                                |app, res| {
                                    match res {
                                        Ok(()) => app.flash_ok("Key stored."),
                                        Err(e) => app.flash_error(format!(
                                            "Could not store the key in the keyring: {e}"
                                        )),
                                    }
                                    Nav::Stay
                                },
                            )))
                        },
                    )
                    .help("Paste, then Enter. Enter alone keeps the current key."),
                ))
            }
            4 => {
                let cur = app.cfg.llm.api_key_env.clone();
                Nav::Push(Box::new(
                    TextPrompt::new("Env var holding the API key", &cur, |app, s| {
                        app.cfg.llm.api_key_env = s.trim().to_string();
                        commit(app);
                        Nav::Stay
                    })
                    .help("Blank clears it."),
                ))
            }
            5 => model_picker(app),
            8 => {
                let secs = app.cfg.llm.idle_ttl_seconds;
                let cur = if secs == 0 {
                    "0".to_string()
                } else {
                    ttl_label(secs)
                };
                Nav::Push(Box::new(
                    TextPrompt::new("Idle TTL", &cur, |app, s| {
                        match parse_duration(&s) {
                            Some(n) => {
                                app.cfg.llm.idle_ttl_seconds = n;
                                commit(app);
                            }
                            None => app
                                .flash_error("Enter seconds (120) or a duration like 90s, 2m, 1h."),
                        }
                        Nav::Stay
                    })
                    .help("Seconds or e.g. 2m, 1h. 0 = server default."),
                ))
            }
            10 => Nav::Push(Box::new(PromptsScreen::new())),
            _ => test_connection(app),
        }
    }

    fn cycle(&mut self, app: &mut App, i: usize, dir: i32) {
        match i {
            0 => {
                if !app.cfg.llm.enabled && !app.llm_configured() {
                    app.flash(
                        crate::app::FlashKind::Warn,
                        "Choose a provider, server and model first.",
                    );
                    return;
                }
                app.cfg.llm.enabled = !app.cfg.llm.enabled;
            }
            6 => {
                let l = &mut app.cfg.llm;
                l.reasoning = cycle_in(ReasoningLevel::ALL, l.reasoning, dir);
            }
            7 => app.cfg.llm.word_choice = !app.cfg.llm.word_choice,
            9 => app.cfg.llm.unload_on_exit = !app.cfg.llm.unload_on_exit,
            _ => return,
        }
        commit(app);
    }

    fn enter_hint(&self) -> &'static str {
        "edit"
    }
}

fn provider_picker(app: &App) -> Nav {
    let labels: Vec<String> = Provider::ALL
        .iter()
        .map(|p| p.label().to_string())
        .collect();
    let cur = Provider::ALL
        .iter()
        .position(|p| *p == app.cfg.llm.provider);
    Nav::Push(Box::new(PickList::new(
        "Provider",
        labels,
        cur,
        |app, i| {
            let p = Provider::ALL[i];
            if p == app.cfg.llm.provider {
                return Nav::Stay;
            }
            app.cfg.llm.provider = p;
            app.cfg.llm.model = p.default_model().to_string();
            commit(app);
            if p == Provider::OpenAiCompat {
                return server_picker(app);
            }
            Nav::Stay
        },
    )))
}

fn server_picker(app: &App) -> Nav {
    let mut labels: Vec<String> = LOCAL_PRESETS
        .iter()
        .map(|(n, u)| format!("{n} ({u})"))
        .collect();
    labels.push("Custom URL".into());
    let cur = LOCAL_PRESETS
        .iter()
        .position(|(_, u)| *u == app.cfg.llm.base_url);
    Nav::Push(Box::new(PickList::new("Server", labels, cur, |app, i| {
        if let Some((_, url)) = LOCAL_PRESETS.get(i) {
            app.cfg.llm.base_url = url.to_string();
            commit(app);
            return Nav::Stay;
        }
        let cur = if app.cfg.llm.base_url.is_empty() {
            DEFAULT_URL.to_string()
        } else {
            app.cfg.llm.base_url.clone()
        };
        Nav::Push(Box::new(TextPrompt::new("Base URL", &cur, |app, s| {
            app.cfg.llm.base_url = s.trim().to_string();
            commit(app);
            Nav::Stay
        })))
    })))
}

fn model_name_prompt(app: &App) -> Nav {
    let l = &app.cfg.llm;
    let cur = if l.model.is_empty() {
        l.provider.default_model().to_string()
    } else {
        l.model.clone()
    };
    Nav::Push(Box::new(TextPrompt::new("Model name", &cur, |app, s| {
        if !s.trim().is_empty() {
            app.cfg.llm.model = s.trim().to_string();
            commit(app);
        }
        Nav::Stay
    })))
}

fn model_picker(app: &App) -> Nav {
    let factory = app.services.llm.clone();
    let cfg = app.cfg.llm.clone();
    let key = app.api_key();
    Nav::Push(Box::new(Busy::new(
        "Fetching models…",
        move || {
            factory
                .create(&cfg, key.as_deref())
                .and_then(|b| b.list_models())
                .map_err(|e| e.to_string())
        },
        |app, res| match res {
            Ok(models) if !models.is_empty() => {
                let mut shown: Vec<String> = models.into_iter().take(40).collect();
                let cur = shown.iter().position(|m| *m == app.cfg.llm.model);
                let n = shown.len();
                shown.push("Type a model name…".into());
                Nav::Push(Box::new(PickList::new(
                    "Model",
                    shown.clone(),
                    cur,
                    move |app, i| {
                        if i >= n {
                            return model_name_prompt(app);
                        }
                        app.cfg.llm.model = shown[i].clone();
                        commit(app);
                        Nav::Stay
                    },
                )))
            }
            Ok(_) => model_name_prompt(app),
            Err(e) => {
                app.flash_error(format!("Could not list models: {e}"));
                model_name_prompt(app)
            }
        },
    )))
}

fn test_connection(app: &mut App) -> Nav {
    app.llm_used = true;
    let factory = app.services.llm.clone();
    let cfg = app.cfg.llm.clone();
    let key = app.api_key();
    Nav::Push(Box::new(Busy::new(
        "Testing…",
        move || {
            stt_llm::process(factory.as_ref(), &cfg, key.as_deref(), TEST_SAMPLE_TEXT)
                .map_err(|e| e.to_string())
        },
        |_app, res| {
            let mut lines = vec![
                Line::styled(format!("Input: {TEST_SAMPLE_TEXT}"), theme::dim()),
                Line::default(),
            ];
            match res {
                Ok(r) => {
                    lines.push(Line::styled("Result", theme::label()));
                    let text = if r.text.is_empty() {
                        "(empty response)".to_string()
                    } else {
                        r.text.clone()
                    };
                    lines.push(Line::styled(text, Style::new().fg(theme::OK)));
                    lines.push(Line::default());
                    lines.push(theme::reasoning_line(&r));
                }
                Err(e) => lines.push(Line::styled(
                    format!("Test failed: {e}"),
                    Style::new().fg(theme::ERR),
                )),
            }
            Nav::Push(Box::new(Info::new("Test connection", lines)))
        },
    )))
}
