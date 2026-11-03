//! Settings › LLM › Custom prompts: list, add/edit (`a`), delete (`d`).

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::app::App;
use crate::screen::{Hint, Nav, Screen};
use crate::theme;
use crate::widgets::{Confirm, PickList, TextPrompt};

use super::commit;

pub(super) struct PromptsScreen;

impl PromptsScreen {
    pub fn new() -> Self {
        Self
    }
}

fn add_or_edit() -> Nav {
    Nav::Push(Box::new(
        TextPrompt::new("Prompt name", "", |app, name| {
            let name = name.trim().to_string();
            if name.is_empty() {
                return Nav::Stay;
            }
            let cur = app
                .cfg
                .llm
                .custom_prompts
                .get(&name)
                .cloned()
                .unwrap_or_default();
            Nav::Push(Box::new(
                TextPrompt::new("System prompt", &cur, move |app, text| {
                    let text = text.trim().to_string();
                    if !text.is_empty() {
                        app.cfg.llm.custom_prompts.insert(name, text);
                        commit(app);
                    }
                    Nav::Stay
                })
                .help("The instruction the model follows in Prompt Mode."),
            ))
        })
        .help("An existing name edits that prompt."),
    ))
}

fn delete(app: &App) -> Nav {
    let names: Vec<String> = app.cfg.llm.custom_prompts.keys().cloned().collect();
    if names.is_empty() {
        return Nav::Stay;
    }
    Nav::Push(Box::new(PickList::new(
        "Delete which",
        names.clone(),
        None,
        move |_app, i| {
            let name = names[i].clone();
            Nav::Push(Box::new(Confirm::new(
                format!("Delete '{name}'?"),
                move |app| {
                    app.cfg.llm.custom_prompts.shift_remove(&name);
                    if app.cfg.llm.preset == name {
                        app.cfg.llm.preset = "deramble".into();
                    }
                    commit(app);
                    Nav::Stay
                },
            )))
        },
    )))
}

impl Screen for PromptsScreen {
    fn render(&mut self, app: &App, frame: &mut Frame, area: Rect) {
        let prompts = &app.cfg.llm.custom_prompts;
        let lines: Vec<Line> = if prompts.is_empty() {
            vec![Line::styled(
                "No custom prompts yet. Press a to add one.",
                theme::dim(),
            )]
        } else {
            prompts
                .iter()
                .map(|(n, p)| {
                    Line::from(vec![
                        Span::styled(n.clone(), Style::new().fg(theme::KEY)),
                        Span::raw(": "),
                        Span::raw(theme::ellipsize(p, 70)),
                    ])
                })
                .collect()
        };
        frame.render_widget(Paragraph::new(lines), area);
    }

    fn on_key(&mut self, app: &mut App, key: KeyEvent) -> Nav {
        match key.code {
            KeyCode::Char('a') => add_or_edit(),
            KeyCode::Char('d') => delete(app),
            KeyCode::Esc => Nav::Pop,
            _ => Nav::Stay,
        }
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        vec![
            ("a", "add/edit".into()),
            ("d", "delete".into()),
            ("Esc", "back".into()),
        ]
    }

    fn title(&self, _app: &App) -> String {
        "LLM › Custom prompts".into()
    }
}
