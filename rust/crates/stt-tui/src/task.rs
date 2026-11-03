//! Background work for screens: run a blocking closure on its own thread and poll for the result from
//! `on_tick`, so the UI never blocks on the network, the model or the audio device.

use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;

/// A closure running on a worker thread. Poll it with [`Task::poll`] on every tick.
pub struct Task<T> {
    rx: Receiver<T>,
    done: Option<T>,
    finished: bool,
}

impl<T: Send + 'static> Task<T> {
    /// Start `f` on a new thread.
    pub fn spawn(f: impl FnOnce() -> T + Send + 'static) -> Self {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(f());
        });
        Self {
            rx,
            done: None,
            finished: false,
        }
    }
}

impl<T> Task<T> {
    /// A task that is already finished with `value` (handy in tests and for synchronous fallbacks).
    pub fn ready(value: T) -> Self {
        let (_tx, rx) = mpsc::channel();
        Self {
            rx,
            done: Some(value),
            finished: true,
        }
    }

    /// The result once, when it is ready; `None` while running and after it has been taken.
    pub fn poll(&mut self) -> Option<T> {
        if let Some(v) = self.done.take() {
            return Some(v);
        }
        if self.finished {
            return None;
        }
        match self.rx.try_recv() {
            Ok(v) => {
                self.finished = true;
                Some(v)
            }
            Err(TryRecvError::Empty) => None,
            // The worker panicked; treat it as finished with no result.
            Err(TryRecvError::Disconnected) => {
                self.finished = true;
                None
            }
        }
    }

    /// True once the result was taken (or the worker died).
    pub fn is_finished(&self) -> bool {
        self.finished && self.done.is_none()
    }

    /// Block until the result arrives (tests only; never on the UI thread).
    pub fn wait(mut self) -> Option<T> {
        if let Some(v) = self.done.take() {
            return Some(v);
        }
        if self.finished {
            return None;
        }
        self.rx.recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_then_wait() {
        assert_eq!(Task::spawn(|| 2 + 2).wait(), Some(4));
    }

    #[test]
    fn ready_polls_once() {
        let mut t = Task::ready(7);
        assert_eq!(t.poll(), Some(7));
        assert_eq!(t.poll(), None);
        assert!(t.is_finished());
    }
}
