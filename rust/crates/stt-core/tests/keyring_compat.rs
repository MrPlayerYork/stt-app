//! Real-keyring compatibility with the Python app: Rust must read the keys Python stored under service
//! `stt-app`. Ignored by default (it talks to the real Secret Service). Read-only: never sets or deletes the
//! real entries. Prints only `user: present|absent len=N sha256=<8 hex>`, never a value.
//!
//! Run: `cargo test -p stt-core --test keyring_compat -- --ignored --nocapture`

use sha2::{Digest, Sha256};
use stt_core::secrets::{KeyringStore, SecretStore};

const USERS: &[&str] = &[
    "openai",
    "anthropic",
    "google",
    "openai_compat",
    "huggingface",
];

#[test]
#[ignore = "reads the real keyring (service stt-app)"]
fn print_keyring_fingerprints() {
    let store = KeyringStore;
    for user in USERS {
        let line = match store.get(user) {
            Ok(Some(secret)) => {
                let digest = Sha256::digest(secret.as_bytes());
                let hex: String = digest.iter().take(4).map(|b| format!("{b:02x}")).collect();
                format!("{user}: present len={} sha256={hex}", secret.len())
            }
            Ok(None) => format!("{user}: absent"),
            Err(e) => format!(
                "{user}: error {}",
                e.to_string().split(':').next().unwrap_or("")
            ),
        };
        println!("{line}");
    }
}
