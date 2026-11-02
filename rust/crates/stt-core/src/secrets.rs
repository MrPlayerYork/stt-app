//! API keys and tokens in the system keyring, shared with the Python app.
//!
//! Entries are keyring service [`SERVICE`] (`"stt-app"`), username = the provider's config string
//! (`"openai"`, `"openai_compat"`, `"anthropic"`, `"google"`) or [`HF_TOKEN_USER`] for the Hugging Face
//! token. The Python app uses `keyring`'s Secret Service backend on this machine, so the Rust
//! [`KeyringStore`] must read the same Secret Service items (attributes `service` + `username`).
//!
//! Lookup order ([`get_api_key`]): keyring first, then the named env var. Keyring errors are treated as
//! "no key" during lookup (Python does the same), but surface as [`SecretError`] when storing.
//!
//! UI code takes a `&dyn SecretStore` so tests can use `fakes::MemorySecretStore`.

use crate::error::SecretError;

/// Keyring service name (same as the Python app).
pub const SERVICE: &str = crate::APP_NAME;

/// Keyring username of the Hugging Face token.
pub const HF_TOKEN_USER: &str = "huggingface";

/// Env var fallback for the Hugging Face token.
pub const HF_TOKEN_ENV: &str = "HF_TOKEN";

/// A place secrets are stored, keyed by username under [`SERVICE`].
pub trait SecretStore: Send + Sync {
    /// The stored secret for `user`, `Ok(None)` if there is none.
    fn get(&self, user: &str) -> Result<Option<String>, SecretError>;

    /// Store (replace) the secret for `user`.
    fn set(&self, user: &str, secret: &str) -> Result<(), SecretError>;

    /// Remove the secret for `user`; removing a missing entry is not an error.
    fn delete(&self, user: &str) -> Result<(), SecretError>;
}

/// The real system keyring (Secret Service via the `keyring` crate).
///
/// **Stub, owned by P2-1**: method bodies are `todo!()`. Implement with `keyring` 4.x in its `v1` mode
/// (`keyring::Entry::new(SERVICE, user)?.get_password()`; zbus Secret Service on Linux), map
/// `keyring::Error::NoEntry` to `Ok(None)` and `NoDefaultStore`/platform failures to
/// [`SecretError::Unavailable`], and verify it reads a key the Python app stored.
#[derive(Debug, Clone, Copy, Default)]
pub struct KeyringStore;

impl SecretStore for KeyringStore {
    fn get(&self, user: &str) -> Result<Option<String>, SecretError> {
        let _ = user;
        todo!("P2-1: KeyringStore::get")
    }

    fn set(&self, user: &str, secret: &str) -> Result<(), SecretError> {
        let _ = (user, secret);
        todo!("P2-1: KeyringStore::set")
    }

    fn delete(&self, user: &str) -> Result<(), SecretError> {
        let _ = user;
        todo!("P2-1: KeyringStore::delete")
    }
}

/// Look up a secret: the keyring entry for `user` if present and non-empty, else the env var `env_name`
/// (when `env_name` is non-empty and the variable is set and non-empty), else `None`.
///
/// Python parity: `get_api_key(provider, env_name)`. Use `(provider.keyring_user(), &cfg.llm.api_key_env)`
/// for LLM keys and `(HF_TOKEN_USER, HF_TOKEN_ENV)` for the Hugging Face token.
pub fn get_api_key(store: &dyn SecretStore, user: &str, env_name: &str) -> Option<String> {
    if let Ok(Some(key)) = store.get(user)
        && !key.is_empty()
    {
        return Some(key);
    }
    if env_name.is_empty() {
        return None;
    }
    std::env::var(env_name).ok().filter(|v| !v.is_empty())
}

/// Store a secret for `user` (Python `set_api_key`).
pub fn set_api_key(store: &dyn SecretStore, user: &str, secret: &str) -> Result<(), SecretError> {
    store.set(user, secret)
}
