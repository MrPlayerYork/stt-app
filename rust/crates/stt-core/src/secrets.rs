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

/// The real system keyring (Secret Service via the `keyring` crate, `v1` mode: zbus on Linux, the same
/// store Python's `keyring` uses here). Entries are service [`SERVICE`], username `user`.
#[derive(Debug, Clone, Copy, Default)]
pub struct KeyringStore;

/// Map a `keyring` error to ours. A missing entry is not an error (callers check for it first).
fn map_keyring_error(err: keyring::Error) -> SecretError {
    match err {
        keyring::Error::NoDefaultStore | keyring::Error::NoStorageAccess(_) => {
            SecretError::Unavailable(err.to_string())
        }
        other => SecretError::Backend(other.to_string()),
    }
}

/// The keyring entry for `user` under `service`.
fn entry(service: &str, user: &str) -> Result<keyring::Entry, SecretError> {
    keyring::Entry::new(service, user).map_err(map_keyring_error)
}

impl SecretStore for KeyringStore {
    fn get(&self, user: &str) -> Result<Option<String>, SecretError> {
        get_in(SERVICE, user)
    }

    fn set(&self, user: &str, secret: &str) -> Result<(), SecretError> {
        set_in(SERVICE, user, secret)
    }

    fn delete(&self, user: &str) -> Result<(), SecretError> {
        delete_in(SERVICE, user)
    }
}

fn get_in(service: &str, user: &str) -> Result<Option<String>, SecretError> {
    match entry(service, user)?.get_password() {
        Ok(secret) => Ok(Some(secret)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(map_keyring_error(e)),
    }
}

fn set_in(service: &str, user: &str, secret: &str) -> Result<(), SecretError> {
    entry(service, user)?
        .set_password(secret)
        .map_err(map_keyring_error)
}

fn delete_in(service: &str, user: &str) -> Result<(), SecretError> {
    match entry(service, user)?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(map_keyring_error(e)),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes to the real Secret Service under a throwaway service name, then deletes the entry.
    /// Ignored by default: it needs a running Secret Service (`cargo test -p stt-core -- --ignored`).
    #[test]
    #[ignore = "touches the real keyring (service stt-app-test-p2-1)"]
    fn real_keyring_round_trip_on_throwaway_service() {
        const SVC: &str = "stt-app-test-p2-1";
        const USER: &str = "roundtrip";
        assert_eq!(
            get_in(SVC, USER).unwrap(),
            None,
            "leftover entry from an earlier run"
        );
        set_in(SVC, USER, "test-value-1").unwrap();
        assert_eq!(get_in(SVC, USER).unwrap().as_deref(), Some("test-value-1"));
        set_in(SVC, USER, "test-value-2").unwrap();
        assert_eq!(get_in(SVC, USER).unwrap().as_deref(), Some("test-value-2"));
        delete_in(SVC, USER).unwrap();
        assert_eq!(get_in(SVC, USER).unwrap(), None);
        delete_in(SVC, USER).unwrap(); // deleting a missing entry is fine
    }
}
