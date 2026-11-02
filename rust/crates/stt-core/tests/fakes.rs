//! The fakes satisfy the traits as trait objects (object safety + `Send`/`Sync`) and behave as documented.

use std::sync::{Arc, Mutex};

use stt_core::fakes::*;
use stt_core::secrets::{self, HF_TOKEN_USER};
use stt_core::*;

fn assert_send<T: Send + ?Sized>() {}
fn assert_send_sync<T: Send + Sync + ?Sized>() {}

#[test]
fn trait_objects_are_thread_safe() {
    assert_send_sync::<dyn Transcriber>();
    assert_send_sync::<dyn TranscriberFactory>();
    assert_send_sync::<dyn AudioSource>();
    assert_send::<dyn Recorder>();
    assert_send::<dyn MultiRecorder>();
    assert_send_sync::<dyn LlmBackend>();
    assert_send_sync::<dyn LlmFactory>();
    assert_send_sync::<dyn Clipboard>();
    assert_send_sync::<dyn SecretStore>();
}

#[test]
fn fake_transcriber() {
    let t: Arc<dyn Transcriber> = Arc::new(FakeTranscriber::new("  hello world "));
    assert!(t.info().is_none());
    assert_eq!(t.transcribe(&[]).unwrap(), "", "empty audio does not load");
    assert!(t.info().is_none());

    let events = Arc::new(Mutex::new(Vec::new()));
    let sink = events.clone();
    let cb: LoadProgressFn = Arc::new(move |e: &LoadEvent| sink.lock().unwrap().push(e.clone()));
    let info = t.load(Some(cb)).unwrap();
    assert_eq!(info.description(), "large-v3-turbo on gpu (Fake)");
    assert!(matches!(
        events.lock().unwrap().last(),
        Some(LoadEvent::Ready(_))
    ));

    // Loading in another thread returns the same info without reloading.
    let t2 = t.clone();
    assert_eq!(
        std::thread::spawn(move || t2.load(None).unwrap())
            .join()
            .unwrap(),
        info
    );
    assert_eq!(t.transcribe(&[0.1; 16]).unwrap(), "hello world");
    t.unload();
    assert!(t.info().is_none());
}

#[test]
fn fake_transcriber_fallback_and_failure() {
    let t = FakeTranscriber::new("x").gpu_fails("out of VRAM");
    let info = t.load(None).unwrap();
    assert_eq!(info.device, ComputeDevice::Cpu);
    assert!(info.description().contains("GPU failed: out of VRAM"));

    let bad = FakeTranscriber::new("x").failing_load("boom");
    assert!(matches!(
        bad.transcribe(&[0.0; 4]),
        Err(TranscribeError::Load { .. })
    ));

    let per_clip = FakeTranscriber::with_fn(|a| Ok(format!("{} samples", a.len())));
    assert_eq!(per_clip.transcribe(&[0.0; 3]).unwrap(), "3 samples");
    assert_eq!(per_clip.calls(), [3]);
}

#[test]
fn fake_transcriber_factory() {
    let f: Box<dyn TranscriberFactory> = Box::new(FakeTranscriberFactory::new("hi"));
    let mut cfg = AppConfig::default();
    cfg.whisper.model = "small".into();
    let t = f.create(&cfg, &MemorySecretStore::new()).unwrap();
    assert_eq!(t.model_name(), "small");
}

#[test]
fn fake_llm_and_factory() {
    let llm = FakeLlm::new("Clean text.")
        .models(&["a", "b"])
        .server_kind(ServerKind::LmStudio);
    let factory = FakeLlmFactory::new(llm.clone());
    let f: &dyn LlmFactory = &factory;

    let mut cfg = AppConfig::default().llm;
    assert!(matches!(f.create(&cfg, None), Err(LlmError::NoBaseUrl)));
    cfg.base_url = "http://localhost:1234/v1".into();
    let backend = f.create(&cfg, None).unwrap();
    assert_eq!(backend.complete("sys", "raw").unwrap().text, "Clean text.");
    assert_eq!(backend.test_connection("sys").unwrap().text, "Clean text.");
    assert_eq!(backend.list_models().unwrap(), ["a", "b"]);
    assert_eq!(backend.server_kind(), ServerKind::LmStudio);
    backend.unload().unwrap();
    assert_eq!(llm.unload_count(), 1);
    assert_eq!(llm.calls()[0], ("sys".to_string(), "raw".to_string()));
    assert_eq!(llm.calls()[1].1, llm::TEST_SAMPLE_TEXT);

    cfg.provider = Provider::Anthropic;
    cfg.api_key_env = "ANTHROPIC_API_KEY".into();
    let err = f.create(&cfg, None).err().unwrap();
    assert_eq!(
        err.to_string(),
        "no API key set for Anthropic (env var ANTHROPIC_API_KEY)"
    );
    assert!(f.create(&cfg, Some("sk-x")).is_ok());

    let failing = FakeLlm::new("x").failing("refused");
    assert!(failing.complete("s", "t").is_err());
}

#[test]
fn fake_audio_source() {
    let src = FakeAudioSource::new()
        .device("USB Mic", false, vec![0.5; 10])
        .device("Monitor of Speakers", true, vec![0.0; 10])
        .device("Broken", false, vec![])
        .failing_open("Broken")
        .level(0.25);
    let src: Box<dyn AudioSource> = Box::new(src);
    let devs = src.list_input_devices().unwrap();
    assert_eq!(devs.len(), 3);
    assert!(devs[1].is_monitor);
    assert_eq!(find_device(&devs, "usb").unwrap().name, "USB Mic");

    let mut rec = src.recorder(None).unwrap();
    assert_eq!(rec.level(), 0.0);
    rec.start().unwrap();
    assert!(rec.is_recording());
    assert_eq!(rec.level(), 0.25);
    assert_eq!(rec.stop().unwrap().len(), 10);
    assert!(
        rec.stop().unwrap().is_empty(),
        "stop when idle returns nothing"
    );

    let mut multi = src.multi_recorder(&devs);
    let failed = multi.start();
    assert_eq!(
        failed.keys().collect::<Vec<_>>(),
        [&DeviceId("Broken".into())]
    );
    assert_eq!(multi.levels().len(), 2);
    let clips = multi.stop();
    assert_eq!(clips.len(), 2);
    assert_eq!(clips[&DeviceId("USB Mic".into())], vec![0.5; 10]);
}

#[test]
fn fake_clipboard() {
    let cb = FakeClipboard::new();
    let c: &dyn Clipboard = &cb;
    assert_eq!(c.copy("one").unwrap(), ClipboardMethod::WlCopy);
    c.copy("two").unwrap();
    assert_eq!(cb.contents().as_deref(), Some("two"));
    assert_eq!(cb.copies(), ["one", "two"]);
    assert!(FakeClipboard::failing().copy("x").is_err());
}

#[test]
fn secrets_keyring_first_then_env() {
    let store = MemorySecretStore::with(&[("anthropic", "from-keyring")]);
    // SAFETY: tests in this binary touch only this uniquely named variable.
    unsafe { std::env::set_var("STT_CORE_TEST_KEY", "from-env") };
    assert_eq!(
        secrets::get_api_key(&store, "anthropic", "STT_CORE_TEST_KEY").as_deref(),
        Some("from-keyring")
    );
    assert_eq!(
        secrets::get_api_key(&store, "google", "STT_CORE_TEST_KEY").as_deref(),
        Some("from-env")
    );
    assert_eq!(secrets::get_api_key(&store, "google", ""), None);
    assert_eq!(
        secrets::get_api_key(&store, HF_TOKEN_USER, "STT_CORE_TEST_UNSET"),
        None
    );
    secrets::set_api_key(&store, "google", "k").unwrap();
    assert_eq!(
        secrets::get_api_key(&store, "google", "").as_deref(),
        Some("k")
    );
}
