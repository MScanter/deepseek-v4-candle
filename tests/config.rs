//! Config deserialization. `max_seq_len` is special: the reference keeps it as a `ModelArgs` field
//! with a default (`4096`) rather than in `config.json`, so the loader must supply that same default
//! when the key is absent and honor it when present. (The E2E test can't catch a wrong value — it
//! only uses positions 0..4 — so it's pinned here.)

use deepseek_v4_candle::Config;
use serde_json::Value;

const TOY_CONFIG: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/toy_config.json");

/// Present in the JSON → read verbatim.
#[test]
fn max_seq_len_read_from_json() {
    let text = std::fs::read_to_string(TOY_CONFIG).expect("read toy_config.json");
    let cfg: Config = serde_json::from_str(&text).expect("parse Config");
    assert_eq!(cfg.max_seq_len, 16);
}

/// Absent from the JSON → defaults to the reference's `ModelArgs` default of 4096.
#[test]
fn max_seq_len_defaults_when_absent() {
    let text = std::fs::read_to_string(TOY_CONFIG).expect("read toy_config.json");
    let mut v: Value = serde_json::from_str(&text).unwrap();
    v.as_object_mut().unwrap().remove("max_seq_len");
    let cfg: Config = serde_json::from_value(v).expect("parse Config without max_seq_len");
    assert_eq!(cfg.max_seq_len, 4096);
}
