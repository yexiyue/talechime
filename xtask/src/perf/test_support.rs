//! Synthetic evidence fixtures; no models or user data.
use super::{
    evidence::{VERSION, digest},
    records::Attempt,
};

pub fn attempts(times: &[f64]) -> Vec<Attempt> {
    times.iter().enumerate().map(|(index, time)| serde_json::from_value(serde_json::json!({
        "schema": VERSION, "run_id":"fixture-run", "case_id":"fixture-case", "attempt_id":format!("fixture-{index}"), "process_index": index,
        "source":{"revision":"fixture", "dirty_sha256":digest(b"clean"), "untracked":[]}, "binary_sha256":digest(b"binary"),
        "conditions":{
            "machine":"fixture", "os":"windows-fixture", "arch":"x86_64", "hardware":"synthetic", "driver":"not_applicable", "provider":"cpu", "math_backend":"synthetic", "precision":"f32", "features":[], "threads":{"rayon":1}, "backend":"fixture", "model":"deterministic", "resource_sha256":digest(b"resource"), "corpus_sha256":digest(b"corpus"), "conditioning_sha256":digest(b"voice"), "parameters":{"seed":"42"}, "mode":"stream", "observation":"baseline", "timing_contract":"consumer_pcm_to_explicit_end/1"
        }, "expected_hot_rounds":1,
        "rounds":[
            {"index":0,"warmup":true,"outcome":"success","synthesis_seconds":100.0,"audio_seconds":2.0,"first_pcm_seconds":1.0,"output_sha256":digest(b"pcm"),"observations":[],"artifacts":[]},
            {"index":1,"warmup":false,"outcome":"success","synthesis_seconds":time,"audio_seconds":2.0,"first_pcm_seconds":time/2.0,"output_sha256":digest(b"pcm"),"observations":[],"artifacts":[]}
        ], "outcome":"success", "cleanup_verified":true,"trace_complete":false,"dropped_trace_records":null
    })).unwrap()).collect()
}
