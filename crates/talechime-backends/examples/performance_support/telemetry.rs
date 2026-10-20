//! Bounded local diagnostics installed only by the developer executable.
use serde::Serialize;
use serde_json::{Map, Value};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};
use tracing::{
    Subscriber,
    span::{Attributes, Id, Record},
};
use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};

#[derive(Default, Clone)]
struct Fields(Map<String, Value>);
impl tracing::field::Visit for Fields {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if self.0.len() < 16 {
            self.0.insert(
                field.name().into(),
                Value::String(value.chars().take(256).collect()),
            );
        }
    }
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        if self.0.len() < 16 {
            self.0.insert(field.name().into(), serde_json::json!(value));
        }
    }
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        if self.0.len() < 16 {
            self.0.insert(field.name().into(), value.into());
        }
    }
    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        if self.0.len() < 16 {
            self.0.insert(field.name().into(), value.into());
        }
    }
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        if self.0.len() < 16 {
            self.0.insert(field.name().into(), value.into());
        }
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.record_str(field, &format!("{value:?}"));
    }
}

#[derive(Clone, Serialize)]
pub struct Raw {
    pub round: u64,
    pub request_span: u64,
    pub name: String,
    pub offset_seconds: f64,
    pub value: Option<f64>,
}

struct State {
    origin: Instant,
    detailed: bool,
    events: Vec<Value>,
    bytes: usize,
    dropped: u64,
    raw: Vec<Raw>,
    raw_dropped: u64,
}
impl State {
    fn event(&mut self, event: Value) {
        if !self.detailed {
            return;
        }
        let bytes = event.to_string().len();
        if self.events.len() >= 8192 || self.bytes + bytes > 16 * 1024 * 1024 {
            self.dropped += 1;
            return;
        }
        self.bytes += bytes;
        self.events.push(event);
    }
}

#[derive(Clone)]
struct LocalLayer(Arc<Mutex<State>>);

fn tid() -> u64 {
    let mut h = DefaultHasher::new();
    std::thread::current().id().hash(&mut h);
    h.finish() & ((1u64 << 53) - 1)
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for LocalLayer {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut fields = Fields::default();
        attrs.record(&mut fields);
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(fields);
        }
    }
    fn on_record(&self, id: &Id, record: &Record<'_>, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id)
            && let Some(fields) = span.extensions_mut().get_mut::<Fields>()
        {
            record.record(fields);
        }
    }
    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        self.span_event(id, ctx, "B");
    }
    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        self.span_event(id, ctx, "E");
    }
    fn on_event(&self, event: &tracing::Event<'_>, ctx: Context<'_, S>) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        let mut round = None;
        let mut request_span = 0;
        if let Some(scope) = ctx.event_scope(event) {
            for span in scope.from_root() {
                if let Some(values) = span.extensions().get::<Fields>()
                    && let Some(value) = values.0.get("round").and_then(Value::as_u64)
                {
                    round = Some(value);
                }
                if span.name() == "segment_request" {
                    request_span = span.id().into_u64();
                }
            }
        }
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let offset = state.origin.elapsed().as_secs_f64();
        if fields.0.get("event").and_then(Value::as_str) == Some("audio_budget_reserved_seconds")
            && let Some(value) = fields.0.get("seconds").and_then(Value::as_f64)
        {
            metrics::gauge!("talechime.audio_budget_reserved_seconds","stage"=>"pending_and_active").set(value);
        }
        if event.metadata().target() == "talechime_perf"
            && let (Some(round), Some(name)) =
                (round, fields.0.get("event").and_then(Value::as_str))
            && [
                "owner_pcm_ready",
                "adapter_channel_enqueue",
                "channel_wait",
                "audio_budget_reserved_seconds",
            ]
            .contains(&name)
        {
            if state.raw.len() < 8192 {
                state.raw.push(Raw {
                    round,
                    request_span,
                    name: name.into(),
                    offset_seconds: offset,
                    value: fields.0.get("seconds").and_then(Value::as_f64),
                });
            } else {
                state.raw_dropped += 1;
            }
        }
        state.event(serde_json::json!({"name":event.metadata().name(),"cat":event.metadata().target(),"ph":"i","s":"t","ts":offset*1e6,"pid":std::process::id(),"tid":tid(),"args":fields.0}));
    }
}

impl LocalLayer {
    fn span_event<S: Subscriber + for<'a> LookupSpan<'a>>(
        &self,
        id: &Id,
        ctx: Context<'_, S>,
        phase: &str,
    ) {
        if let Some(span) = ctx.span(id) {
            let mut args = span
                .extensions()
                .get::<Fields>()
                .cloned()
                .unwrap_or_default()
                .0;
            args.insert("span_id".into(), id.into_u64().into());
            if let Some(parent) = span.parent() {
                args.insert("parent_span_id".into(), parent.id().into_u64().into());
            }
            let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
            let time = state.origin.elapsed().as_secs_f64() * 1e6;
            state.event(serde_json::json!({"name":span.name(),"cat":span.metadata().target(),"ph":phase,"ts":time,"pid":std::process::id(),"tid":tid(),"args":args}));
        }
    }
}

pub struct Telemetry {
    state: Arc<Mutex<State>>,
    recorder: talechime_backends::performance::recorder::BoundedRecorder,
    enabled: bool,
    mode: String,
}

impl Telemetry {
    pub fn install(mode: &str) -> anyhow::Result<Self> {
        anyhow::ensure!(
            ["off", "baseline", "trace", "profile"].contains(&mode),
            "unknown observation mode"
        );
        let detailed = mode == "trace" || mode == "profile";
        let state = Arc::new(Mutex::new(State {
            origin: Instant::now(),
            detailed,
            events: vec![],
            bytes: 0,
            dropped: 0,
            raw: vec![],
            raw_dropped: 0,
        }));
        let recorder = talechime_backends::performance::recorder::BoundedRecorder::default();
        if mode != "off" {
            metrics::set_global_recorder(recorder.clone())
                .map_err(|_| anyhow::anyhow!("metrics recorder already installed"))?;
            tracing_subscriber::registry()
                .with(
                    LocalLayer(state.clone()).with_filter(tracing_subscriber::filter::filter_fn(
                        move |metadata| detailed || metadata.target() == "talechime_perf",
                    )),
                )
                .try_init()?;
        }
        Ok(Self {
            state,
            recorder,
            enabled: mode != "off",
            mode: mode.into(),
        })
    }
    pub fn offset_at(&self, instant: Instant) -> f64 {
        instant
            .duration_since(self.state.lock().unwrap_or_else(|e| e.into_inner()).origin)
            .as_secs_f64()
    }
    pub fn raw(&self, round: u32) -> (Vec<Raw>, u64) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let (selected, remaining) = std::mem::take(&mut state.raw)
            .into_iter()
            .partition(|r| r.round == round as u64);
        state.raw = remaining;
        (selected, state.raw_dropped)
    }
    pub fn finish(&self, output: &Path, completed: bool) -> anyhow::Result<()> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        write(
            output,
            "metrics.json",
            &serde_json::json!({"enabled":self.enabled,"data":self.recorder.snapshot()}),
        )?;
        if state.detailed {
            write(
                output,
                "trace.json",
                &serde_json::json!({"traceEvents":state.events,"displayTimeUnit":"ms"}),
            )?;
        }
        write(
            output,
            "diagnostics.json",
            &serde_json::json!({"mode":self.mode,"trace_complete":state.detailed && completed && state.dropped==0 && state.raw_dropped==0,"dropped_trace_records":state.dropped,"dropped_raw_observations":state.raw_dropped,"flush_complete":true,"source":"host wall/submission spans; not GPU kernel execution","disabled":!self.enabled}),
        )
    }
}

fn write(output: &Path, name: &str, value: &impl Serialize) -> anyhow::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join(name))?;
    serde_json::to_writer(&mut file, value)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_context_and_metrics_cross_owner_threads() -> anyhow::Result<()> {
        let telemetry = Telemetry::install("trace")?;
        let round = tracing::info_span!(target:"talechime_perf","perf_round",round=7u64);
        let segment =
            tracing::info_span!(target:"talechime_perf",parent:&round,"segment_request",call=3u64);
        let owner = segment.in_scope(talechime_backends::performance::OwnerContext::capture);
        std::thread::spawn(move || {
            let _entered = owner.enter();
            tracing::info!(target:"talechime_perf",event="owner_pcm_ready");
            metrics::counter!("talechime.requests","stage"=>"cross_owner_test").increment(1);
        })
        .join()
        .unwrap();
        let (raw, dropped) = telemetry.raw(7);
        assert_eq!(dropped, 0);
        assert!(
            raw.iter()
                .any(|r| r.request_span > 0 && r.name == "owner_pcm_ready")
        );
        assert!(
            telemetry.recorder.snapshot()["series"]
                .as_array()
                .unwrap()
                .iter()
                .any(|s| s["counter"] == 1)
        );
        Ok(())
    }
    #[test]
    fn trace_buffers_are_bounded_and_losses_are_disclosed() {
        let mut state = State {
            origin: Instant::now(),
            detailed: true,
            events: vec![],
            bytes: 0,
            dropped: 0,
            raw: vec![],
            raw_dropped: 0,
        };
        for _ in 0..9000 {
            state.event(serde_json::json!({"name":"fixture","ph":"i"}));
        }
        assert_eq!(state.events.len(), 8192);
        assert_eq!(state.dropped, 808);
    }
}
