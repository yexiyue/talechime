//! Bounded aggregate metrics. Hosts decide whether and where to install a recorder.
use metrics::{
    Counter, CounterFn, Gauge, GaugeFn, Histogram, HistogramFn, Key, KeyName, Metadata, Recorder,
    SharedString, Unit,
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

#[derive(Default, Serialize)]
struct HistogramState {
    count: u64,
    sum: f64,
    min: Option<f64>,
    max: Option<f64>,
    buckets: [u64; 32],
}

#[derive(Default)]
struct Series {
    counter: AtomicU64,
    gauge: Mutex<Option<f64>>,
    histogram: Mutex<HistogramState>,
}

impl CounterFn for Series {
    fn increment(&self, value: u64) {
        let mut current = self.counter.load(Ordering::Relaxed);
        loop {
            match self.counter.compare_exchange_weak(
                current,
                current.saturating_add(value),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }
    fn absolute(&self, value: u64) {
        self.counter.fetch_max(value, Ordering::Relaxed);
    }
}
impl GaugeFn for Series {
    fn increment(&self, value: f64) {
        if value.is_finite() {
            let mut gauge = self.gauge.lock().unwrap_or_else(|e| e.into_inner());
            let next = gauge.unwrap_or(0.0) + value;
            if next.is_finite() {
                *gauge = Some(next);
            }
        }
    }
    fn decrement(&self, value: f64) {
        GaugeFn::increment(self, -value);
    }
    fn set(&self, value: f64) {
        if value.is_finite() {
            *self.gauge.lock().unwrap_or_else(|e| e.into_inner()) = Some(value);
        }
    }
}
impl HistogramFn for Series {
    fn record(&self, value: f64) {
        if !value.is_finite() || value < 0.0 {
            return;
        }
        let mut state = self.histogram.lock().unwrap_or_else(|e| e.into_inner());
        if !(state.sum + value).is_finite() {
            return;
        }
        state.count = state.count.saturating_add(1);
        state.sum += value;
        state.min = Some(state.min.map_or(value, |n| n.min(value)));
        state.max = Some(state.max.map_or(value, |n| n.max(value)));
        let bin = if value == 0.0 {
            0
        } else {
            (value.log2().floor() + 16.0).clamp(0.0, 31.0) as usize
        };
        state.buckets[bin] = state.buckets[bin].saturating_add(1);
    }
}

#[derive(Clone, Default)]
pub struct BoundedRecorder {
    series: Arc<Mutex<BTreeMap<MetricKey, Arc<Series>>>>,
    rejected: Arc<AtomicU64>,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct MetricKey {
    kind: String,
    name: String,
    labels: Vec<(String, String)>,
}

impl BoundedRecorder {
    fn get(&self, key: &Key, kind: &str) -> Option<Arc<Series>> {
        let allowed_names = [
            "talechime.requests",
            "talechime.errors",
            "talechime.chunks",
            "talechime.request_seconds",
            "talechime.channel_wait_seconds",
            "talechime.audio_budget_reserved_seconds",
        ];
        let mut labels = Vec::new();
        if !allowed_names.contains(&key.name()) {
            self.rejected.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        for label in key.labels() {
            if !["backend", "model", "device", "mode", "stage", "outcome"].contains(&label.key())
                || label.value().len() > 64
                || labels.len() >= 6
            {
                self.rejected.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            labels.push((label.key().to_owned(), label.value().to_owned()));
        }
        labels.sort();
        let identity = MetricKey {
            kind: kind.into(),
            name: key.name().into(),
            labels,
        };
        let mut registry = self.series.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(series) = registry.get(&identity) {
            return Some(series.clone());
        }
        if registry.len() >= 256 {
            self.rejected.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let series = Arc::new(Series::default());
        registry.insert(identity, series.clone());
        Some(series)
    }

    pub fn snapshot(&self) -> serde_json::Value {
        let registry = self.series.lock().unwrap_or_else(|e| e.into_inner());
        let series=registry.iter().map(|(key,state)| serde_json::json!({"kind":key.kind,"name":key.name,"labels":key.labels,"unit":if key.name.ends_with("_seconds"){"seconds"}else{"count"},"counter":state.counter.load(Ordering::Relaxed),"gauge":*state.gauge.lock().unwrap_or_else(|e|e.into_inner()),"histogram":*state.histogram.lock().unwrap_or_else(|e|e.into_inner())})).collect::<Vec<_>>();
        serde_json::json!({"schema":"talechime/performance-metrics/1","series":series,"rejected_registrations":self.rejected.load(Ordering::Relaxed),"max_series":256,"histogram_bounds":"bucket 0 < 2^-15; bucket i=1..30 [2^(i-16),2^(i-15)); bucket 31 >= 2^15; seconds for *_seconds metrics","authoritative_rounds":false})
    }
}

impl Recorder for BoundedRecorder {
    fn describe_counter(&self, _key: KeyName, _unit: Option<Unit>, _description: SharedString) {}
    fn describe_gauge(&self, _key: KeyName, _unit: Option<Unit>, _description: SharedString) {}
    fn describe_histogram(&self, _key: KeyName, _unit: Option<Unit>, _description: SharedString) {}
    fn register_counter(&self, key: &Key, _metadata: &Metadata<'_>) -> Counter {
        self.get(key, "counter")
            .map_or_else(Counter::noop, Counter::from_arc)
    }
    fn register_gauge(&self, key: &Key, _metadata: &Metadata<'_>) -> Gauge {
        self.get(key, "gauge")
            .map_or_else(Gauge::noop, Gauge::from_arc)
    }
    fn register_histogram(&self, key: &Key, _metadata: &Metadata<'_>) -> Histogram {
        self.get(key, "histogram")
            .map_or_else(Histogram::noop, Histogram::from_arc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owner_updates_are_visible_and_unique_request_labels_are_rejected() {
        let recorder = BoundedRecorder::default();
        let metadata = Metadata::new("test", metrics::Level::INFO, None);
        let counter = recorder.register_counter(&Key::from_name("talechime.requests"), &metadata);
        std::thread::spawn(move || counter.increment(3))
            .join()
            .unwrap();
        let histogram =
            recorder.register_histogram(&Key::from_name("talechime.request_seconds"), &metadata);
        for _ in 0..10000 {
            histogram.record(0.5);
        }
        let forbidden = Key::from_parts(
            "talechime.requests",
            vec![metrics::Label::new("request_id", "unbounded")],
        );
        recorder
            .register_counter(&forbidden, &metadata)
            .increment(99);
        let snapshot = recorder.snapshot();
        assert_eq!(snapshot["rejected_registrations"], 1);
        assert_eq!(snapshot["series"].as_array().unwrap().len(), 2);
        assert!(
            snapshot["series"]
                .as_array()
                .unwrap()
                .iter()
                .any(|s| s["counter"] == 3)
        );
        assert_eq!(
            BoundedRecorder::default().snapshot()["series"],
            serde_json::json!([])
        );
    }
}
