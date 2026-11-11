//! UTF-8 byte FST execution, following the fixed WeText/NeMo graphs.
use crate::Result;
use rustfst::{EPS_LABEL, Label, algorithms::compose::compose, prelude::*, utils::acceptor};
use std::{collections::VecDeque, path::Path};

pub struct FstNormalizer(VectorFst<TropicalWeight>);
impl FstNormalizer {
    pub fn load(path: &Path) -> Result<Self> {
        VectorFst::read(path).map(Self).map_err(|error| {
            crate::Error::Invalid(format!("load TN FST {}: {error}", path.display()))
        })
    }
    pub fn apply(&self, input: &str) -> Result<String> {
        self.apply_with_cancel(input, &|| false)
    }
    /// Return the selected text while observing the owning request's stop signal.
    pub fn apply_with_cancel(&self, input: &str, cancelled: &dyn Fn() -> bool) -> Result<String> {
        self.apply_optional_with_cancel(input, cancelled)?
            .ok_or_else(|| crate::Error::Invalid("TN FST has no path for the input".into()))
    }
    pub fn apply_optional(&self, input: &str) -> Result<Option<String>> {
        self.apply_optional_with_cancel(input, &|| false)
    }
    /// Observe cancellation before/after composition and during path search.
    pub fn apply_optional_with_cancel(
        &self,
        input: &str,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<String>> {
        if cancelled() {
            return Err(crate::Error::Cancelled);
        }
        if input.is_empty() {
            return Ok(Some(String::new()));
        }
        let labels: Vec<Label> = input.bytes().map(Label::from).collect();
        let acceptor: VectorFst<TropicalWeight> = acceptor(&labels, TropicalWeight::one());
        let composed: VectorFst<TropicalWeight> = compose::<
            TropicalWeight,
            VectorFst<TropicalWeight>,
            VectorFst<TropicalWeight>,
            VectorFst<TropicalWeight>,
            _,
            _,
        >(&acceptor, &self.0)
        .map_err(|error| crate::Error::Invalid(format!("compose TN FST: {error}")))?;
        exact_output_with_cancel(&composed, cancelled)
    }
}

/// rustfst 1.3.1 TropicalWeight::eq uses KDELTA=1/1024. Official TN
/// grammars distinguish paths by 1e-5..1e-4, so its shortest_path can retain
/// a higher-cost reading. Relax raw F32 costs with strict comparison instead.
/// Detect a negative cycle if a shorter path repeats a predecessor state.
#[cfg(test)]
fn exact_output(fst: &VectorFst<TropicalWeight>) -> Result<Option<String>> {
    exact_output_with_cancel(fst, &|| false)
}
fn exact_output_with_cancel(
    fst: &VectorFst<TropicalWeight>,
    cancelled: &dyn Fn() -> bool,
) -> Result<Option<String>> {
    let Some(start) = fst.start() else {
        return Ok(None);
    };
    let size = fst.num_states();
    let mut distance = vec![f32::INFINITY; size];
    let mut parents: Vec<Option<(StateId, Label)>> = vec![None; size];
    let mut queued = vec![false; size];
    let mut depth = vec![0; size];
    let mut queue = VecDeque::from([start]);
    distance[start as usize] = 0.0;
    queued[start as usize] = true;
    while let Some(state) = queue.pop_front() {
        if cancelled() {
            return Err(crate::Error::Cancelled);
        }
        queued[state as usize] = false;
        for tr in fst.get_trs(state).map_err(fst_error)?.trs() {
            let candidate = distance[state as usize] + tr.weight.value();
            let next = tr.nextstate as usize;
            if candidate < distance[next] {
                depth[next] = depth[state as usize] + 1;
                if depth[next] >= size {
                    crate::fail!("TN FST contains a reachable negative cycle");
                }
                distance[next] = candidate;
                parents[next] = Some((state, tr.olabel));
                if !queued[next] {
                    queue.push_back(tr.nextstate);
                    queued[next] = true;
                }
            }
        }
    }
    let mut best = f32::INFINITY;
    let mut final_state = None;
    for state in fst.states_iter() {
        if let Some(weight) = fst.final_weight(state).map_err(fst_error)? {
            let candidate = distance[state as usize] + weight.value();
            if candidate < best {
                best = candidate;
                final_state = Some(state);
            }
        }
    }
    let Some(mut state) = final_state else {
        return Ok(None);
    };
    let mut bytes = Vec::new();
    while state != start {
        let (parent, label) = parents[state as usize]
            .ok_or_else(|| crate::Error::Invalid("TN shortest path has no predecessor".into()))?;
        if label != EPS_LABEL {
            bytes.push(u8::try_from(label).map_err(|_| {
                crate::Error::Invalid("TN FST contains a non-byte output label".into())
            })?);
        }
        state = parent;
    }
    bytes.reverse();
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|error| crate::Error::Invalid(format!("TN FST output is not UTF-8: {error}")))
}
fn fst_error(error: impl std::fmt::Display) -> crate::Error {
    crate::Error::Invalid(format!("TN FST: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn distinct_sub_delta_costs_choose_the_true_shortest_reading() {
        let mut fst = VectorFst::new();
        let start = fst.add_state();
        let end = fst.add_state();
        fst.set_start(start).unwrap();
        fst.set_final(end, TropicalWeight::one()).unwrap();
        fst.add_tr(
            start,
            Tr::new(1, u32::from(b'a'), TropicalWeight::new(0.0006), end),
        )
        .unwrap();
        fst.add_tr(
            start,
            Tr::new(1, u32::from(b'b'), TropicalWeight::new(0.00041), end),
        )
        .unwrap();
        assert_eq!(exact_output(&fst).unwrap(), Some("b".into()));
    }
    #[test]
    fn no_path_and_negative_cycles_are_distinct() {
        let mut fst = VectorFst::new();
        let start = fst.add_state();
        fst.set_start(start).unwrap();
        assert!(exact_output(&fst).unwrap().is_none());
        fst.add_tr(start, Tr::new(0, 0, TropicalWeight::new(-0.1), start))
            .unwrap();
        assert!(exact_output(&fst).is_err());
    }
    #[test]
    fn long_graph_search_observes_cancellation_at_each_state() {
        let mut fst = VectorFst::new();
        for _ in 0..10_000 {
            fst.add_state();
        }
        fst.set_start(0).unwrap();
        fst.set_final(9_999, TropicalWeight::one()).unwrap();
        for state in 0..9_999 {
            fst.add_tr(
                state,
                Tr::new(0, u32::from(b'x'), TropicalWeight::one(), state + 1),
            )
            .unwrap();
        }
        let calls = std::cell::Cell::new(0);
        let result = exact_output_with_cancel(&fst, &|| {
            calls.set(calls.get() + 1);
            calls.get() > 32
        });
        assert!(matches!(result, Err(crate::Error::Cancelled)));
        assert_eq!(calls.get(), 33);
    }
}
