//! Session-level loudness alignment across synthesis segments.
//!
//! Segment-local sampling varies audible energy even with the same fixed
//! reference voice. The aligner tracks a slow baseline of
//! voiced loudness and pulls each packet toward it with a bounded, gently
//! smoothed gain: loudness steps between segments become gradual drift while
//! pauses (below the voice floor) freeze tracking so gaps never pump.
use crate::backend::Pcm;

/// Corrective gain bounds, roughly ±3 dB around the tracked baseline.
const MIN_GAIN: f32 = 0.7;
const MAX_GAIN: f32 = 1.4;
/// Packet-level tracking rates for the fast level and slow baseline estimates.
const LEVEL_RATE: f32 = 0.25;
const BASELINE_RATE: f32 = 0.02;
/// How fast the applied gain approaches the corrective target.
const GAIN_RATE: f32 = 0.25;
/// RMS below this is pause, not speech.
const VOICE_FLOOR: f32 = 0.004;
/// Working headroom before the peak guard rescales a packet.
const CEILING: f32 = 0.99;

#[derive(Debug)]
pub(crate) struct LoudnessAligner {
    gain: f32,
    /// Fast RMS estimate of voiced packets; `None` until the first one.
    level: Option<f32>,
    baseline: f32,
}

impl Default for LoudnessAligner {
    fn default() -> Self {
        Self {
            gain: 1.0,
            level: None,
            baseline: 0.0,
        }
    }
}

impl LoudnessAligner {
    /// Apply the ramped gain to one packet. Raw input PCM is never modified;
    /// the caller receives a fresh owned buffer.
    pub(crate) fn push(&mut self, pcm: &Pcm) -> Pcm {
        let rms = rms(&pcm.samples);
        if rms > VOICE_FLOOR {
            match self.level {
                None => {
                    self.level = Some(rms);
                    self.baseline = rms;
                }
                Some(level) => {
                    let level = level + (rms - level) * LEVEL_RATE;
                    self.level = Some(level);
                    self.baseline += (level - self.baseline) * BASELINE_RATE;
                }
            }
        }
        let desired = match self.level {
            Some(level) if level > f32::EPSILON => {
                (self.baseline / level).clamp(MIN_GAIN, MAX_GAIN)
            }
            _ => 1.0,
        };
        let from = self.gain;
        let to = from + (desired - from) * GAIN_RATE;
        self.gain = to;
        // Linear ramp within the packet keeps gain changes click-free.
        let count = pcm.samples.len();
        let step = if count > 1 {
            (to - from) / (count - 1) as f32
        } else {
            0.0
        };
        let mut peak = 0f32;
        let mut samples = Vec::with_capacity(count);
        for (index, sample) in pcm.samples.iter().enumerate() {
            let value = sample * (from + step * index as f32);
            peak = peak.max(value.abs());
            samples.push(value);
        }
        if peak > CEILING {
            let scale = CEILING / peak;
            for sample in &mut samples {
                *sample *= scale;
            }
        }
        Pcm {
            samples,
            sample_rate: pcm.sample_rate,
            channels: pcm.channels,
        }
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    ((samples.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / samples.len() as f64).sqrt())
        as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    fn packet(level: f32, len: usize) -> Pcm {
        Pcm {
            samples: vec![level; len],
            sample_rate: 1000,
            channels: 1,
        }
    }
    fn feed(aligner: &mut LoudnessAligner, level: f32, packets: usize) -> Vec<f32> {
        let mut peaks = Vec::new();
        for _ in 0..packets {
            let out = aligner.push(&packet(level, 480));
            peaks.push(out.samples.iter().copied().fold(0f32, f32::max));
        }
        peaks
    }
    #[test]
    fn on_target_and_silent_audio_keep_unity_gain() {
        let mut aligner = LoudnessAligner::default();
        for peak in feed(&mut aligner, 0.2, 8) {
            assert!((peak - 0.2).abs() < 1e-6, "{peak}");
        }
        for peak in feed(&mut aligner, 0.0, 4) {
            assert_eq!(peak, 0.0);
        }
        for peak in feed(&mut aligner, 0.2, 4) {
            assert!((peak - 0.2).abs() < 1e-6, "{peak}");
        }
    }
    #[test]
    fn quiet_segments_boost_toward_the_baseline() {
        let mut aligner = LoudnessAligner::default();
        feed(&mut aligner, 0.2, 10);
        let peaks = feed(&mut aligner, 0.05, 40);
        let last = peaks.last().copied().unwrap();
        assert!(last > 0.062, "expected a boost above 0.062, got {last}");
        assert!(last <= 0.05 * MAX_GAIN + 1e-6, "{last}");
    }
    #[test]
    fn loud_segments_attenuate_toward_the_baseline() {
        let mut aligner = LoudnessAligner::default();
        feed(&mut aligner, 0.2, 10);
        let peaks = feed(&mut aligner, 0.7, 40);
        let last = peaks.last().copied().unwrap();
        // The baseline itself drifts up over a long loud run, so attenuation
        // settles at the gain floor rather than at the initial ratio.
        assert!(last < 0.55, "expected attenuation below 0.55, got {last}");
        assert!(last >= 0.7 * MIN_GAIN - 1e-3, "{last}");
    }
    #[test]
    fn peak_guard_rescales_boosted_transients() {
        let mut aligner = LoudnessAligner::default();
        feed(&mut aligner, 0.4, 10);
        // Quiet run drives the gain toward MAX_GAIN, then a loud transient
        // must stay under the ceiling instead of clipping.
        feed(&mut aligner, 0.15, 40);
        let peaks = feed(&mut aligner, 0.8, 3);
        for peak in peaks {
            assert!(peak <= CEILING + 1e-6, "{peak}");
        }
    }
}
