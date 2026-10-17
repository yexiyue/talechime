//! Seedable RNG utilities.
//!
//! Two concerns live here:
//!
//! 1. [`TorchCpuRng`] reproduces `torch.randn` / `torch.rand` on the CPU
//!    with the generic ATen CPU algorithm for a given seed. CosyVoice3 bakes two noise buffers into
//!    its inference path (the CFM `rand_noise` created under seed 0 and the
//!    SineGen2 fixed noise), so matching PyTorch exactly is what makes
//!    fixture comparisons meaningful. The algorithm mirrors ATen's CPU
//!    `normal_fill` (16-wide in-place Box-Muller over a uniform fill) and
//!    `uniform_real_distribution<float>` (`(u32 & 0xFFFFFF) * 2^-24`) on the
//!    MT19937 engine with standard seeding.
//! 2. [`StepRng`] is a small PCG-style generator used for LLM sampling where
//!    cross-framework bit parity does not exist (PyTorch samples through
//!    `multinomial`), but reproducible seeds do.

use candle_core::{Result, Tensor};

/// MT19937 as implemented by ATen (`MT19937RNGEngine.h`), 32-bit output.
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; 624];
        state[0] = (seed & 0xFFFF_FFFF) as u32;
        for i in 1..624 {
            let prev = state[i - 1];
            state[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7FFF_FFFF);
                let mut next = y >> 1;
                if y & 1 != 0 {
                    next ^= 0x9908_B0DF;
                }
                self.state[i] = self.state[(i + 397) % 624] ^ next;
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^= y >> 18;
        y
    }

    /// `at::uniform_real_distribution<float>(0, 1)`.
    fn next_uniform_f32(&mut self) -> f32 {
        (self.next_u32() & 0x00FF_FFFF) as f32 * (1.0 / (1 << 24) as f32)
    }
}

/// Reproduces CPU `torch.randn(..., generator seeded with `seed`)` and
/// `torch.rand(...)` for float tensors.
pub struct TorchCpuRng {
    engine: Mt19937,
    /// Cached second half of a Box-Muller pair, as ATen keeps per generator
    /// for draws shorter than 16 elements.
    cached_normal: Option<f64>,
}

impl TorchCpuRng {
    pub fn new(seed: u64) -> Self {
        Self {
            engine: Mt19937::new(seed),
            cached_normal: None,
        }
    }

    /// `torch.rand(n)` as f32.
    pub fn rand_vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.engine.next_uniform_f32()).collect()
    }

    /// `torch.randn(n)` as f32, following ATen `normal_fill`.
    pub fn randn_vec(&mut self, n: usize) -> Vec<f32> {
        if n < 16 {
            return (0..n).map(|_| self.scalar_normal() as f32).collect();
        }
        let mut data = self.rand_vec(n);
        for block in data.as_chunks_mut::<16>().0 {
            normal_fill_16(block);
        }
        // ATen redraws the final overlapping block, rather than filling
        // a scalar remainder. This also advances the generator by 16.
        if !n.is_multiple_of(16) {
            for v in &mut data[n - 16..] {
                *v = self.engine.next_uniform_f32();
            }
            normal_fill_16(&mut data[n - 16..]);
        }
        data
    }

    fn uniform_f64(&mut self) -> f64 {
        let bits = (u64::from(self.engine.next_u32()) << 32) | u64::from(self.engine.next_u32());
        (bits & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64
    }

    /// Small contiguous draws use ATen's cached double normal sampler.
    fn scalar_normal(&mut self) -> f64 {
        if let Some(c) = self.cached_normal.take() {
            return c;
        }
        let u1 = self.uniform_f64();
        let u2 = self.uniform_f64();
        let r = (-2.0 * (-u2).ln_1p()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u1;
        self.cached_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Fill a tensor of `shape` with seed-0-style `torch.randn` values.
    pub fn randn_tensor(
        shape: &[usize],
        seed: u64,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let mut rng = Self::new(seed);
        let data = rng.randn_vec(n);
        Tensor::from_vec(data, shape, device)
    }
}

/// ATen `normal_fill_16`: in-place Box-Muller over one 16-element block.
fn normal_fill_16(data: &mut [f32]) {
    for j in 0..8 {
        let u1 = 1.0 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = (2.0 * std::f64::consts::PI * f64::from(u2)) as f32;
        data[j] = radius * theta.cos();
        data[j + 8] = radius * theta.sin();
    }
}

/// Small deterministic generator for sampling; PCG-XSH-RR 64→32.
#[derive(Clone)]
pub struct StepRng {
    state: u64,
    inc: u64,
}

impl StepRng {
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (seed << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(0x853C_49E6_748F_EA9D);
        rng.next_u32();
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_reference_values() {
        // Standard MT19937 test vectors for seed 5489 (the canonical
        // default seed), first outputs: 3499211612, 581869302, ...
        let mut rng = Mt19937::new(5489);
        assert_eq!(rng.next_u32(), 3_499_211_612);
        assert_eq!(rng.next_u32(), 581_869_302);
        assert_eq!(rng.next_u32(), 3_890_346_734);
    }

    #[test]
    fn randn_matches_manual_normal_fill() {
        // Cross-check the vectorized path against a scalar reference of the
        // same ATen algorithm: uniform fill, then 16-block transform.
        let mut rng = TorchCpuRng::new(0);
        let a = rng.randn_vec(32);
        let mut rng2 = TorchCpuRng::new(0);
        let uniforms: Vec<f32> = (0..32).map(|_| rng2.engine.next_uniform_f32()).collect();
        let mut b = uniforms;
        normal_fill_16(&mut b[0..16]);
        normal_fill_16(&mut b[16..32]);
        assert_eq!(a, b);
    }

    #[test]
    fn randn_matches_independent_pytorch_cpu_oracle() {
        let fixtures: std::collections::BTreeMap<String, Vec<f32>> =
            serde_json::from_str(include_str!("../tests/fixtures/torch-cpu-rng.json")).unwrap();
        for (n, expected) in fixtures {
            let n = n.parse::<usize>().unwrap();
            let actual = TorchCpuRng::new(0).randn_vec(n);
            for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
                assert!((a - b).abs() < 2e-6, "draw {n}, element {i}: {a} != {b}");
            }
        }
    }

    #[test]
    fn step_rng_deterministic() {
        let mut a = StepRng::new(42);
        let mut b = StepRng::new(42);
        for _ in 0..8 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
        assert_ne!(StepRng::new(42).next_u32(), StepRng::new(43).next_u32());
    }
}
