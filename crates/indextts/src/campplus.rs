//! CAMPPlus speaker embedding (`indextts/s2mel/modules/campplus/DTDNN.py`,
//! `classifier.py`), feat_dim 80 → embedding 192.
//!
//! Input features are 80-bin Kaldi fbank frames (25 ms / 10 ms, dither 0,
//! 16 kHz) with per-dimension mean subtraction, computed by the caller via
//! [`crate::dsp::kaldi_fbank_80`].
//!
//! Structure: the `FCM` 2-D frontend (`head.conv1/bn1`, two stride-2
//! `BasicResBlock` stages, `head.conv2/bn2` collapsing 80 bins to 10) feeds
//! a stride-2 `TDNNLayer`, three `CAMDenseTDNNBlock` growth stages with
//! channel-attention (`CAMLayer`) and `TransitLayer` halving, then the
//! affine eval-mode `out_nonlinear` batch norm + ReLU, statistics pooling
//! (mean + unbiased std) and a final `DenseLayer` projecting to the 192-dim
//! embedding. Weight names are the `campplus_cn_common.bin` state-dict
//! names verbatim.

use candle_core::{Result, Tensor};
use candle_nn::{
    BatchNorm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Module, ModuleT, VarBuilder,
};

/// torch `nn.BatchNorm` default.
const BN_EPS: f64 = 1e-5;
const GROWTH: usize = 32;
const BN_CHANNELS: usize = 4 * GROWTH;
const INIT_CHANNELS: usize = 128;
const M_CHANNELS: usize = 32;
const FEAT_DIM: usize = 80;
const EMBEDDING: usize = 192;

fn bn_affine(vb: &VarBuilder, ch: usize) -> Result<BatchNorm> {
    BatchNorm::new(
        ch,
        vb.get((ch,), "running_mean")?,
        vb.get((ch,), "running_var")?,
        vb.get((ch,), "weight")?,
        vb.get((ch,), "bias")?,
        BN_EPS,
    )
}

/// `get_nonlinear("batchnorm_")`: BatchNorm1d with `affine=False`.
fn bn_eval_only(vb: &VarBuilder, ch: usize) -> Result<BatchNorm> {
    BatchNorm::new_no_bias(
        ch,
        vb.get((ch,), "running_mean")?,
        vb.get((ch,), "running_var")?,
        BN_EPS,
    )
}

fn conv1d(
    vb: &VarBuilder,
    (in_c, out_c): (usize, usize),
    kernel: usize,
    cfg: Conv1dConfig,
    bias: bool,
) -> Result<Conv1d> {
    let w = vb.get((out_c, in_c, kernel), "weight")?;
    let b = if bias {
        Some(vb.get((out_c,), "bias")?)
    } else {
        None
    };
    Ok(Conv1d::new(w, b, cfg))
}

fn conv2d(
    vb: &VarBuilder,
    (in_c, out_c): (usize, usize),
    kernel: usize,
    padding: usize,
    freq_stride: bool,
) -> Result<Conv2dFreqStride> {
    let w = vb.get((out_c, in_c, kernel, kernel), "weight")?;
    Ok(Conv2dFreqStride {
        conv: Conv2d::new(
            w,
            None,
            Conv2dConfig {
                padding,
                ..Default::default()
            },
        ),
        freq_stride,
    })
}

/// candle's `Conv2dConfig` only takes a scalar stride, while the FCM blocks
/// stride `(2, 1)` (frequency only). The conv therefore runs with stride 1
/// and the frequency axis is subsampled afterwards — bit-identical because
/// every output row only depends on its own input rows.
struct Conv2dFreqStride {
    conv: Conv2d,
    freq_stride: bool,
}

impl Conv2dFreqStride {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(x)?;
        if !self.freq_stride {
            return Ok(y);
        }
        let (b, c, f, t) = y.dims4()?;
        if f % 2 != 0 {
            candle_core::bail!("frequency axis {f} cannot be halved");
        }
        // Keep even rows: reshape (b, c, f/2, 2, t) and select slot 0.
        y.reshape((b, c, f / 2, 2, t))?.narrow(3, 0, 1)?.squeeze(3)
    }
}

/// Upstream `seg_pooling`: average pooling over non-overlapping 100-frame
/// segments with `ceil_mode=True` semantics (the trailing partial segment
/// divides by its actual length), each segment mean repeated back over its
/// frames and truncated to the input length.
fn seg_pooling(x: &Tensor) -> Result<Tensor> {
    const SEG: usize = 100;
    let (b, c, t) = x.dims3()?;
    let segs = t.div_ceil(SEG);
    let mut parts = Vec::with_capacity(segs);
    for s in 0..segs {
        let start = s * SEG;
        let len = (t - start).min(SEG);
        parts.push(x.narrow(2, start, len)?.mean_keepdim(2)?);
    }
    Tensor::cat(&parts, 2)?
        .unsqueeze(3)?
        .broadcast_as((b, c, segs, SEG))?
        .reshape((b, c, segs * SEG))?
        .narrow(2, 0, t)
}

struct CamLayer {
    linear_local: Conv1d,
    linear1: Conv1d,
    linear2: Conv1d,
}

impl CamLayer {
    fn load(vb: &VarBuilder, kernel: usize, dilation: usize) -> Result<Self> {
        let kcfg = |padding: usize, dilation: usize| Conv1dConfig {
            padding,
            dilation,
            ..Default::default()
        };
        Ok(Self {
            linear_local: conv1d(
                &vb.pp("linear_local"),
                (BN_CHANNELS, GROWTH),
                kernel,
                kcfg((kernel - 1) / 2 * dilation, dilation),
                false,
            )?,
            linear1: conv1d(
                &vb.pp("linear1"),
                (BN_CHANNELS, BN_CHANNELS / 2),
                1,
                kcfg(0, 1),
                true,
            )?,
            linear2: conv1d(
                &vb.pp("linear2"),
                (BN_CHANNELS / 2, GROWTH),
                1,
                kcfg(0, 1),
                true,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.linear_local.forward(x)?;
        // Global mean (keepdim) plus the piecewise segment means.
        let context = x.mean_keepdim(2)?.broadcast_add(&seg_pooling(x)?)?;
        let context = self.linear1.forward(&context)?.relu()?;
        let m = candle_nn::ops::sigmoid(&self.linear2.forward(&context)?)?;
        y * m
    }
}

struct CamDenseTdnnLayer {
    nonlinear1: BatchNorm,
    linear1: Conv1d,
    nonlinear2: BatchNorm,
    cam_layer: CamLayer,
}

impl CamDenseTdnnLayer {
    fn load(vb: &VarBuilder, in_channels: usize, kernel: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            nonlinear1: bn_affine(&vb.pp("nonlinear1").pp("batchnorm"), in_channels)?,
            linear1: conv1d(
                &vb.pp("linear1"),
                (in_channels, BN_CHANNELS),
                1,
                Default::default(),
                false,
            )?,
            nonlinear2: bn_affine(&vb.pp("nonlinear2").pp("batchnorm"), BN_CHANNELS)?,
            cam_layer: CamLayer::load(&vb.pp("cam_layer"), kernel, dilation)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.nonlinear1.forward_t(x, false)?.relu()?;
        let x = self.linear1.forward(&x)?;
        let x = self.nonlinear2.forward_t(&x, false)?.relu()?;
        self.cam_layer.forward(&x)
    }
}

/// Dense block: each layer's output is concatenated to the input of the
/// next, so channel count grows by `growth` per layer.
struct CamDenseTdnnBlock {
    layers: Vec<CamDenseTdnnLayer>,
}

impl CamDenseTdnnBlock {
    fn load(
        vb: &VarBuilder,
        num_layers: usize,
        in_channels: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self> {
        let layers = (0..num_layers)
            .map(|i| {
                CamDenseTdnnLayer::load(
                    &vb.pp(format!("tdnnd{}", i + 1)),
                    in_channels + i * GROWTH,
                    kernel,
                    dilation,
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            let out = layer.forward(&x)?;
            x = Tensor::cat(&[x, out], 1)?;
        }
        Ok(x)
    }
}

struct TdnnLayer {
    linear: Conv1d,
    nonlinear: BatchNorm,
}

impl TdnnLayer {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let (in_c, out_c, k, stride) = (M_CHANNELS * (FEAT_DIM / 8), INIT_CHANNELS, 5, 2);
        Ok(Self {
            linear: conv1d(
                &vb.pp("linear"),
                (in_c, out_c),
                k,
                Conv1dConfig {
                    stride,
                    padding: (k - 1) / 2,
                    ..Default::default()
                },
                false,
            )?,
            nonlinear: bn_affine(&vb.pp("nonlinear").pp("batchnorm"), out_c)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear.forward(x)?;
        self.nonlinear.forward_t(&x, false)?.relu()
    }
}

struct TransitLayer {
    nonlinear: BatchNorm,
    linear: Conv1d,
}

impl TransitLayer {
    fn load(vb: &VarBuilder, in_channels: usize) -> Result<Self> {
        Ok(Self {
            nonlinear: bn_affine(&vb.pp("nonlinear").pp("batchnorm"), in_channels)?,
            linear: conv1d(
                &vb.pp("linear"),
                (in_channels, in_channels / 2),
                1,
                Default::default(),
                false,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.nonlinear.forward_t(x, false)?.relu()?;
        self.linear.forward(&x)
    }
}

struct DenseLayer {
    linear: Conv1d,
    nonlinear: BatchNorm,
}

impl DenseLayer {
    fn load(vb: &VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        Ok(Self {
            linear: conv1d(
                &vb.pp("linear"),
                (in_channels, out_channels),
                1,
                Default::default(),
                false,
            )?,
            nonlinear: bn_eval_only(&vb.pp("nonlinear").pp("batchnorm"), out_channels)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 2-D input: apply the 1x1 conv on a trailing singleton frame axis.
        let x = if x.dims().len() == 2 {
            self.linear.forward(&x.unsqueeze(2)?)?.squeeze(2)?
        } else {
            self.linear.forward(x)?
        };
        self.nonlinear.forward_t(&x, false)
    }
}

/// `BasicResBlock`: 3x3 convs striding on the frequency axis only.
struct BasicResBlock {
    conv1: Conv2dFreqStride,
    bn1: BatchNorm,
    conv2: Conv2dFreqStride,
    bn2: BatchNorm,
    shortcut: Option<(Conv2dFreqStride, BatchNorm)>,
}

impl BasicResBlock {
    fn load(vb: &VarBuilder, stride2: bool) -> Result<Self> {
        let shortcut = if stride2 {
            Some((
                conv2d(&vb.pp("shortcut.0"), (M_CHANNELS, M_CHANNELS), 1, 0, true)?,
                bn_affine(&vb.pp("shortcut.1"), M_CHANNELS)?,
            ))
        } else {
            None
        };
        Ok(Self {
            conv1: conv2d(&vb.pp("conv1"), (M_CHANNELS, M_CHANNELS), 3, 1, stride2)?,
            bn1: bn_affine(&vb.pp("bn1"), M_CHANNELS)?,
            conv2: conv2d(&vb.pp("conv2"), (M_CHANNELS, M_CHANNELS), 3, 1, false)?,
            bn2: bn_affine(&vb.pp("bn2"), M_CHANNELS)?,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = self.bn1.forward_t(&self.conv1.forward(x)?, false)?.relu()?;
        out = self.bn2.forward_t(&self.conv2.forward(&out)?, false)?;
        let shortcut = match &self.shortcut {
            Some((conv, bn)) => bn.forward_t(&conv.forward(x)?, false)?,
            None => x.clone(),
        };
        (out + shortcut)?.relu()
    }
}

/// FCM 2-D frontend; time stride is always 1, frequency is downsampled
/// 80 -> 40 -> 20 -> 10 before the (C*F, T) reshape.
struct Fcm {
    conv1: Conv2dFreqStride,
    bn1: BatchNorm,
    layer1: Vec<BasicResBlock>,
    layer2: Vec<BasicResBlock>,
    conv2: Conv2dFreqStride,
    bn2: BatchNorm,
}

impl Fcm {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let res_stage =
            |vb: &VarBuilder, name: &str, num_blocks: usize| -> Result<Vec<BasicResBlock>> {
                (0..num_blocks)
                    .map(|i| BasicResBlock::load(&vb.pp(format!("{name}.{i}")), i == 0))
                    .collect()
            };
        Ok(Self {
            conv1: conv2d(&vb.pp("conv1"), (1, M_CHANNELS), 3, 1, false)?,
            bn1: bn_affine(&vb.pp("bn1"), M_CHANNELS)?,
            layer1: res_stage(vb, "layer1", 2)?,
            layer2: res_stage(vb, "layer2", 2)?,
            conv2: conv2d(&vb.pp("conv2"), (M_CHANNELS, M_CHANNELS), 3, 1, true)?,
            bn2: bn_affine(&vb.pp("bn2"), M_CHANNELS)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = self
            .bn1
            .forward_t(&self.conv1.forward(&x.unsqueeze(1)?)?, false)?
            .relu()?;
        for block in &self.layer1 {
            out = block.forward(&out)?;
        }
        for block in &self.layer2 {
            out = block.forward(&out)?;
        }
        let out = self
            .bn2
            .forward_t(&self.conv2.forward(&out)?, false)?
            .relu()?;
        let (b, c, f, t) = out.dims4()?;
        out.reshape((b, c * f, t))
    }
}

/// `statistics_pooling` with `unbiased=True`: mean and sample std over the
/// time axis, concatenated.
fn stats_pool(x: &Tensor) -> Result<Tensor> {
    let t = x.dim(2)?;
    if t < 2 {
        candle_core::bail!("statistics pooling needs at least 2 frames, got {t}");
    }
    let mean = x.mean_keepdim(2)?;
    let centered = x.broadcast_sub(&mean)?;
    let std = (centered.sqr()?.sum_keepdim(2)? / (t - 1) as f64)?.sqrt()?;
    // cat over the channel axis: [mean(0..C), std(0..C)].
    Tensor::cat(&[mean, std], 1)?.squeeze(2)
}

pub struct CampPlus {
    head: Fcm,
    tdnn: TdnnLayer,
    blocks: Vec<(CamDenseTdnnBlock, TransitLayer)>,
    out_nonlinear: BatchNorm,
    dense: DenseLayer,
}

impl CampPlus {
    /// Weight names come from `campplus_cn_common.bin` verbatim.
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let head = Fcm::load(&vb.pp("head"))?;
        let mut channels = INIT_CHANNELS;
        let mut blocks = Vec::with_capacity(3);
        for (i, (num_layers, dilation)) in [(12usize, 1usize), (24, 2), (16, 2)]
            .into_iter()
            .enumerate()
        {
            let bvb = vb.pp(format!("xvector.block{}", i + 1));
            let block = CamDenseTdnnBlock::load(&bvb, num_layers, channels, 3, dilation)?;
            channels += num_layers * GROWTH;
            let transit =
                TransitLayer::load(&vb.pp(format!("xvector.transit{}", i + 1)), channels)?;
            channels /= 2;
            blocks.push((block, transit));
        }
        Ok(Self {
            head,
            tdnn: TdnnLayer::load(&vb.pp("xvector.tdnn"))?,
            blocks,
            out_nonlinear: bn_affine(&vb.pp("xvector.out_nonlinear").pp("batchnorm"), channels)?,
            dense: DenseLayer::load(&vb.pp("xvector.dense"), channels * 2, EMBEDDING)?,
        })
    }

    /// `feat`: `(1, T, 80)` mean-subtracted fbank. Returns `(1, 192)`.
    pub fn forward(&self, feat: &Tensor) -> Result<Tensor> {
        let mut x = feat.transpose(1, 2)?; // (B, T, F) -> (B, F, T)
        x = self.head.forward(&x)?;
        x = self.tdnn.forward(&x)?;
        for (block, transit) in &self.blocks {
            x = block.forward(&x)?;
            x = transit.forward(&x)?;
        }
        // `out_nonlinear` is `get_nonlinear("batchnorm-relu", ch)`: eval-mode
        // affine batch norm followed by ReLU, before statistics pooling.
        x = self.out_nonlinear.forward_t(&x, false)?.relu()?;
        let x = stats_pool(&x)?; // (B, 2 * channels)
        self.dense.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarBuilder as Vb;
    use candle_nn::VarMap;

    fn varmap_vb() -> (VarMap, Vb<'static>) {
        let vm = VarMap::new();
        let vb = Vb::from_varmap(&vm, DType::F32, &candle_core::Device::Cpu);
        (vm, vb)
    }

    #[test]
    fn seg_pooling_piecewise_means() {
        // 250 frames: segment means are mean(0..100), mean(100..200) and
        // mean(200..250) (partial segment divides by 50), tiled to 100
        // frames each and truncated to 250.
        let t = 250;
        let data: Vec<f32> = (0..t).map(|i| i as f32).collect();
        let x = Tensor::from_vec(data, (1, 1, t), &candle_core::Device::Cpu).unwrap();
        let y = seg_pooling(&x).unwrap();
        assert_eq!(y.dims(), vec![1, 1, t]);
        let v = y.to_vec3::<f32>().unwrap();
        let m1: f32 = (0..100).map(|i| i as f32).sum::<f32>() / 100.0;
        let m2: f32 = (100..200).map(|i| i as f32).sum::<f32>() / 100.0;
        let m3: f32 = (200..250).map(|i| i as f32).sum::<f32>() / 50.0;
        for i in 0..t {
            let want = [m1, m2, m3][i / 100];
            assert!((v[0][0][i] - want).abs() < 1e-4, "frame {i}");
        }
    }

    #[test]
    fn stats_pool_unbiased_std() {
        // [1, 2, 3]: mean 2, sample std sqrt(((1+0+1)/2)) = 1.
        let x = Tensor::new(&[[[1f32, 2.0, 3.0]]], &candle_core::Device::Cpu).unwrap();
        let y = stats_pool(&x).unwrap();
        assert_eq!(y.dims(), vec![1, 2]);
        let v = y.to_vec2::<f32>().unwrap();
        assert_eq!(v[0][0], 2.0);
        assert!((v[0][1] - 1.0).abs() < 1e-5);
        assert!(
            stats_pool(&Tensor::zeros((1, 3, 1), DType::F32, &candle_core::Device::Cpu).unwrap())
                .is_err()
        );
    }

    #[test]
    fn campplus_shapes_with_varmap() {
        // VarMap-backed VarBuilder zero-initializes every tensor on first
        // access, exercising the full name map without the real checkpoint.
        let (_vm, vb) = varmap_vb();
        let model = CampPlus::load(&vb).unwrap();
        let feat = Tensor::zeros((1, 200, 80), DType::F32, &candle_core::Device::Cpu).unwrap();
        let out = model.forward(&feat).unwrap();
        assert_eq!(out.dims(), vec![1, EMBEDDING]);
        // Non-trivial length exercises both partial and full segments.
        let feat = Tensor::zeros((1, 243, 80), DType::F32, &candle_core::Device::Cpu).unwrap();
        let out = model.forward(&feat).unwrap();
        assert_eq!(out.dims(), vec![1, EMBEDDING]);
    }

    /// Per-stage comparison against the Python reference dump produced by
    /// `target/indextts-oracle/dump_campplus.py` (forward hooks on the
    /// upstream model). Env-gated so ordinary test runs never touch it.
    #[test]
    #[ignore = "requires INDEXTTS_CONVERTED_DIR, INDEXTTS_ORACLE_JSON, INDEXTTS_CAMPPLUS_LAYERS"]
    fn campplus_stages_match_python() {
        let (Some(dir), Some(layers_path)) = (
            std::env::var_os("INDEXTTS_CONVERTED_DIR"),
            std::env::var_os("INDEXTTS_CAMPPLUS_LAYERS"),
        ) else {
            eprintln!("INDEXTTS_CONVERTED_DIR/INDEXTTS_CAMPPLUS_LAYERS not set; skipping");
            return;
        };
        let oracle_path = std::env::var_os("INDEXTTS_ORACLE_JSON").expect("INDEXTTS_ORACLE_JSON");
        let device = candle_core::Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[std::path::Path::new(&dir).join("campplus.safetensors")],
                DType::F32,
                &device,
            )
        }
        .unwrap();
        let model = CampPlus::load(&vb).unwrap();

        let oracle: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(oracle_path).unwrap()).unwrap();
        let dumps: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(layers_path).unwrap()).unwrap();
        let fbank_entry = &oracle["campplus"]["fbank_mean_sub"];
        let fbank: Vec<f32> = fbank_entry["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let fbank_shape: Vec<usize> = fbank_entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let feat = Tensor::from_vec(fbank, fbank_shape, &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let check = |name: &str, got: &Tensor| {
            let entry = &dumps[name];
            let want: Vec<f32> = entry["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let shape: Vec<usize> = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(got.dims(), shape, "campplus.{name}: shape mismatch");
            let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let mut max_abs = 0f64;
            for (a, w) in g.iter().zip(&want) {
                let d = (*a - *w) as f64;
                max_abs = max_abs.max(d.abs());
                let rel = d.abs() / w.abs().max(1e-6) as f64;
                assert!(
                    d.abs() <= 1e-4 || rel <= 1e-3,
                    "campplus.{name}: FIRST divergent stage, diff {} (rel {rel}) near {w}",
                    d.abs()
                );
            }
            println!(
                "campplus.{name}: max_abs {max_abs:.3e} over {} values",
                want.len()
            );
        };

        // ---- head (FCM) stage by stage ----
        let x = feat.transpose(1, 2).unwrap(); // (B, F, T)
        let c1 = model.head.conv1.forward(&x.unsqueeze(1).unwrap()).unwrap();
        check("head.conv1", &c1);
        let b1 = model.head.bn1.forward_t(&c1, false).unwrap();
        check("head.bn1", &b1);
        let mut out = b1.relu().unwrap();
        out = model.head.layer1[0].forward(&out).unwrap();
        check("head.layer1.0", &out);
        out = model.head.layer1[1].forward(&out).unwrap();
        check("head.layer1.1", &out);
        out = model.head.layer2[0].forward(&out).unwrap();
        check("head.layer2.0", &out);
        out = model.head.layer2[1].forward(&out).unwrap();
        check("head.layer2.1", &out);
        let c2 = model.head.conv2.forward(&out).unwrap();
        let b2 = model.head.bn2.forward_t(&c2, false).unwrap();
        let h = b2.relu().unwrap();
        let (b, c, f, t) = h.dims4().unwrap();
        let h = h.reshape((b, c * f, t)).unwrap();
        check("head", &h);

        // ---- tdnn ----
        let x0 = model.tdnn.forward(&h).unwrap();
        check("tdnn", &x0);

        // ---- first CAMDenseTDNNLayer of block1, sub-stage by sub-stage ----
        let layer = &model.blocks[0].0.layers[0];
        let n1 = layer.nonlinear1.forward_t(&x0, false).unwrap();
        check("b1l1.nonlinear1.bn", &n1);
        let n1 = n1.relu().unwrap();
        check("b1l1.nonlinear1", &n1);
        let l1 = layer.linear1.forward(&n1).unwrap();
        check("b1l1.linear1", &l1);
        let n2 = layer.nonlinear2.forward_t(&l1, false).unwrap();
        check("b1l1.nonlinear2.bn", &n2);
        let n2 = n2.relu().unwrap();
        check("b1l1.nonlinear2", &n2);
        let cam = &layer.cam_layer;
        let y = cam.linear_local.forward(&n2).unwrap();
        check("b1l1.cam.linear_local", &y);
        let context = n2
            .mean_keepdim(2)
            .unwrap()
            .broadcast_add(&seg_pooling(&n2).unwrap())
            .unwrap();
        let c1c = cam.linear1.forward(&context).unwrap();
        check("b1l1.cam.linear1", &c1c);
        let logits = cam.linear2.forward(&c1c.relu().unwrap()).unwrap();
        check("b1l1.cam.linear2", &logits);
        let m = candle_nn::ops::sigmoid(&logits).unwrap();
        let cam_out = (y * m).unwrap();
        check("b1l1.cam", &cam_out);
        check("b1l1", &cam_out);

        // ---- dense blocks + transits ----
        let mut x = x0;
        for (i, (block, transit)) in model.blocks.iter().enumerate() {
            x = block.forward(&x).unwrap();
            check(&format!("block{}", i + 1), &x);
            x = transit.forward(&x).unwrap();
            check(&format!("transit{}", i + 1), &x);
        }
        let on = model.out_nonlinear.forward_t(&x, false).unwrap();
        check("out_nonlinear.bn", &on);
        let on = on.relu().unwrap();
        check("out_nonlinear", &on);
        let st = stats_pool(&on).unwrap();
        check("stats", &st);
        let d = model.dense.forward(&st).unwrap();
        check("dense", &d);
        check("style", &d);

        let full = model.forward(&feat).unwrap();
        check("style", &full);
        // The shared forward must equal the staged pipeline exactly.
        let (a, b) = (
            full.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            d.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        let m: f32 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(m < 1e-5, "campplus: staged vs forward mismatch {m}");
        println!("campplus stages: all within tolerance");
    }

    #[test]
    #[ignore = "requires the converted checkpoint; set INDEXTTS_CONVERTED_DIR"]
    fn campplus_real_weights() {
        let Some(dir) = std::env::var_os("INDEXTTS_CONVERTED_DIR") else {
            eprintln!("INDEXTTS_CONVERTED_DIR not set; skipping");
            return;
        };
        let path = std::path::Path::new(&dir).join("campplus.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &candle_core::Device::Cpu)
        }
        .unwrap();
        let model = CampPlus::load(&vb).unwrap();
        // Zero-variance features would explode the batch norms; use a small
        // deterministic ramp instead.
        let t = 230;
        let data: Vec<f32> = (0..t * 80)
            .map(|i| (((i % 97) as f32 / 97.0) - 0.5) * 2.0)
            .collect();
        let feat = Tensor::from_vec(data, (1, t, 80), &candle_core::Device::Cpu).unwrap();
        let out = model.forward(&feat).unwrap();
        assert_eq!(out.dims(), vec![1, EMBEDDING]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.1, "embedding norm {norm}");
    }
}
