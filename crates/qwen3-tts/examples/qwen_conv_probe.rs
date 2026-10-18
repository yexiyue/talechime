//! Development-only microbenchmark for 12Hz decoder convolution geometries.
use anyhow::{Context, ensure};
use candle_core::{Device, Tensor};
use qwen3_tts::{models::codec, parse_device};
use serde_json::json;
use std::{fs, path::Path, time::Instant};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        (2..=4).contains(&args.len()),
        "usage: DEVICE OUTPUT_DIR [ROUNDS] [CODEC_FRAMES]"
    );
    let device = parse_device(&args[0])?;
    let rounds: usize = args.get(2).map_or(Ok(3), |s| s.parse()).context("rounds")?;
    let frames: usize = args
        .get(3)
        .map_or(Ok(268), |s| s.parse())
        .context("codec frames")?;
    ensure!(
        rounds > 0 && frames > 0,
        "rounds and codec frames must be positive"
    );

    let directory = Path::new(&args[1]);
    fs::create_dir_all(directory)?;
    let mut records = Vec::new();
    for case in convolution_cases(frames) {
        let mut samples = Vec::new();
        let output = run_case(&device, &case, rounds, &mut samples)?;
        samples.sort_by(|a, b| a.total_cmp(b));
        let record = json!({
            "name": case.name,
            "kind": case.kind.label(),
            "input_channels": case.input_channels,
            "output_channels": case.output_channels,
            "kernel": case.kernel,
            "dilation": case.dilation,
            "input_frames": case.length,
            "rounds": rounds,
            "min_ms": samples[0],
            "p50_ms": samples[samples.len() / 2],
            "mean_ms": samples.iter().sum::<f64>() / samples.len() as f64,
            "max_ms": *samples.last().expect("non-empty"),
            "finite_output": output,
        });
        println!("{record}");
        records.push(record);
    }
    fs::write(
        directory.join("summary.json"),
        serde_json::to_vec_pretty(&records)?,
    )?;
    Ok(())
}

#[derive(Clone, Copy)]
enum ConvKind {
    Dense,
    Transposed,
}

impl ConvKind {
    fn label(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Transposed => "transposed",
        }
    }
}

struct ConvCase {
    name: &'static str,
    kind: ConvKind,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    length: usize,
}

fn convolution_cases(frames: usize) -> Vec<ConvCase> {
    vec![
        ConvCase {
            name: "pre_conv_k3",
            kind: ConvKind::Dense,
            input_channels: 512,
            output_channels: 1024,
            kernel: 3,
            dilation: 1,
            stride: 1,
            length: frames,
        },
        ConvCase {
            name: "pre_upsample0_k2",
            kind: ConvKind::Transposed,
            input_channels: 1024,
            output_channels: 1024,
            kernel: 2,
            dilation: 1,
            stride: 2,
            length: frames,
        },
        ConvCase {
            name: "pre_upsample1_k2",
            kind: ConvKind::Transposed,
            input_channels: 1024,
            output_channels: 1024,
            kernel: 2,
            dilation: 1,
            stride: 2,
            length: frames * 2,
        },
        ConvCase {
            name: "decoder_init_k7",
            kind: ConvKind::Dense,
            input_channels: 1024,
            output_channels: 1536,
            kernel: 7,
            dilation: 1,
            stride: 1,
            length: frames * 4,
        },
        ConvCase {
            name: "block1_trans_k16",
            kind: ConvKind::Transposed,
            input_channels: 1536,
            output_channels: 768,
            kernel: 16,
            dilation: 1,
            stride: 8,
            length: frames * 4,
        },
        ConvCase {
            name: "block1_res_k7_d1",
            kind: ConvKind::Dense,
            input_channels: 768,
            output_channels: 768,
            kernel: 7,
            dilation: 1,
            stride: 1,
            length: frames * 32,
        },
        ConvCase {
            name: "block1_res_k7_d3",
            kind: ConvKind::Dense,
            input_channels: 768,
            output_channels: 768,
            kernel: 7,
            dilation: 3,
            stride: 1,
            length: frames * 32,
        },
        ConvCase {
            name: "block1_res_k7_d9",
            kind: ConvKind::Dense,
            input_channels: 768,
            output_channels: 768,
            kernel: 7,
            dilation: 9,
            stride: 1,
            length: frames * 32,
        },
        ConvCase {
            name: "block2_trans_k10",
            kind: ConvKind::Transposed,
            input_channels: 768,
            output_channels: 384,
            kernel: 10,
            dilation: 1,
            stride: 5,
            length: frames * 32,
        },
        ConvCase {
            name: "block2_res_k7_d1",
            kind: ConvKind::Dense,
            input_channels: 384,
            output_channels: 384,
            kernel: 7,
            dilation: 1,
            stride: 1,
            length: frames * 160,
        },
        ConvCase {
            name: "block3_trans_k8",
            kind: ConvKind::Transposed,
            input_channels: 384,
            output_channels: 192,
            kernel: 8,
            dilation: 1,
            stride: 4,
            length: frames * 160,
        },
        ConvCase {
            name: "block3_res_k7_d1",
            kind: ConvKind::Dense,
            input_channels: 192,
            output_channels: 192,
            kernel: 7,
            dilation: 1,
            stride: 1,
            length: frames * 640,
        },
        ConvCase {
            name: "block4_trans_k6",
            kind: ConvKind::Transposed,
            input_channels: 192,
            output_channels: 96,
            kernel: 6,
            dilation: 1,
            stride: 3,
            length: frames * 640,
        },
        ConvCase {
            name: "block4_res_k7_d1",
            kind: ConvKind::Dense,
            input_channels: 96,
            output_channels: 96,
            kernel: 7,
            dilation: 1,
            stride: 1,
            length: frames * 1920,
        },
        ConvCase {
            name: "final_conv_k7",
            kind: ConvKind::Dense,
            input_channels: 96,
            output_channels: 1,
            kernel: 7,
            dilation: 1,
            stride: 1,
            length: frames * 1920,
        },
    ]
}

fn run_case(
    device: &Device,
    case: &ConvCase,
    rounds: usize,
    samples: &mut Vec<f64>,
) -> anyhow::Result<bool> {
    let input = deterministic_tensor(case.input_channels * case.length, 0xA17, case.name, device)?
        .reshape((1, case.input_channels, case.length))?;
    let output = match case.kind {
        ConvKind::Dense => {
            let weight_shape = (case.output_channels, case.input_channels, case.kernel);
            let weight = deterministic_tensor(
                weight_shape.0 * weight_shape.1 * weight_shape.2,
                0x5EED,
                case.name,
                device,
            )?
            .reshape(weight_shape)?;
            let bias = deterministic_tensor(case.output_channels, 0xB1A5, case.name, device)?;
            let conv = codec::CausalConv1d::from_weights(weight, Some(bias), case.dilation)?;
            run_rounds(device, rounds, samples, || conv.forward(&input))?
        }
        ConvKind::Transposed => {
            let weight_shape = (case.input_channels, case.output_channels, case.kernel);
            let weight = deterministic_tensor(
                weight_shape.0 * weight_shape.1 * weight_shape.2,
                0x7E5,
                case.name,
                device,
            )?
            .reshape(weight_shape)?;
            let bias = deterministic_tensor(case.output_channels, 0xC0DE, case.name, device)?;
            let conv = codec::CausalTransConv1d::from_weights(weight, Some(bias), case.stride)?;
            run_rounds(device, rounds, samples, || conv.forward(&input))?
        }
    };
    let values = output.flatten_all()?.to_vec1::<f32>()?;
    Ok(values.iter().copied().all(f32::is_finite))
}

fn run_rounds<T>(
    device: &Device,
    rounds: usize,
    samples: &mut Vec<f64>,
    mut forward: impl FnMut() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let mut output = forward()?;
    device.synchronize()?;
    for _ in 0..rounds {
        let started = Instant::now();
        output = forward()?;
        device.synchronize()?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(output)
}

fn deterministic_tensor(
    count: usize,
    seed: u32,
    salt: &str,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let salt = salt.bytes().map(usize::from).sum::<usize>() as u32;
    let mut state = seed ^ salt;
    let values: Vec<f32> = (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((state >> 8) & 0xffff) as f32 / 1_048_576.0 - 0.03125
        })
        .collect();
    Ok(Tensor::from_vec(values, count, device)?)
}
