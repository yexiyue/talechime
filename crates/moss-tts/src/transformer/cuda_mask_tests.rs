use super::*;
use candle_core::Device;

thread_local! {
    pub(super) static FORCE_MASKED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn with_original_mask<T>(operation: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            FORCE_MASKED.set(self.0);
        }
    }
    let _reset = Reset(FORCE_MASKED.replace(true));
    operation()
}

pub(super) fn masked_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    query_offset: usize,
    key_offset: usize,
    context: Option<usize>,
) -> Result<Tensor> {
    let (_, _, qlen, dim) = q.dims4()?;
    let klen = k.dim(2)?;
    let scores =
        (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? / (dim as f64).sqrt())?;
    let mask: Vec<u8> = (0..qlen)
        .flat_map(|i| {
            (0..klen).map(move |j| {
                let qp = query_offset + i;
                let kp = key_offset + j;
                u8::from(kp > qp || context.is_some_and(|c| qp.saturating_sub(kp) >= c))
            })
        })
        .collect();
    let mask =
        Tensor::from_vec(mask, (1, 1, qlen, klen), q.device())?.broadcast_as(scores.shape())?;
    let negative = Tensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?
        .to_dtype(scores.dtype())?;
    let scores = mask.where_cond(&negative, &scores)?;
    candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v.contiguous()?)
}

#[test]
#[ignore = "explicit CUDA numeric oracle; no model downloads"]
fn real_cuda_unmasked_decode_matches_original_mask_bits() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let mut checked = 0;
    for klen in [1, 2, 17, 226, 269] {
        for qlen in [1, 3] {
            for key_offset in [0, 7] {
                let q = Tensor::from_vec(
                    (0..4 * qlen * 64)
                        .map(|i| ((i * 17 % 101) as f32 - 50.) / 50.)
                        .collect(),
                    (1, 4, qlen, 64),
                    &device,
                )?;
                let k = Tensor::from_vec(
                    (0..4 * klen * 64)
                        .map(|i| ((i * 13 % 97) as f32 - 48.) / 48.)
                        .collect(),
                    (1, 4, klen, 64),
                    &device,
                )?;
                let v = Tensor::from_vec(
                    (0..4 * klen * 64)
                        .map(|i| ((i * 23 % 89) as f32 - 44.) / 44.)
                        .collect(),
                    (1, 4, klen, 64),
                    &device,
                )?;
                for dtype in [DType::F32, DType::F16, DType::BF16] {
                    let q = q.to_dtype(dtype)?;
                    let k = k.to_dtype(dtype)?;
                    let v = v.to_dtype(dtype)?;
                    for query_offset in [key_offset, key_offset + klen - 1, key_offset + klen] {
                        for context in [None, Some(1), Some(8)] {
                            let a = attend(&q, &k, &v, query_offset, key_offset, context)?
                                .flatten_all()?
                                .to_dtype(DType::F32)?
                                .to_vec1::<f32>()?;
                            let b =
                                masked_reference(&q, &k, &v, query_offset, key_offset, context)?
                                    .flatten_all()?
                                    .to_dtype(DType::F32)?
                                    .to_vec1::<f32>()?;
                            if context.is_none() {
                                assert!(a.iter().all(|x| x.is_finite()));
                            }
                            assert_eq!(
                                a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                "k={klen} q={qlen} key={key_offset} query={query_offset} context={context:?}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("{checked} CUDA attention cases match original masked implementation bitwise");
    Ok(())
}
