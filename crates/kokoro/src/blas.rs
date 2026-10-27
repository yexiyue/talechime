//! macOS CPU affine GEMM. PyTorch initializes the destination with bias and
//! calls BLAS with beta=1; adding bias after a beta=0 GEMM rounds differently.
use candle_core::{Result, Tensor};
use std::ffi::{c_char, c_int};

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn sgemm_(
        ta: *const c_char,
        tb: *const c_char,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const f32,
        a: *const f32,
        lda: *const c_int,
        b: *const f32,
        ldb: *const c_int,
        beta: *const f32,
        c: *mut f32,
        ldc: *const c_int,
    );
}

fn gemm(a: &[f32], b: &[f32], c: &mut [f32], dims: [usize; 3], transpose_a: bool) -> Result<()> {
    let [m, n, k] = dims;
    if m == 0 || n == 0 || k == 0 || a.len() != m * k || b.len() != k * n || c.len() != m * n {
        candle_core::bail!("invalid affine GEMM dimensions");
    }
    let m = c_int::try_from(m).map_err(candle_core::Error::wrap)?;
    let n = c_int::try_from(n).map_err(candle_core::Error::wrap)?;
    let k = c_int::try_from(k).map_err(candle_core::Error::wrap)?;
    let ta = if transpose_a { b'T' } else { b'N' } as c_char;
    let tb = b'N' as c_char;
    let lda = if transpose_a { k } else { m };
    // SAFETY: all column-major matrix extents and leading dimensions are
    // checked above; source and destination allocations do not overlap.
    unsafe {
        sgemm_(
            &ta,
            &tb,
            &m,
            &n,
            &k,
            &1.0,
            a.as_ptr(),
            &lda,
            b.as_ptr(),
            &k,
            &1.0,
            c.as_mut_ptr(),
            &m,
        );
    }
    Ok(())
}

pub(crate) fn linear(x: &Tensor, w: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let input = x.dim(x.rank() - 1)?;
    let (output, columns) = w.dims2()?;
    if input == 0 || columns != input || bias.dims() != [output] {
        candle_core::bail!("invalid affine linear shape");
    }
    let rows = x.elem_count() / input;
    let values = x.flatten_all()?.to_vec1::<f32>()?;
    let weights = w.flatten_all()?.to_vec1::<f32>()?;
    let bias = bias.to_vec1::<f32>()?;
    let mut result = bias.repeat(rows);
    gemm(&weights, &values, &mut result, [output, rows, input], true)?;
    let mut shape = x.dims().to_vec();
    *shape.last_mut().unwrap() = output;
    Tensor::from_vec(result, shape, x.device())
}

pub(crate) fn conv(
    x: &Tensor,
    w: &Tensor,
    bias: &Tensor,
    stride: usize,
    dilation: usize,
) -> Result<Tensor> {
    let (batch, input, steps) = x.dims3()?;
    let (output, columns, kernel) = w.dims3()?;
    if input == 0
        || steps == 0
        || kernel == 0
        || stride == 0
        || dilation == 0
        || input != columns
        || bias.dims() != [output]
    {
        candle_core::bail!("invalid convolution affine shape");
    }
    let values = x.flatten_all()?.to_vec1::<f32>()?;
    let weights = w.flatten_all()?.to_vec1::<f32>()?;
    let bias = bias.to_vec1::<f32>()?;
    let padding = (kernel - 1) * dilation / 2;
    let output_steps = (steps + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
    let mut result = Vec::with_capacity(batch * output * output_steps);
    for _ in 0..batch {
        for &value in &bias {
            result.extend(std::iter::repeat_n(value, output_steps));
        }
    }
    for b in 0..batch {
        let mut columns = vec![0.0; input * kernel * output_steps];
        for c in 0..input {
            for k in 0..kernel {
                for t in 0..output_steps {
                    let position = t * stride + k * dilation;
                    if position >= padding && position - padding < steps {
                        columns[(c * kernel + k) * output_steps + t] =
                            values[(b * input + c) * steps + position - padding];
                    }
                }
            }
        }
        gemm(
            &columns,
            &weights,
            &mut result[b * output * output_steps..(b + 1) * output * output_steps],
            [output_steps, output, input * kernel],
            false,
        )?;
    }
    Tensor::from_vec(result, (batch, output, output_steps), x.device())
}
