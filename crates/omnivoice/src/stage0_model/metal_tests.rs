use super::*;

#[test]
fn class_top_k_above_metal_threadgroup_limit_matches_cpu() -> Result<()> {
    let device = match Device::new_metal(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("Skipping Metal parity test: {error}");
            return Ok(());
        }
    };
    for width in [1024, 1025, 4097] {
        let values = (0..2 * width)
            .map(|i| (i % width) as f32)
            .collect::<Vec<_>>();
        let cpu = Tensor::from_vec(values, (2, width), &Device::Cpu)?;
        let actual = filter_top_k(&cpu.to_device(&device)?, 0.1)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = filter_top_k(&cpu, 0.1)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(actual, expected, "vocabulary width {width}");
    }
    Ok(())
}

#[test]
fn unmask_selection_above_metal_threadgroup_limit_matches_cpu() -> Result<()> {
    let device = match Device::new_metal(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("Skipping Metal parity test: {error}");
            return Ok(());
        }
    };
    // 130 frames * 8 codebooks requires a 2048-thread bitonic sort.
    let dims = (1, 8, 130);
    for update_count in [7, 1040] {
        let run = |device: &Device| -> Result<Vec<i64>> {
            let current = Tensor::full(1024i64, dims, device)?;
            let predictions =
                Tensor::from_vec((0..1040).map(|i| (i % 1024) as i64).collect(), dims, device)?;
            let scores = Tensor::from_vec((0..1040).map(|i| i as f32).collect(), dims, device)?;
            let penalties = Tensor::zeros((1, 8, 1), DType::F32, device)?;
            let updated = apply_step_updates_device(
                &current,
                &predictions,
                &scores,
                1024,
                update_count,
                &penalties,
                0.0,
                None,
                None,
            )?;
            Ok(updated
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<i64>()?)
        };
        assert_eq!(
            run(&device)?,
            run(&Device::Cpu)?,
            "update count {update_count}"
        );
    }
    Ok(())
}
