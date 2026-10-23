use super::*;

#[test]
fn cpu_canary_skips_unsupported_f16_gemm() {
    assert!(sync_probe_canary(&Device::Cpu).is_none());
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn canaries_belong_to_their_own_device_instances() -> Result<()> {
    let first = Device::new_metal(0)?;
    let second = Device::new_metal(0)?;
    for device in [&first, &second] {
        let (lhs, rhs) = sync_probe_canary(device).expect("Metal canary");
        assert!(lhs.device().same_device(device));
        assert!(rhs.device().same_device(device));
        lhs.matmul(&rhs)?;
        device.synchronize()?;
    }
    Ok(())
}
