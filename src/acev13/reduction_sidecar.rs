//! Fail-closed loader for the detached +1 LMR shadow sidecar.

use sha2::{Digest, Sha256};
use std::path::Path;

use super::net::live_weights_sha256;

pub const INPUTS: usize = 37;
const MAGIC: &[u8; 8] = b"TISRDX1\0";
const BYTES: usize = 8 + 12 + 32 + 8 + INPUTS * 8 + 4 * 8 + 32;

#[derive(Debug, Clone)]
pub struct ReductionSidecar {
    weights: [f64; INPUTS],
    bias: f64,
    calibration_scale: f64,
    calibration_shift: f64,
    threshold: f64,
}

impl ReductionSidecar {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("sidecar read failed: {e}"))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != BYTES {
            return Err(format!("sidecar size mismatch: {} != {BYTES}", bytes.len()));
        }
        if &bytes[..8] != MAGIC {
            return Err("sidecar magic mismatch".into());
        }
        let read_u32 = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        if read_u32(8) != 1 || read_u32(12) != 1 || read_u32(16) as usize != INPUTS {
            return Err("sidecar schema mismatch".into());
        }
        if bytes[20..52] != live_weights_sha256() {
            return Err("sidecar trunk hash mismatch".into());
        }
        if read_u32(52) != 1 || read_u32(56) != 1 {
            return Err("sidecar calibration/data version mismatch".into());
        }
        let payload_end = BYTES - 32;
        let digest: [u8; 32] = Sha256::digest(&bytes[..payload_end]).into();
        if bytes[payload_end..] != digest {
            return Err("sidecar payload hash mismatch".into());
        }
        let mut offset = 60;
        let mut weights = [0.0; INPUTS];
        for weight in &mut weights {
            *weight = f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            offset += 8;
        }
        let mut next_f64 = || {
            let value = f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            offset += 8;
            value
        };
        let bias = next_f64();
        let calibration_scale = next_f64();
        let calibration_shift = next_f64();
        let threshold = next_f64();
        if !weights.iter().all(|v| v.is_finite())
            || ![bias, calibration_scale, calibration_shift, threshold]
                .iter()
                .all(|v| v.is_finite())
            || !(0.0..=1.0).contains(&threshold)
        {
            return Err("sidecar contains invalid numeric values".into());
        }
        Ok(Self {
            weights,
            bias,
            calibration_scale,
            calibration_shift,
            threshold,
        })
    }

    pub fn predict(&self, hidden: &[f64; 32], context: &[f64; 5]) -> f64 {
        let mut logit = self.bias;
        for (weight, value) in self.weights[..32].iter().zip(hidden) {
            logit += weight * value;
        }
        for (weight, value) in self.weights[32..].iter().zip(context) {
            logit += weight * value;
        }
        let calibrated = self.calibration_scale * logit + self.calibration_shift;
        if !calibrated.is_finite() {
            return 0.0;
        }
        1.0 / (1.0 + (-calibrated.clamp(-60.0, 60.0)).exp())
    }

    pub fn would_activate(&self, probability: f64) -> bool {
        probability.is_finite() && probability >= self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(INPUTS as u32).to_le_bytes());
        bytes.extend_from_slice(&live_weights_sha256());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        for _ in 0..INPUTS {
            bytes.extend_from_slice(&0.0f64.to_le_bytes());
        }
        bytes.extend_from_slice(&(-6.0f64).to_le_bytes());
        bytes.extend_from_slice(&1.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.99f64.to_le_bytes());
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        bytes.extend_from_slice(&digest);
        bytes
    }

    #[test]
    fn neutral_sidecar_favors_no_activation() {
        let sidecar = ReductionSidecar::from_bytes(&blob()).unwrap();
        let p = sidecar.predict(&[0.5; 32], &[0.5; 5]);
        assert!(p < 0.01);
        assert!(!sidecar.would_activate(p));
    }

    #[test]
    fn malformed_hash_and_trunk_fail_closed() {
        let mut bad_payload = blob();
        bad_payload[100] ^= 1;
        assert!(ReductionSidecar::from_bytes(&bad_payload).is_err());
        let mut bad_trunk = blob();
        bad_trunk[20] ^= 1;
        let payload_end = bad_trunk.len() - 32;
        let digest: [u8; 32] = Sha256::digest(&bad_trunk[..payload_end]).into();
        bad_trunk[payload_end..].copy_from_slice(&digest);
        assert!(ReductionSidecar::from_bytes(&bad_trunk).is_err());
    }

    #[test]
    fn nan_fails_closed() {
        let mut bytes = blob();
        bytes[60..68].copy_from_slice(&f64::NAN.to_le_bytes());
        let payload_end = bytes.len() - 32;
        let digest: [u8; 32] = Sha256::digest(&bytes[..payload_end]).into();
        bytes[payload_end..].copy_from_slice(&digest);
        assert!(ReductionSidecar::from_bytes(&bytes).is_err());
    }
}

#[cfg(test)]
pub(crate) fn neutral_test_blob() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(INPUTS as u32).to_le_bytes());
    bytes.extend_from_slice(&live_weights_sha256());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    for _ in 0..INPUTS {
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
    }
    bytes.extend_from_slice(&(-6.0f64).to_le_bytes());
    bytes.extend_from_slice(&1.0f64.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&0.99f64.to_le_bytes());
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&digest);
    bytes
}
