//! ACE v10 HalfPW net — weights from `net_weights.bin` (H=32 NET_DATA blob).
//!
//! Philosophy: NN = geometric prior, search = tactical proof. See `field_planes.rs`.
//!
//! Two embedded blobs:
//!   `net_weights.bin`        — v15 live (micro-train + deploy updates this)
//!   `net_weights_frozen.bin` — pinned v13 baseline (ti-pure anchor + v15-frozen)
//!   `net_weights_medium.bin` — browser Medium tier, also used by native proxy
//!
//! Blob layout (little-endian f64):
//!   Wskip[20] B1[32] W2[32] W1C[36864] PO[2592] PX[2592]
//!   goal_inv_p0, goal_inv_p1, pawn_fwd_p0, pawn_fwd_p1,
//!   corridor_delta_p0, corridor_delta_p1, path_cross_p0, path_cross_p1,
//!   choke_p0, choke_p1, contested  (each 81×32 except contested is shared)
use sha2::{Digest, Sha256};
use std::sync::OnceLock;
pub const NET_H: usize = 32;
pub const WSKIP_LEN: usize = 20;
const W1C_LEN: usize = 9 * 128 * NET_H;
const PO_LEN: usize = 81 * NET_H;
const PX_LEN: usize = 81 * NET_H;
const FIELD_PLANE_LEN: usize = 81;
const FIELD_PLANE_SETS: usize = 5;
/// Legacy blob size: the 5 route field planes only.
pub const NET_WEIGHT_F64S: usize =
    WSKIP_LEN + NET_H + NET_H + W1C_LEN + PO_LEN + PX_LEN + FIELD_PLANE_LEN * FIELD_PLANE_SETS;
/// Retraining-ready blob size: adds ONE `cat_heat` plane (the combined CAT impact
/// heatmap as a direct net input, alongside the atomic route/near/contested planes
/// so the net needn't reconstruct CAT from its parts). Current weights files are the
/// legacy size; the loader zero-pads `cat_heat`, so the live net is unaffected until
/// a retrain ships a blob of this larger size. New planes (raw distance fields,
/// extra path-set bands) extend here the same way.
pub const NET_WEIGHT_F64S_CAT: usize = NET_WEIGHT_F64S + FIELD_PLANE_LEN;
static NET_BYTES: &[u8] = include_bytes!("net_weights.bin");
static NET_FROZEN_BYTES: &[u8] = include_bytes!("net_weights_frozen.bin");
static NET_MEDIUM_BYTES: &[u8] = include_bytes!("net_weights_medium.bin");
pub struct Net {
    pub ws: [f64; WSKIP_LEN],
    pub b1: [f64; NET_H],
    pub w2: [f64; NET_H],
    pub w1c: Vec<f64>,
    pub po: Vec<f64>,
    pub px: Vec<f64>,
    /// Sparse route embeddings, canonicalized to side-to-move coordinates.
    pub route_me: Vec<f64>,
    pub route_opp: Vec<f64>,
    pub route_near_me: Vec<f64>,
    pub route_near_opp: Vec<f64>,
    pub route_contested: Vec<f64>,
    pub route_active: bool,
    /// Combined CAT impact heatmap as a direct input plane (81, side-to-move
    /// canonical). Zero in legacy blobs (loader zero-pads) → `cat_active` false →
    /// not even computed, so the live net is unaffected. A retrained blob carries
    /// learned weights → `cat_active` true → contributes.
    pub cat_heat: Vec<f64>,
    pub cat_active: bool,
}
fn read_f64s(bytes: &[u8], offset: &mut usize, count: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let chunk: [u8; 8] = bytes[*offset..*offset + 8].try_into().unwrap();
        out.push(f64::from_le_bytes(chunk));
        *offset += 8;
    }
    out
}
fn load_net_from_bytes(bytes: &[u8]) -> Net {
    // Accept the legacy blob (5 route planes) OR the retraining-ready blob that
    // additionally carries the `cat_heat` plane. Legacy → cat_heat zero-padded.
    let has_cat = bytes.len() == NET_WEIGHT_F64S_CAT * 8;
    assert!(
        bytes.len() == NET_WEIGHT_F64S * 8 || has_cat,
        "net_weights blob size mismatch — run training/freeze_baseline_weights.py"
    );
    let mut offset = 0;
    let ws_v = read_f64s(bytes, &mut offset, WSKIP_LEN);
    let b1_v = read_f64s(bytes, &mut offset, NET_H);
    let w2_v = read_f64s(bytes, &mut offset, NET_H);
    let w1c = read_f64s(bytes, &mut offset, W1C_LEN);
    let po = read_f64s(bytes, &mut offset, PO_LEN);
    let px = read_f64s(bytes, &mut offset, PX_LEN);
    let route_me = read_f64s(bytes, &mut offset, FIELD_PLANE_LEN);
    let route_opp = read_f64s(bytes, &mut offset, FIELD_PLANE_LEN);
    let route_near_me = read_f64s(bytes, &mut offset, FIELD_PLANE_LEN);
    let route_near_opp = read_f64s(bytes, &mut offset, FIELD_PLANE_LEN);
    let route_contested = read_f64s(bytes, &mut offset, FIELD_PLANE_LEN);
    let route_active = route_me
        .iter()
        .chain(&route_opp)
        .chain(&route_near_me)
        .chain(&route_near_opp)
        .chain(&route_contested)
        .any(|&w| w != 0.0);
    let cat_heat = if has_cat {
        read_f64s(bytes, &mut offset, FIELD_PLANE_LEN)
    } else {
        vec![0.0; FIELD_PLANE_LEN]
    };
    let cat_active = cat_heat.iter().any(|&w| w != 0.0);
    Net {
        ws: ws_v.try_into().unwrap(),
        b1: b1_v.try_into().unwrap(),
        w2: w2_v.try_into().unwrap(),
        w1c,
        po,
        px,
        route_me,
        route_opp,
        route_near_me,
        route_near_opp,
        route_contested,
        route_active,
        cat_heat,
        cat_active,
    }
}
/// Training / deployed weights (`net_weights.bin`, overridable via `TITANIUM_NET_WEIGHTS_PATH`).
pub fn net() -> &'static Net {
    static NET: OnceLock<Net> = OnceLock::new();
    NET.get_or_init(|| {
        if let Ok(path) = std::env::var("TITANIUM_NET_WEIGHTS_PATH") {
            let bytes = std::fs::read(&path)
                .unwrap_or_else(|e| panic!("TITANIUM_NET_WEIGHTS_PATH read failed ({path}): {e}"));
            load_net_from_bytes(&bytes)
        } else {
            load_net_from_bytes(NET_BYTES)
        }
    })
}
/// Original v13 baseline — same search as v15, frozen HalfPW (`net_weights_frozen.bin`).
pub fn net_frozen() -> &'static Net {
    static NET: OnceLock<Net> = OnceLock::new();
    NET.get_or_init(|| load_net_from_bytes(NET_FROZEN_BYTES))
}

pub fn live_weights_sha256() -> [u8; 32] {
    Sha256::digest(NET_BYTES).into()
}

pub fn frozen_weights_sha256() -> [u8; 32] {
    Sha256::digest(NET_FROZEN_BYTES).into()
}

pub const NET_WEIGHT_BYTE_LEN: usize = NET_WEIGHT_F64S * 8;

static NET_MEDIUM: OnceLock<Net> = OnceLock::new();

/// Runtime medium-tier weights (fetched by the browser worker).
pub fn install_medium_weights(bytes: &[u8]) -> Result<(), &'static str> {
    if bytes.len() != NET_WEIGHT_BYTE_LEN && bytes.len() != NET_WEIGHT_F64S_CAT * 8 {
        return Err("medium weights size mismatch");
    }
    let net = load_net_from_bytes(bytes);
    NET_MEDIUM
        .set(net)
        .map_err(|_| "medium weights already installed")
}

pub fn net_medium() -> Option<&'static Net> {
    if let Some(net) = NET_MEDIUM.get() {
        return Some(net);
    }
    static NET_BUILTIN_MEDIUM: OnceLock<Net> = OnceLock::new();
    Some(NET_BUILTIN_MEDIUM.get_or_init(|| load_net_from_bytes(NET_MEDIUM_BYTES)))
}
// ── Symmetry tables (match the JS NET_MIRC / NET_MIRS / NET_BKT loops) ────────
const fn build_mirc() -> [usize; 81] {
    let mut arr = [0usize; 81];
    let mut i = 0;
    while i < 81 {
        arr[i] = (8 - i / 9) * 9 + i % 9;
        i += 1;
    }
    arr
}
const fn build_mirs() -> [usize; 64] {
    let mut arr = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        arr[i] = (7 - i / 8) * 8 + i % 8;
        i += 1;
    }
    arr
}
const fn build_bkt() -> [usize; 81] {
    let mut arr = [0usize; 81];
    let mut i = 0;
    while i < 81 {
        arr[i] = (i / 9 / 3) * 3 + (i % 9) / 3;
        i += 1;
    }
    arr
}
pub static NET_MIRC: [usize; 81] = build_mirc();
pub static NET_MIRS: [usize; 64] = build_mirs();
pub static NET_BKT: [usize; 81] = build_bkt();
