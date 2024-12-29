use core::fmt::Debug;

use std::sync::{Arc, RwLock, RwLockWriteGuard};

use hashbrown::HashMap;
use sp1_curves::{edwards::ed25519::ed25519_sqrt, params::FieldParameters, BigUint, Integer, One};

use crate::Executor;

/// A runtime hook, wrapped in a smart pointer.
pub type BoxedHook<'a> = Arc<RwLock<dyn Hook + Send + Sync + 'a>>;

pub use sp1_primitives::consts::fd::*;

/// A runtime hook. May be called during execution by writing to a specified file descriptor,
/// accepting and returning arbitrary data.
pub trait Hook {
    /// Invoke the runtime hook with a standard environment and arbitrary data.
    /// Returns the computed data.
    fn invoke_hook(&mut self, env: HookEnv, buf: &[u8]) -> Vec<Vec<u8>>;
}

impl<F: FnMut(HookEnv, &[u8]) -> Vec<Vec<u8>>> Hook for F {
    /// Invokes the function `self` as a hook.
    fn invoke_hook(&mut self, env: HookEnv, buf: &[u8]) -> Vec<Vec<u8>> {
        self(env, buf)
    }
}

/// Wrap a function in a smart pointer so it may be placed in a `HookRegistry`.
///
/// Note: the Send + Sync requirement may be logically extraneous. Requires further investigation.
pub fn hookify<'a>(
    f: impl FnMut(HookEnv, &[u8]) -> Vec<Vec<u8>> + Send + Sync + 'a,
) -> BoxedHook<'a> {
    Arc::new(RwLock::new(f))
}

/// A registry of hooks to call, indexed by the file descriptors through which they are accessed.
#[derive(Clone)]
pub struct HookRegistry<'a> {
    /// Table of registered hooks. Prefer using `Runtime::hook`, ` Runtime::hook_env`,
    /// and `HookRegistry::get` over interacting with this field directly.
    pub(crate) table: HashMap<u32, BoxedHook<'a>>,
}

impl<'a> HookRegistry<'a> {
    /// Create a default [`HookRegistry`].
    #[must_use]
    pub fn new() -> Self {
        HookRegistry::default()
    }

    /// Create an empty [`HookRegistry`].
    #[must_use]
    pub fn empty() -> Self {
        Self { table: HashMap::default() }
    }

    /// Get a hook with exclusive write access, if it exists.
    ///
    /// Note: This function should not be called in async contexts, unless you know what you are
    /// doing.
    #[must_use]
    pub fn get(&self, fd: u32) -> Option<RwLockWriteGuard<dyn Hook + Send + Sync + 'a>> {
        // Calling `.unwrap()` panics on a poisoned lock. Should never happen normally.
        self.table.get(&fd).map(|x| x.write().unwrap())
    }
}

impl<'a> Default for HookRegistry<'a> {
    fn default() -> Self {
        // When `LazyCell` gets stabilized (1.81.0), we can use it to avoid unnecessary allocations.
        let table = HashMap::from([
            // Note: To ensure any `fd` value is synced with `zkvm/precompiles/src/io.rs`,
            // add an assertion to the test `hook_fds_match` below.
            (FD_ECRECOVER_HOOK, hookify(hook_ecrecover)),
            (FD_EDDECOMPRESS, hookify(hook_ed_decompress)),
            (FD_RSA_MUL_MOD, hookify(hook_rsa_mul_mod)),
        ]);

        Self { table }
    }
}

impl<'a> Debug for HookRegistry<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut keys = self.table.keys().collect::<Vec<_>>();
        keys.sort_unstable();
        f.debug_struct("HookRegistry")
            .field(
                "table",
                &format_args!("{{{} hooks registered at {:?}}}", self.table.len(), keys),
            )
            .finish()
    }
}

/// Environment that a hook may read from.
pub struct HookEnv<'a, 'b: 'a> {
    /// The runtime.
    pub runtime: &'a Executor<'b>,
}

/// The hook for the `ecrecover` patches.
///
/// The input should be of the form [(`curve_id_u8` | `r_is_y_odd_u8` << 7) || `r` || `alpha`] where:
/// * `curve_id` is 1 for secp256k1 and 2 for secp256r1
/// * `r_is_y_odd` is 0 if r is even and 1 if r is is odd
/// * r is the x-coordinate of the point, which should be 32 bytes,
/// * alpha := r * r * r * (a * r) + b, which should be 32 bytes.
///
/// Returns vec![vec![1], `y`, `r_inv`] if the point is decompressable
/// and vec![vec![0],`nqr_hint`] if not.
#[must_use]
pub fn hook_ecrecover(_: HookEnv, buf: &[u8]) -> Vec<Vec<u8>> {
    assert!(buf.len() == 64 + 1, "ecrecover should have length 65");

    let curve_id = buf[0] & 0b0111_1111;
    let r_is_y_odd = buf[0] & 0b1000_0000 != 0;

    let r_bytes: [u8; 32] = buf[1..33].try_into().unwrap();
    let alpha_bytes: [u8; 32] = buf[33..65].try_into().unwrap();

    match curve_id {
        1 => ecrecover::handle_secp256k1(r_bytes, alpha_bytes, r_is_y_odd),
        2 => ecrecover::handle_secp256r1(r_bytes, alpha_bytes, r_is_y_odd),
        _ => unimplemented!("Unsupported curve id: {}", curve_id),
    }
}

mod ecrecover {
    use sp1_curves::{k256, p256};

    /// The non-quadratic residue for the curve for secp256k1 and secp256r1.
    const NQR: [u8; 32] = {
        let mut nqr = [0; 32];
        nqr[31] = 3;
        nqr
    };

    pub(super) fn handle_secp256k1(r: [u8; 32], alpha: [u8; 32], r_y_is_odd: bool) -> Vec<Vec<u8>> {
        use k256::elliptic_curve::ff::PrimeField;
        use k256::FieldBytes as K256FieldBytes;
        use k256::FieldElement as K256FieldElement;
        use k256::Scalar as K256Scalar;

        let r = K256FieldElement::from_bytes(K256FieldBytes::from_slice(&r)).unwrap();
        assert!(!bool::from(r.is_zero()), "r should not be zero");

        let alpha = K256FieldElement::from_bytes(K256FieldBytes::from_slice(&alpha)).unwrap();
        assert!(!bool::from(alpha.is_zero()), "alpha should not be zero");

        // nomralize the y-coordinate always to be consistent.
        if let Some(mut y_coord) = alpha.sqrt().into_option().map(|y| y.normalize()) {
            let r = K256Scalar::from_repr(r.to_bytes()).unwrap();
            let r_inv = r.invert().expect("Non zero r scalar");

            if r_y_is_odd != bool::from(y_coord.is_odd()) {
                y_coord = y_coord.negate(1);
                y_coord = y_coord.normalize();
            }

            vec![vec![1], y_coord.to_bytes().to_vec(), r_inv.to_bytes().to_vec()]
        } else {
            let nqr_field = K256FieldElement::from_bytes(K256FieldBytes::from_slice(&NQR)).unwrap();
            let qr = alpha * nqr_field;
            let root = qr.sqrt().expect("if alpha is not a square, then qr should be a square");

            vec![vec![0], root.to_bytes().to_vec()]
        }
    }

    pub(super) fn handle_secp256r1(r: [u8; 32], alpha: [u8; 32], r_y_is_odd: bool) -> Vec<Vec<u8>> {
        use p256::elliptic_curve::ff::PrimeField;
        use p256::FieldBytes as P256FieldBytes;
        use p256::FieldElement as P256FieldElement;
        use p256::Scalar as P256Scalar;

        let r = P256FieldElement::from_bytes(P256FieldBytes::from_slice(&r)).unwrap();
        assert!(!bool::from(r.is_zero()), "r should not be zero");
        let alpha = P256FieldElement::from_bytes(P256FieldBytes::from_slice(&alpha)).unwrap();
        assert!(!bool::from(alpha.is_zero()), "alpha should not be zero");

        if let Some(mut y_coord) = alpha.sqrt().into_option() {
            let r = P256Scalar::from_repr(r.to_bytes()).unwrap();
            let r_inv = r.invert().expect("Non zero r scalar");

            if r_y_is_odd != bool::from(y_coord.is_odd()) {
                y_coord = -y_coord;
            }

            vec![vec![1], y_coord.to_bytes().to_vec(), r_inv.to_bytes().to_vec()]
        } else {
            let nqr_field = P256FieldElement::from_bytes(P256FieldBytes::from_slice(&NQR)).unwrap();
            let qr = alpha * nqr_field;
            let root = qr.sqrt().expect("if alpha is not a square, then qr should be a square");

            vec![vec![0], root.to_bytes().to_vec()]
        }
    }
}

/// Checks if a compressed Edwards point can be decompressed.
///
/// # Arguments
/// * `env` - The environment in which the hook is invoked.
/// * `buf` - The buffer containing the compressed Edwards point.
///    - The compressed Edwards point is 32 bytes.
///    - The high bit of the last byte is the sign bit.
///
/// Returns vec![vec![1]] if the point is decompressable.
/// Returns vec![vec![0], `v_inv`, `nqr_hint`] if the point is not decompressable.
///
/// WARNING: This function merely hints at the validity of the compressed point. These values must
/// be constrained by the zkVM for correctness.
#[must_use]
pub fn hook_ed_decompress(_: HookEnv, buf: &[u8]) -> Vec<Vec<u8>> {
    const NQR_CURVE_25519: u8 = 2;
    let modulus = sp1_curves::edwards::ed25519::Ed25519BaseField::modulus();

    let mut bytes: [u8; 32] = buf[..32].try_into().unwrap();
    // Mask the sign bit.
    bytes[31] &= 0b0111_1111;

    // The AIR asserts canon inputs, so hint here if it cant be satisified.
    let y = BigUint::from_bytes_le(&bytes);
    if y >= modulus {
        return vec![vec![0]];
    }

    let v = BigUint::from_bytes_le(&buf[32..]);
    // This is computed as dy^2 - 1
    // so it should always be in the field.
    assert!(v < modulus, "V is not a valid field element");

    // For a point to be decompressable, (yy - 1) / (yy * d + 1) must be a quadratic residue.
    let v_inv = v.modpow(&(&modulus - BigUint::from(2u64)), &modulus);
    let u = (&y * &y + &modulus - BigUint::one()) % &modulus;
    let u_div_v = (&u * &v_inv) % &modulus;

    // Note: Our sqrt impl doesnt care about canon represenation,
    // however we have already checked that were less than the modulus.
    if ed25519_sqrt(&u_div_v).is_some() {
        vec![vec![1]]
    } else {
        let qr = (u_div_v * NQR_CURVE_25519) % &modulus;
        let root = ed25519_sqrt(&qr).unwrap();

        // Pad the results, since this may not be a full 32 bytes.
        let v_inv_bytes = v_inv.to_bytes_le();
        let mut v_inv_padded = [0_u8; 32];
        v_inv_padded[..v_inv_bytes.len()].copy_from_slice(&v_inv.to_bytes_le());

        let root_bytes = root.to_bytes_le();
        let mut root_padded = [0_u8; 32];
        root_padded[..root_bytes.len()].copy_from_slice(&root.to_bytes_le());

        vec![vec![0], v_inv_padded.to_vec(), root_padded.to_vec()]
    }
}

/// Given the product of some 256-byte numbers and a modulus, this function does a modular
/// reduction and hints back the values to the vm in order to constrain it.
///
/// # Arguments
///
/// * `env` - The environment in which the hook is invoked.
/// * `buf` - The buffer containing the le bytes of the 512 byte product and the 256 byte modulus.
///
/// Returns The le bytes of the product % modulus (512 bytes)
/// and the quotient floor(product/modulus) (256 bytes).
///
/// WANRING: This function is used to perform a modular reduction outside of the zkVM context.
/// These values must be constrained by the zkVM for correctness.
#[must_use]
pub fn hook_rsa_mul_mod(_: HookEnv, buf: &[u8]) -> Vec<Vec<u8>> {
    assert_eq!(
        buf.len(),
        256 + 256 + 256,
        "rsa_mul_mod input should have length 256 + 256 + 256, this is a bug."
    );

    let prod: &[u8; 512] = buf[..512].try_into().unwrap();
    let m: &[u8; 256] = buf[512..].try_into().unwrap();

    let prod = BigUint::from_bytes_le(prod);
    let m = BigUint::from_bytes_le(m);

    let (q, rem) = prod.div_rem(&m);

    let mut rem = rem.to_bytes_le();
    rem.resize(256, 0);

    let mut q = q.to_bytes_le();
    q.resize(256, 0);

    vec![rem, q]
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    pub fn registry_new_is_inhabited() {
        assert_ne!(HookRegistry::new().table.len(), 0);
        println!("{:?}", HookRegistry::new());
    }

    #[test]
    pub fn registry_empty_is_empty() {
        assert_eq!(HookRegistry::empty().table.len(), 0);
    }
}
