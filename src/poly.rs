use num_bigint::BigInt;
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Poly<T: Eq + Hash> {
    coeffs: HashMap<T, BigInt>,
}
