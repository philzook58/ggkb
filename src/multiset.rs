// instead of big, use checked versions that panic?
use num_bigint::BigInt;
use num_bigint::BigUint;

use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops;

#[derive(Debug, Clone, PartialEq, Eq)]
struct MultiSet<T>
where
    T: Eq + Hash,
{
    counts: BTreeMap<T, BigUint>,
    // size : BigUint,
    // maxlit: Option<(T, BigUint)>,
}
// Minheap off the maximal element?
impl<T: Eq + Hash> MultiSet<T> {
    fn len(&self) -> BigUint {
        self.counts.values().sum()
    }

    fn divrem_mut(&mut self, other: &MultiSet<T>) -> Option<BigUint>
    where
        T: Clone + Ord,
    {
        // mutate self into a remainder
        // get minimum
        let mut div = None;
        for (k, v) in other.counts.iter() {
            let self_v = self.counts.get(k)?;
            if self_v < v {
                return None;
            } else {
                if let Some(q) = div {
                    div = Some(min(q, self_v / v));
                } else {
                    div = Some(self_v / v);
                }
            }
        }
        let div = div?;
        for (k, v) in other.counts.iter() {
            self.counts.entry(k.clone()).and_modify(|v1| {
                *v1 -= v;
            });
        }
        return Some(div);
    }

    fn replace(&mut self, lhs: &MultiSet<T>, rhs: &MultiSet<T>) -> bool
    where
        T: Clone + Ord,
    {
        let div = self.divrem_mut(lhs);
        if let Some(div) = div {
            for (k, v) in rhs.counts.iter() {
                let entry = self.counts.entry(k.clone()).or_insert(BigUint::from(0u32));
                *entry += v * &div;
            }
            return true;
        } else {
            return false;
        }
    }
}

impl<T: Eq + Hash + Clone + Ord> ops::Add for MultiSet<T> {
    type Output = MultiSet<T>;
    fn add(self, other: MultiSet<T>) -> MultiSet<T> {
        let mut result = self.counts.clone();
        for (k, v) in other.counts.iter() {
            let entry = result.entry(k.clone()).or_insert(BigUint::from(0u32));
            *entry += v;
        }
        MultiSet { counts: result }
    }
}

type MSRW<T> = BTreeMap<T, Vec<(MultiSet<T>, MultiSet<T>)>>;

pub type Id = usize;
/*
fn msrewrite(mut ms: &MultiSet<Id>, msrw: &MSRW<Id>) -> bool {
    loop {
        let mut done = true;
        for k in ms.counts.keys() {
            if let Some(rws) = msrw.get(&k) {
                for (lhs, rhs) in rws {
                    if ms.replace(lhs, rhs) {
                        done = false;
                    }
                }
            }
        }
        if done {
            return true;
        }
    }
}
*/

/*
#[cfg(test)]
mod multiset_tests {
    use super::*;
    #[test]
    fn test_multiset_divrem() {
        let mut a_counts = HashMap::new();
        a_counts.insert('a', BigUint::from(4u32));
        a_counts.insert('b', BigUint::from(6u32));
        let a = MultiSet { counts: a_counts };
        let mut b_counts = HashMap::new();
        b_counts.insert('a', BigUint::from(2u32));
        b_counts.insert('b', BigUint::from(3u32));
        let b = MultiSet { counts: b_counts };
        let (q, r) = a.divrem(&b).unwrap();
        assert_eq!(q, BigUint::from(2u32));
        let mut expected_r_counts = HashMap::new();
        expected_r_counts.insert('a', BigUint::from(0u32));
        expected_r_counts.insert('b', BigUint::from(0u32));
        let expected_r = MultiSet {
            counts: expected_r_counts,
        };
        assert_eq!(r, expected_r);
    }
}
*/
