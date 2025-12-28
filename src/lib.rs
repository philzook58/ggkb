// String Rewriting
// https://www.philipzucker.com/string_knuth/
use std::{
    collections::{HashMap, VecDeque},
    path::Iter,
};

pub mod multiset;
pub mod poly;
pub mod term;
pub mod unionfind;

pub type Id = usize; // Probably should consider going smaller.
pub type Word = Vec<Id>;

// substring
fn all_substring(s: &[Id], pat: &[Id]) -> Vec<usize> {
    s.windows(pat.len())
        .enumerate()
        .filter_map(|(i, window)| if window == pat { Some(i) } else { None })
        .collect()
}
fn substring(s: &[Id], pat: &[Id]) -> Option<usize> {
    if pat.len() > s.len() {
        None
    } else {
        s.windows(pat.len()).position(|window| window == pat)
    }
}

/*
if pat.len() == 0 || s.len() < pat.len() {
    return None;
}
for i in 0..=(s.len() - pat.len()) {
    if &s[i..i + pat.len()] == pat {
        return Some(i);
    }
}
return None;
*/

fn replace(t: &[Id], pos: usize, len: usize, rhs: &[Id]) -> Vec<Id> {
    let mut res = Vec::with_capacity(t.len() - len + rhs.len());
    res.extend_from_slice(&t[0..pos]);
    res.extend_from_slice(rhs);
    res.extend_from_slice(&t[pos + len..]);
    res
}

fn rewrite(s: &[Id], rules: &[(Vec<Id>, Vec<Id>)]) -> Option<Vec<Id>> {
    let mut s = s.to_vec();
    for count in 0.. {
        let mut done = true;
        for (lhs, rhs) in rules {
            if let Some(pos) = substring(&s, lhs) {
                s = replace(&s, pos, lhs.len(), rhs);
                done = false;
            }
        }
        if done {
            if count == 0 {
                return None;
            } else {
                return Some(s);
            }
        }
    }
    panic!("unreachable");
}

fn overlaps(lhs: &[Id], rhs: &[Id], lhs1: &[Id], rhs1: &[Id]) -> Vec<(Vec<Id>, Vec<Id>)> {
    // Could I do a swapsy wapsy?
    if lhs.len() < lhs1.len() {
        return overlaps(lhs1, rhs1, lhs, rhs);
    }
    // lhs is now smaller
    let mut res = Vec::new();
    // Find  all submatches of lhs1 in lhs
    for pos in all_substring(lhs, lhs1) {
        let mut t1 = Vec::with_capacity(rhs1.len() + (lhs.len() - lhs1.len()));
        t1.extend_from_slice(&lhs[0..pos]);
        t1.extend_from_slice(rhs1);
        t1.extend_from_slice(&lhs[pos + lhs1.len()..]);
        res.push((rhs.to_vec(), t1));
    }
    // Possible overlaps sizes extend from 1 to len(lhs1)-1
    for osize in 1..lhs1.len() {
        // overlaps begging of lhs1 with end of lhs
        if &lhs[lhs.len() - osize..] == &lhs1[..osize] {
            let mut t1 = Vec::with_capacity(rhs1.len() + (lhs.len() - osize));
            t1.extend_from_slice(&lhs[..lhs.len() - osize]);
            t1.extend_from_slice(rhs1);
            let mut t2 = Vec::with_capacity(rhs.len() + (lhs1.len() - osize));
            t2.extend_from_slice(&rhs);
            t2.extend_from_slice(&lhs1[osize..]);
            res.push((t1, t2));
        }
        // overlaps end of lhs1 with beggining of lhs
        if &lhs1[lhs1.len() - osize..] == &lhs[..osize] {
            let mut t1 = Vec::new();
            t1.extend_from_slice(&lhs1[..lhs1.len() - osize]);
            t1.extend_from_slice(rhs);
            let mut t2 = Vec::new();
            t2.extend_from_slice(&rhs1);
            t2.extend_from_slice(&lhs[osize..]);
            res.push((t1, t2));
        }
    }
    res
}

fn orient_shortlex(lhs: Vec<Id>, rhs: Vec<Id>) -> (Vec<Id>, Vec<Id>) {
    if lhs.len() < rhs.len() || (lhs.len() == rhs.len() && lhs < rhs) {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

pub fn cleanup(eqs: &[(Vec<Id>, Vec<Id>)]) -> Vec<(Vec<Id>, Vec<Id>)> {
    // sort possibly
    let mut rules = Vec::new();
    for (lhs, rhs) in eqs {
        let lhs = rewrite(lhs, &rules).unwrap_or(lhs.clone());
        let rhs = rewrite(rhs, &rules).unwrap_or(rhs.clone());
        if lhs != rhs {
            let (lhs, rhs) = orient_shortlex(lhs, rhs);
            rules.push((lhs, rhs));
        }
    }
    rules
}
pub fn saturate(eqs: Vec<(Vec<Id>, Vec<Id>)>) -> Vec<(Vec<Id>, Vec<Id>)> {
    let mut eqs = VecDeque::from(eqs);
    let mut rules = Vec::new();
    while let Some((lhs, rhs)) = eqs.pop_front() {
        let lhs = rewrite(&lhs, &rules).unwrap_or(lhs);
        let rhs = rewrite(&rhs, &rules).unwrap_or(rhs);
        if lhs != rhs {
            let (lhs, rhs) = orient_shortlex(lhs, rhs);
            for (lhs1, rhs1) in &rules {
                eqs.extend(overlaps(&lhs, &rhs, &lhs1, &rhs1));
            }

            rules.push((lhs, rhs));
        }
    }
    /*
    rules.sort_unstable_by(|a ,b |
    {
        if a.0.len() == b.0.len() {
            a.0.len().cmp(&b.0.len())
        } else {
            a.0.cmp(&b.0)
        }
    } )
    */
    rules
}

// quickcheck test confluence of resulting system?

fn parse_term(s: &str, names: &HashMap<String, Id>) -> Result<Vec<Id>, String> {
    s.split_whitespace()
        .map(|name| {
            names
                .get(name)
                .cloned()
                .ok_or(format!("Error: Name \"{}\" not declared", name.to_string()))
        })
        .collect()
}

pub fn simple_parse(s: &str) -> Result<(HashMap<String, Id>, Vec<(Word, Word)>), String> {
    // first line is generators in order
    // every line after is a lhs then rhs of a rule
    let mut names = HashMap::new();
    let mut iter = s.lines();
    let line0 = iter.next().ok_or("Error: Empty input")?;
    for (i, name) in line0.split_whitespace().enumerate() {
        names.insert(name.to_string(), i);
    }
    let mut eqs = Vec::new();
    for line in iter {
        let (lhs, rhs) = line
            .split_once("=")
            .ok_or("Error: line missing '=' separator")?;
        let lhs = parse_term(lhs.trim(), &names)?;
        let rhs = parse_term(rhs.trim(), &names)?;
        eqs.push((lhs, rhs));
    }
    Ok((names, eqs))
}

#[cfg(test)]
mod string_rewriting_tests {
    use super::*;
    #[test]
    fn test_substring() {
        let s = vec![1, 2, 3, 4, 2, 3];
        let pat = vec![2, 3];
        assert_eq!(all_substring(&s, &pat), vec![1, 4]);
        let pat2 = vec![3, 4, 5];
        assert_eq!(all_substring(&s, &pat2), vec![]);
    }
    #[test]
    fn test_rewrite() {
        let s = vec![1, 2, 3, 4, 2, 3];
        let rules = vec![(vec![2, 3], vec![5, 6]), (vec![3, 4], vec![7, 8])];
        let result = rewrite(&s, &rules).unwrap();
        assert_eq!(result, vec![1, 5, 6, 4, 5, 6]);
    }
    #[test]
    fn test_overlaps() {
        let res = overlaps(&vec![1, 2], &vec![2, 1], &vec![2, 3], &vec![3, 2]);
        assert_eq!(res, vec![(vec![1, 3, 2], vec![2, 1, 3])]);
    }
    #[test]
    fn test_saturate() {
        let eqs = vec![(vec![1, 2], vec![2, 1]), (vec![1, 3], vec![3, 1])];
        let rules = saturate(eqs);
        assert_eq!(
            rules,
            vec![(vec![2, 1], vec![1, 2]), (vec![3, 1], vec![1, 3])]
        );

        let eqs = vec![(vec![1, 1], vec![2]), (vec![1, 1, 1], vec![3])];
        let rules = saturate(eqs);
        assert_eq!(
            rules,
            vec![
                (vec![1, 1], vec![2]),
                (vec![2, 1], vec![3]),
                (vec![3, 1], vec![2, 2]),
                (vec![3, 2], vec![2, 3]),
                (vec![2, 2, 2], vec![3, 3])
            ]
        );
    }
    #[test]
    fn test_simple_parse() {
        let s = "a b c\na b = b a\nb c = c b";
        let (names, eqs) = simple_parse(s).unwrap();
        assert_eq!(names.get("a"), Some(&0));
        assert_eq!(names.get("b"), Some(&1));
        assert_eq!(names.get("c"), Some(&2));
        assert_eq!(
            eqs,
            vec![(vec![0, 1], vec![1, 0]), (vec![1, 2], vec![2, 1])]
        );
    }
}
/*
fn replace(s: &[Id], lhs: &[Id], rhs: &[Id]) -> Option<Vec<Id>> {
    if let Some(idx) = substring(s, lhs) {
        let mut result = Vec::with_capacity((s.len() - lhs.len()) + rhs.len());
        result.extend_from_slice(&s[0..idx]);
        result.extend_from_slice(rhs);
        result.extend_from_slice(&s[idx + lhs.len()..]);
        return Some(result);
    } else {
        return None;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Trie<V> {
    trie: HashMap<Id, Trie<V>>,
    value: Option<V>,
}
impl<V: Clone> Trie<V> {
    fn new() -> Self {
        Trie {
            trie: HashMap::new(),
            value: None,
        }
    }
    fn insert(&mut self, key: &[Id], value: V) {
        let mut node = self;
        for &k in key {
            node = node.trie.entry(k).or_insert_with(Trie::new);
        }
        node.value = Some(value);
    }
    fn remove(&mut self, key: &[Id]) {
        let mut node = self;
        for &k in key {
            if let Some(next_node) = node.trie.get_mut(&k) {
                node = next_node;
            } else {
                return;
            }
        }
        node.value = None;
    }
    fn longest_match(&self, key: &[Id]) -> Option<(usize, V)> {
        let mut size = 0;
        let mut node = self;
        for k in key {
            if let Some(next_node) = node.trie.get(k) {
                node = next_node;
                size += 1;
            } else {
                break;
            }
        }
        let value = node.value.clone()?;
        Some((size, value))
    }
}

type RW = Trie<Vec<Id>>;
*/
/*
fn string_rewrite(s: &mut Vec<Id>, rw: &RW) {
    for i in 0..s.len() {
        let (size, replacement_option) = rw.longest_match(&s[i..]);
        if let Some(replacement) = replacement_option {
            s.splice(i..i + size, replacement.iter().cloned());
        }
    }
}

    fn scan_replace(s: &[Id], rw: &RW) -> Option<Vec<Id>> {
    let mut res = Vec::new();
    let mut i = 0;
    let mut success = false;
    while i < s.len() {
        if let Some((size, replacement)) = rw.longest_match(&s[i..]) {
            res.extend_from_slice(&replacement);
            i += size;
            success = true;
        } else {
            res.push(s[i]);
            i += 1;
        }
    }
    if success { Some(res) } else { None }
}
    */

/*

fn rewrite(s: &[Id], rw: &RW) -> Option<Vec<Id>> {
    if let Some(mut next) = scan_replace(s, rw) {
        while let Some(next1) = scan_replace(&next, rw) {
            next = next1
        }
        return Some(next);
    } else {
        return None;
    }
}

enum Overlap {
    LSub(usize),
    RSub(usize),
    Over,
}

fn simp_eq(t1: &mut Vec<Id>, t2: &mut Vec<Id>, rw: &RW) -> bool {
    loop {
        if t1 == t2 {
            return true;
        } else if t1.len() >= t2.len() {
            if let Some(x) = scan_replace(t1, &rw) {
                continue;
            } else {
                if let Some(x) = scan_replace(t2, &rw) {
                    continue;
                } else {
                    return false;
                }
            }
        }
    }
}
*/
/*
def overlaps(s,t):
    """critical pairs https://en.wikipedia.org/wiki/Critical_pair_(term_rewriting)"""
    # make len(t) >= len(s)
    if len(t) < len(s):
        s,t = t,s
    if subseq(s,t) is not None:
        yield t
    # iterate over possible overlap sizes 1 to the len(s) at edges
    for osize in range(1,len(s)):
        if t[-osize:] == s[:osize]:
            yield t + s[osize:]
        if s[-osize:] == t[:osize]:
            yield s + t[osize:]
*/
/*
fn overlaps(lhs1: &[Id], rhs1: &[Id], lhs2: &[Id], rhs2: &[Id]) -> Vec<(Vec<Id>, Vec<Id>)> {
    if lhs1.len() < lhs2.len() {
        return overlaps(lhs2, rhs2, lhs1, rhs1);
    }
    let mut res = Vec::new();
    if let Some(_) = substring(lhs2, lhs1) {
        let mut new_rhs = Vec::new();
        new_rhs.extend_from_slice(rhs1);
        new_rhs.extend_from_slice(&lhs2[lhs1.len()..]);
        res.push((new_rhs, rhs2.to_vec()))
    }
    for osize in 1..lhs2.len() {
        if &lhs1[lhs1.len() - osize..] == &lhs2[..osize] {
            let mut new_lhs = Vec::new();
            new_lhs.extend_from_slice(lhs1);
            new_lhs.extend_from_slice(&lhs2[osize..]);
            let mut new_rhs = Vec::new();
            new_rhs.extend_from_slice(rhs1);
            new_rhs.extend_from_slice(&lhs2[osize..]);
            res.push((new_lhs, new_rhs));
        }
        if &lhs2[lhs2.len() - osize..] == &lhs1[..osize] {
            let mut new_lhs = Vec::new();
            new_lhs.extend_from_slice(lhs2);
            new_lhs.extend_from_slice(&lhs1[osize..]);
            let mut new_rhs = Vec::new();
            new_rhs.extend_from_slice(rhs2);
            new_rhs.extend_from_slice(&lhs1[osize..]);
            res.push((new_lhs, new_rhs));
        }
    }
    res
}

fn complete(rw: &mut RW, eqs: &Vec<(Vec<Id>, Vec<Id>)>) {
    let mut rw = Trie::new();
    while let Some((lhs, rhs)) = eqs.pop() {
            if !simp_eq(lhs, rhs, &rw) {
                if rhs.len() < lhs.len() || lhs.len() == rhs.len() && rhs < lhs {
                    rw.insert(&lhs, rhs);
                    for lhs1,rhs1 in rw {
                        eqs.extend(overlaps(&lhs, &rhs, &lhs1, &rhs1));
                    }
                } else if lhs.len() < rhs.len() {
                    rw.insert(&rhs, lhs);
                    for lhs1,rhs1 in rw {
                        eqs.extend(overlaps(&lhs, &rhs, &lhs, &rhs));
                    }
                }
            }
        }
    }

*/
/*
fn critical_pairs(t: Vec<Id>, rhs : Vec<Id>, rw: &RW) -> Vec<(Vec<Id>, Vec<Id>)>  {
    let mut res = Vec::new();
    for i in 0..t.len() {
        let mut node = rw;
        for j in i..t.len() {
            node = node.trie.get(&t[j]);
            if let Some(v) = &node.value {
                // construct the two critical pairs
                // i->j slice matches
                let mut p1 = Vec::new();
                p1.extend_from_slice(&t[0..i]);
                p1.extend_from_slice(v);
                p1.extend_from_slice(&t[j + 1..]);
                let p1 = rewrite(p1, rw).unwrap_or(p1);
                if p1 != rhs {
                    res.push((rhs, p1));
                    rhs = p1;

                }
            }
        }
        // everything that remains is a tail overlap
        let mut todo = vec![(vec![], node)];
        while let Some((pos, n)) = todo.pop() {
            if let Some(v) = &n.value {
                let mut p2 = Vec::new();
                p2.extend_from_slice(&t[0..i]);
                p2.extend_from_slice(v);
    }
    res
}
*/

/*
#[test]
fn test_replace() {
    let s = vec![1, 2, 3, 4, 2, 3];
    let lhs = vec![2, 3];
    let rhs = vec![5, 6];
    assert_eq!(replace(&s, &lhs, &rhs), Some(vec![1, 5, 6, 4, 2, 3]));
    let lhs2 = vec![3, 4, 5];
    assert_eq!(replace(&s, &lhs2, &rhs), None);
}
#[test]
fn test_trie() {
    let mut trie = Trie::new();
    trie.insert(&[1, 2, 3], "a");
    trie.insert(&[1, 2], "b");
    let (size, value) = trie.longest_match(&[1, 2, 3, 4]).unwrap();
    assert_eq!(size, 3);
    assert_eq!(value, "a");
    let (size2, value2) = trie.longest_match(&[1, 2, 4]).unwrap();
    assert_eq!(size2, 2);
    assert_eq!(value2, "b");
}
#[test]
fn test_scan_replace() {
    let mut rw = RW::new();
    rw.insert(&[1, 2], vec![5, 6]);
    rw.insert(&[3, 4], vec![7, 8]);
    let s = vec![0, 1, 2, 3, 4, 9];
    let result = scan_replace(&s, &rw).unwrap();
    assert_eq!(result, vec![0, 5, 6, 7, 8, 9]);

    let mut rw2 = RW::new();
    rw2.insert(&[1, 1], vec![2]);
    rw2.insert(&[2, 2], vec![1]);
    let result = rewrite(&vec![1, 1, 1, 1, 2], &rw2).unwrap();
    assert_eq!(result, vec![1, 2]);
}
*/
