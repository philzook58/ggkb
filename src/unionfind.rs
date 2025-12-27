#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UF {
    parents: Vec<usize>,
}

impl UF {
    pub fn makeset(&mut self) -> usize {
        let n = self.parents.len();
        self.parents.push(n);
        return n;
    }
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parents[x] != x {
            x = self.parents[x];
        }
        return x;
    }
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let px = self.find(x);
        let py = self.find(y);
        if px != py {
            self.parents[px] = py;
            return false;
        } else {
            return true;
        }
    }
}

// Group UF, GF2
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_uf() {
        let mut uf = UF { parents: vec![] };
        let a = uf.makeset();
        let b = uf.makeset();
        let c = uf.makeset();
        assert_eq!(uf.find(a), a);
        assert_eq!(uf.find(b), b);
        assert_eq!(uf.find(c), c);
        assert_eq!(uf.union(a, b), false);
        assert_eq!(uf.union(a, b), true);
        assert_eq!(uf.find(a), uf.find(b));
        assert_ne!(uf.find(a), uf.find(c));
    }
}
