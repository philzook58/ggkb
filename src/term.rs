use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq)]
enum Term {
    Lit(usize),
    App(usize, usize),
}

struct RecExpr {
    litmemo: HashMap<usize, usize>,
    appmemo: HashMap<(usize, usize), usize>,
    terms: Vec<Term>,
    sizes: Vec<usize>,
}

impl RecExpr {
    pub fn new() -> RecExpr {
        RecExpr {
            litmemo: HashMap::new(),
            appmemo: HashMap::new(),
            terms: Vec::new(),
            sizes: Vec::new(),
        }
    }

    pub fn app(&mut self, f: usize, x: usize) -> usize {
        if let Some(&t) = self.appmemo.get(&(f, x)) {
            return t;
        }
        let t = Term::App(f, x);
        self.terms.push(t);
        self.sizes.push(1 + self.sizes[f] + self.sizes[x]);
        self.appmemo.insert((f, x), self.terms.len() - 1);
        return self.terms.len() - 1;
    }

    pub fn lit(&mut self, n: usize) -> usize {
        if let Some(&t) = self.litmemo.get(&n) {
            return t;
        }
        let t = Term::Lit(n);
        self.terms.push(t);
        self.sizes.push(1);
        self.litmemo.insert(n, self.terms.len() - 1);
        return self.terms.len() - 1;
    }

    fn kbo(&self, a: usize, b: usize) -> std::cmp::Ordering {
        if a == b {
            return std::cmp::Ordering::Equal;
        }
        let sa = self.sizes[a];
        let sb = self.sizes[b];
        if sa != sb {
            return sa.cmp(&sb);
        }
        match (&self.terms[a], &self.terms[b]) {
            (Term::Lit(x), Term::Lit(y)) => x.cmp(y),
            (Term::Lit(_), Term::App(_, _)) => panic!("Unreachable"), // impossible
            (Term::App(_, _), Term::Lit(_)) => panic!("unreachable"), // impossible
            (Term::App(f1, x1), Term::App(f2, x2)) => {
                let cf = self.kbo(*f1, *f2);
                if cf != std::cmp::Ordering::Equal {
                    return cf;
                }
                return self.kbo(*x1, *x2);
            }
        }
    }
    fn subterms(&self, node: usize) -> HashSet<usize> {
        let mut todo = vec![node];
        let mut res = HashSet::new();
        while let Some(n) = todo.pop() {
            match self.terms[n] {
                Term::Lit(_num) => {}
                Term::App(f, x) => {
                    if !res.contains(&f) {
                        todo.push(f);
                        res.insert(f);
                    }
                    if !res.contains(&x) {
                        todo.push(x);
                        res.insert(x);
                    }
                }
            }
        }
        res
    }

    fn substitute_one(&mut self, node: usize, lhs: usize, rhs: usize) -> usize {
        if node == lhs {
            return rhs;
        } else {
            match self.terms[node].clone() {
                Term::Lit(_n) => node,
                Term::App(f, x) => {
                    let f2 = self.substitute_one(f, lhs, rhs);
                    let x2 = self.substitute_one(x, lhs, rhs);
                    if f2 != f || x2 != x {
                        return self.app(f2, x2);
                    } else {
                        return node;
                    }
                }
            }
        }
    }

    fn substitute(&mut self, node: usize, rw: &HashMap<usize, usize>) -> usize {
        match rw.get(&node) {
            Some(&rhs) => rhs,
            None => match self.terms[node].clone() {
                Term::Lit(_n) => node,
                Term::App(f, x) => {
                    let f2 = self.substitute(f, rw);
                    let x2 = self.substitute(x, rw);
                    if f2 != f || x2 != x {
                        return self.app(f2, x2);
                    } else {
                        return node;
                    }
                }
            },
        }
    }

    /*
    fn saturate(&mut self, eqs: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut eqs = VecDeque::new();
        eqs.extend(eqs.iter().cloned());
        let mut rules = HashMap::new();
        while let Some((lhs, rhs)) = eqs.pop_front() {
            let lhs = self.substitute(lhs, &rules);
            let rhs = self.substitute(rhs, &rules);
            if lhs == rhs {
                continue;
            }
            else{
              match self.kbo(*lhs, *rhs); {
                std::cmp::Ordering::Less => rules.push((*lhs, *rhs)),
                std::cmp::Ordering::Greater => rules.push((*rhs, *lhs)),
                std::cmp::Ordering::Equal => panic!("Unreachable"),
            };
        }
        rules
    }
    */
}

mod tests {
    use super::*;
    #[test]
    fn test_recexpr() {
        let mut expr = RecExpr::new();
        let a = expr.lit(1);
        let a1 = expr.lit(1);
        assert_eq!(a, a1);
        let b = expr.lit(2);
        let f = expr.app(a, b);
        let g = expr.app(b, a);
        assert_eq!(expr.substitute_one(a, a, b), b);
        let f2 = expr.substitute_one(f, a, b);
        assert_eq!(f2, expr.app(b, b));

        assert_eq!(expr.kbo(a, b), std::cmp::Ordering::Less);
        assert_eq!(expr.kbo(b, a), std::cmp::Ordering::Greater);
        assert_eq!(expr.kbo(f, g), std::cmp::Ordering::Less);
    }
}
