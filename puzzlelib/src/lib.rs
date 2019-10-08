extern crate rand;
extern crate rayon;

use rand::Rng;
use std::convert::TryFrom;
use std::fmt;
use std::time;
use std::sync::Arc;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::collections::HashMap;
use rayon::prelude::*;
use std::time::{SystemTime, Duration};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub struct Point {
    pub h: i8,
    pub v: i8,
}

#[pyclass(module = "puzzle_lib")]
#[derive(Hash, Eq, PartialEq, Clone)]
pub struct Set {
    #[pyo3(get)]
    pub g: u32,
    #[pyo3(get)]
    pub h: u32,
    #[pyo3(get)]
    pub f: u32,
    #[pyo3(get)]
    pub position: Vec<i8>,
}


fn calculate_hash<T: Hash> (t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

fn get_best_step(map: & HashMap<Vec<i8>, Step>, f: u32) -> Vec<i8> {
    let mut val = vec![];
    let best = map.par_iter().min_by(|(k1, s1),(k2,s2)| (s1.set.f.cmp(&s2.set.f)));
    let (key, value) = best.unwrap();
    val=key.to_vec();
    val
}

fn get_by_hash(map: & HashMap<Vec<i8>, Step>, hash: u64) -> Vec<i8> {
    let mut val = vec![];
    let result = map.par_iter().find_first(|(k1, s1)| (s1.hash_current == hash));
    let (key, value) = result.unwrap();
    val = key.to_vec();
    val
}

#[pyclass(module = "puzzle_lib")]
#[derive(Clone)]
pub struct Step {
    pub hash_prev: u64,
    #[pyo3(get)]
    pub set: Set,
    pub hash_current: u64,
}

#[pyclass(module = "puzzle_lib")]
pub struct Puzzle {
    #[pyo3(get)]
    pub size_h: i8,
    #[pyo3(get)]
    pub size_v: i8,
    #[pyo3(get)]
    pub puzzle: Vec<i8>,
    #[pyo3(get)]
    pub start: Vec<i8>,
    #[pyo3(get)]
    pub goal: Vec<i8>,
}


impl fmt::Debug for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}]", self.h, self.v)
    }
}

impl fmt::Debug for Set {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[g - {}, h - {}, f - {}, position - {:?}]", self.g, self.h, self.f,
                                                                        self.position)
    }
}

#[pymethods]
impl Puzzle {
    #[new]
    fn new(obj: &PyRawObject, size_h: i8, size_v: i8) {
        let puzzle: Vec<i8> = vec![];
        let mut start: Vec<i8> = vec![];
        let mut goal: Vec<i8> = vec![];
        if size_h > 1 && size_h < 10 && size_v > 1 && size_v < 10 {
            for i in 1..size_v*size_h {
                start.push(i);
                goal.push(i);
            }
            start.push(0);
            goal.push(0);
            obj.init(Puzzle {size_h: size_h, size_v: size_v, puzzle: puzzle, start: start, goal: goal });
        } else {
            for i in 1..4*4 {         //Default
                start.push(i);
                goal.push(i);
            }
            start.push(0);
            goal.push(0);
            obj.init(Puzzle {size_h: 4, size_v: 4, puzzle: puzzle, start: start, goal: goal });
        }
    }

    fn generate(& mut self) {
        let mut is_generate: bool = true;
        let mut x: i8;
        while is_generate {
            self.puzzle = vec![];
            while self.puzzle.len() < usize::try_from(self.size_h * self.size_v - 1).unwrap() {
                x = rand::thread_rng().gen_range(1,self.size_h * self.size_v);
                if self.puzzle.iter().position(|&r| r == x) == None {
                    self.puzzle.push(x);
                }
            }

            let mut summ:i32 = 0;
            let mut _b:i32 = 0;
            for (c, i) in self.puzzle.iter().enumerate() {
                if *i != 0 {
                    _b = 0;
                    let s_puzzle:Vec<i8> = self.puzzle[c..self.puzzle.len()].to_vec();
                    for j in s_puzzle.iter() {
                        if *j < *i && *j!=0 {
                            _b += 1;
                        }
                    }
                    summ = summ + _b;
                }
            }
            if summ % 2 == 0 {
                is_generate = false;
            }
        }
        self.puzzle.push(0);
        self.start = self.puzzle.clone();
    }

    fn set_puzzle(& mut self, puz:Vec<i8>) -> PyResult<bool> {
        let mut is_good:bool = false;
        if puz.len() as i8 % (self.size_v*self.size_h) !=0 {
            Ok(is_good)
        } else {
            self.puzzle = puz.clone();
            self.start = puz.clone();
            let mut summ:i32 = 0;
            let mut _b:i32 = 0;
            for (c, i) in self.puzzle.iter().enumerate() {
                if *i != 0 {
                    _b = 0;
                    let s_puzzle:Vec<i8> = self.puzzle[c..self.puzzle.len()].to_vec();
                    for j in s_puzzle.iter() {
                        if *j < *i && *j!=0 {
                            _b += 1;
                        }
                    }
                    summ = summ + _b;
                }
            }
            if summ % 2 == 0 {
                is_good = true;
            }
            Ok(is_good)
        }
    }

    fn search_solution(& self) -> PyResult<Vec<Step>> {
        let h:i8 = self.size_h;
        let v:i8 = self.size_v;
        let mut path_map:Vec<Step> = vec![];
        let mut h_open_sets:HashMap<Vec<i8>, Step> = HashMap::with_capacity(100000);
        let mut h_close_sets:HashMap<Vec<i8>, Step> = HashMap::with_capacity(1000000);
        let mut sets: Vec<Vec<i8>>;
        let mut g:u32 = 0;
        let mut f:u32;
        let (cost, h) = self.cost(self.start.clone());
        f = g+h;
        h_open_sets.insert(self.start.clone(), Step{hash_prev: 0, set:Set{g:g, h:h, f:f, position:self.start.clone()},
            hash_current: calculate_hash(&Set{g:g, h:h, f:f, position:self.start.clone()})});
        let mut sys_time = SystemTime::now();
        while h_open_sets.len() !=0 {
            let mut now = SystemTime::now();
            if now > sys_time + Duration::from_secs(3) {
                print!("\rValue of calculated states - {:?}", h_close_sets.len());
                sys_time = now;
            }
            let mut val = get_best_step(&h_open_sets, f);
                //h_open_sets.remove(&val);
            let mut prev = h_open_sets.remove(&val).unwrap();
            h_close_sets.insert(val, prev.clone());
            if prev.set.position == self.goal {
                path_map.push(prev.clone());
                while prev.hash_prev != 0 {
                    let mut val = get_by_hash(&mut h_close_sets, prev.hash_prev);
                    prev = h_close_sets.remove(&val).unwrap();
                    path_map.push(prev.clone());
                }
                break;
            }
            sets = self.search_sets(prev.clone().set.position);
            for new in sets {
                match h_close_sets.get(&new) {
                    Some(result) => {continue},
                    None => {}
                }
                let tentative_g_score:u32 = prev.set.g + 1;
                let mut tentative_is_better:bool = false;
                let mut tentative_is_better_from_other: bool = false;
                match h_open_sets.get(&new) {
                    Some(pos_opt) => {
                        if tentative_g_score < pos_opt.set.g {
                            tentative_is_better = true;
                            tentative_is_better_from_other = true;
                        }
                    },
                    None => {
                        tentative_is_better = true;
                    }
                }
                if tentative_is_better_from_other {
                    h_open_sets.remove(&new);
                }
                if tentative_is_better {
                    let (cost_, h_) = self.cost(new.clone());
                    if cost_ {
                        g = tentative_g_score;
                        f = g+h_;
                        let set:Set = Set{g:g, h:h_, f:f, position:new};
                        let hash = calculate_hash(&set);
                        h_open_sets.insert(set.position.clone(),Step{hash_prev:prev.clone().hash_current, set:set, hash_current:hash});

                    }
                }

            }
        }
        return Ok(path_map);
    }

    fn search_sets(& self, prev: Vec<i8>) -> Vec<Vec<i8>> {
        if prev.len()  != usize::try_from(self.size_h * self.size_v).unwrap() {
            vec![]
        } else {
            let mut tmp: Vec<i8> = vec![];
            let mut result: Vec<Vec<i8>> = vec![];
            let mut c:i8;
            let pos:i8;
            let pos_opt:Option<usize>;

                pos_opt = prev.iter().position(|&r| r == 0);

               if pos_opt == None {
                result
            } else {
                   pos = pos_opt.unwrap() as i8;
                   if pos / self.size_h > 0 {
                       tmp = prev.clone();
                       c = prev[(pos - self.size_h) as usize];
                       tmp[(pos - self.size_h) as usize] = 0;
                       tmp[pos as usize] = c;
                       result.push(tmp);
                   }
                   if (pos + self.size_h) < (self.size_v*self.size_h) {
                       tmp = prev.clone();
                       c = prev[(pos + self.size_h) as usize];
                       tmp[(pos + self.size_h) as usize] = 0;
                       tmp[pos as usize] = c;
                       result.push(tmp);
                   }
                   if pos % self.size_v > 0 {
                       tmp = prev.clone();
                       c = prev[(pos - 1) as usize];
                       tmp[(pos - 1) as usize] = 0;
                       tmp[pos as usize] = c;
                       result.push(tmp);
                   }
                   if (pos % self.size_h) < (self.size_h-1) {
                       tmp = prev.clone();
                       c = prev[(pos + 1) as usize];
                       tmp[(pos + 1) as usize] = 0;
                       tmp[pos as usize] = c;
                       result.push(tmp);
                   }
                   result
               }
        }
    }

    fn search_all_sets(& self, prev: Vec<i8>) -> Vec<Vec<i8>> {
        if prev.len()  != usize::try_from(self.size_h * self.size_v).unwrap() {
            vec![]
        } else {
            let mut tmp: Vec<i8> = vec![];
            let mut result: Vec<Vec<i8>> = vec![];
            let mut c:i8;
            let pos:i8;
            let pos_opt:Option<usize>;

            pos_opt = prev.iter().position(|&r| r == 0);

            if pos_opt == None {
                result
            } else {
                pos = pos_opt.unwrap() as i8;
                if pos / self.size_h > 0 {
                    tmp = prev.clone();
                    c = prev[(pos - self.size_h) as usize];
                    tmp[(pos - self.size_h) as usize] = 0;
                    tmp[pos as usize] = c;
                    result.push(tmp);
                } else {
                    tmp = prev.clone();
                    tmp[(pos as usize)] = -1;
                    result.push(tmp);
                }
                if (pos + self.size_h) < (self.size_v*self.size_h) {
                    tmp = prev.clone();
                    c = prev[(pos + self.size_h) as usize];
                    tmp[(pos + self.size_h) as usize] = 0;
                    tmp[pos as usize] = c;
                    result.push(tmp);
                } else {
                    tmp = prev.clone();
                    tmp[(pos as usize)] = -1;
                    result.push(tmp);
                }
                if pos % self.size_v > 0 {
                    tmp = prev.clone();
                    c = prev[(pos - 1) as usize];
                    tmp[(pos - 1) as usize] = 0;
                    tmp[pos as usize] = c;
                    result.push(tmp);
                } else {
                    tmp = prev.clone();
                    tmp[(pos as usize)] = -1;
                    result.push(tmp);
                }
                if (pos % self.size_h) < (self.size_h-1) {
                    tmp = prev.clone();
                    c = prev[(pos + 1) as usize];
                    tmp[(pos + 1) as usize] = 0;
                    tmp[pos as usize] = c;
                    result.push(tmp);
                } else {
                    tmp = prev.clone();
                    tmp[(pos as usize)] = -1;
                    result.push(tmp);
                }
                result
            }
        }
    }

    fn check_linear_conflict(& self, idx: i8, line: Vec<i8>) -> u32 {
        let mut is_conflict:u32 = 0;
        if line.len() as i8 > self.size_h {
            is_conflict
        } else {
            for (c, j) in line[..line.len()-1].iter().enumerate() {
                if *j > (0 + idx*self.size_h) && (*j <= self.size_h*(idx+1)) {
                    for i in line[c + 1..line.len()].to_vec() {
                        if i > (0 + idx*self.size_h) && (i <= self.size_h*(idx+1)) && *j>i {
                            is_conflict = 2;
                            break;
                        }
                    }
                    if is_conflict > 0 {
                        break;
                    }
                }
            }
            is_conflict
        }
    }

    fn check_column_conflict(& self, idx: i8, line: Vec<i8>) -> u32 {
        let mut is_conflict:u32 = 0;
        if line.len() as i8 > self.size_v {
            is_conflict
        } else {
            for (c, j) in line[..line.len()-1].iter().enumerate() {
                if *j % self.size_h == ((idx+1) % self.size_h) {
                    for i in line[c + 1..line.len()].to_vec() {
                        if i!=0 && (i % self.size_h) == ((idx+1) % self.size_h)
                            && *j/self.size_h > i/self.size_h {
                            is_conflict = 2;
                            break;
                        }
                    }
                }
                if is_conflict > 0 {
                    break;
                }
            }
            is_conflict
        }
    }

    fn cost(& self, line: Vec<i8>) -> (bool,u32) {
        let mut cost:u32 = 0;
        if line.len() as i8 != self.size_v*self.size_h {
            (false, cost)
        } else {
            for  (c, j) in line.iter().enumerate() {
                if *j != 0 {
                    let mut v: i8 = (((*j-1) / self.size_v) - (c as i8 / self.size_v)).abs();
                    let mut h: i8 = (((*j-1) % self.size_h) - (c as i8 % self.size_h)).abs();
                    cost = cost + v as u32 + h as u32;
                }
            }
            if cost > 0 && self.size_v > 2 && self.size_h > 2 {

                // check linear conflict for all lines
                for i in 0..self.size_v {
                        cost = cost + self.check_linear_conflict(i,
                                                                 line[(self.size_h*i) as usize..((self.size_h)*(i+1)) as usize].to_vec());
                }
                // check column conflict for all columns
                for i in 0..self.size_h {
                    let mut col: Vec<i8> = vec![];
                    for j in 0..self.size_v {
                        col.push(line[(i + j*self.size_h) as usize]);
                    }
                    cost = cost + self.check_column_conflict(i, col);

                }
                // check last move conflict
                let mut position1: i8;
                let mut position2: i8;
                position1 = line.iter().position(|&r| r == (self.size_h - 1) * self.size_v).unwrap() as i8;
                position2 = line.iter().position(|&r| r == self.size_h * self.size_v - 1).unwrap() as i8;
                if (position2 + 1) % self.size_h != 0 && (position1 < self.size_h*(self.size_v-1)) {
                    cost +=2;
                }
                // check left top agle on conflict
                if line[1] == 2 && line[self.size_h as usize]==self.size_h +1 && line[0] != 1{
                    cost +=2;
                }
                // check right top agle on conflict
                if line[(self.size_h -2) as usize] == self.size_h - 1 &&
                    line[(2*self.size_h -1) as usize] == self.size_h*2  &&
                    line[(self.size_h - 1) as usize] != self.size_h {
                    cost +=2;

                }
                // check left bottom  agle on conflict
                if line[(self.size_h*(self.size_v - 2)) as usize] == self.size_h*(self.size_v - 2) &&
                    line[(self.size_h*(self.size_v-1) + 1) as usize] == self.size_h*(self.size_v-1) + 2 &&
                    line[(self.size_h*(self.size_v-1)) as usize] != self.size_h*(self.size_v-1) +1 {
                    cost +=2;
                }

            }
            (true, cost)
        }
    }

}

#[pymodule]
fn puzzlelib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    //m.add_wrapped(wrap_pyfunction!(set_puzzle))?;
    //m.add_class::<Set>()?;
    //m.add_class::<Step>()?;
    m.add_class::<Puzzle>()?;

    Ok(())
}
