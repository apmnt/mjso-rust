use rand::prelude::SliceRandom;
use rand::{Rng, rng};
use std::f64::consts::PI;

/// Normal distribution via Box–Muller transform.
fn sample_normal<R: Rng + ?Sized>(rng: &mut R, mean: f64, std: f64) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    mean + z0 * std
}

/// Cauchy distribution via inverse transform.
fn sample_cauchy<R: Rng + ?Sized>(rng: &mut R, loc: f64, scale: f64) -> f64 {
    let u: f64 = rng.random();
    loc + scale * ((PI * (u - 0.5)).tan())
}

/// Sample an index in `[0..weights.len())` proportional to non-negative `weights`.
/// Returns `None` if `weights` is empty or all zero.
fn sample_weighted_index<R: Rng + ?Sized>(rng: &mut R, weights: &[f64]) -> Option<usize> {
    // total weight
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return None;
    }

    // pick a point in [0, total)
    let mut target = rng.random::<f64>() * total;

    // walk until we exceed target
    for (i, &w) in weights.iter().enumerate() {
        target -= w;
        if target < 0.0 {
            return Some(i);
        }
    }

    // edge‐case due to floating error: clamp to last
    Some(weights.len() - 1)
}
/// Single-call MjSO optimizer with optional parameters.
pub fn mjso_optimize<F>(
    objective: F,
    dim: usize,
    lower_bounds: Option<&[f64]>,
    upper_bounds: Option<&[f64]>,
    max_evals: Option<usize>,
    init_pop: Option<usize>,
    min_pop: Option<usize>,
    memory_size: Option<usize>,
    p_min: Option<f64>,
    p_max: Option<f64>,
    m_scale: Option<f64>,
    tolerance: Option<f64>,
) -> (Vec<f64>, f64, i32)
where
    F: Fn(&[f64]) -> f64,
{
    // Defaults & bounds setup
    let lb = lower_bounds
        .map(|s| s.to_vec())
        .unwrap_or_else(|| vec![-100.0; dim]);
    let ub = upper_bounds
        .map(|s| s.to_vec())
        .unwrap_or_else(|| vec![100.0; dim]);
    let max_evals = max_evals.unwrap_or(dim * 10_000);
    let init_pop = init_pop.unwrap_or(20);
    let min_pop = min_pop.unwrap_or(4);
    let mem_size = memory_size.unwrap_or(5);
    let p_min = p_min.unwrap_or(0.125);
    let p_max = p_max.unwrap_or(0.25);
    let m_scale = m_scale.unwrap_or(6.0);
    let tolerance = tolerance.unwrap_or(1e-12);

    let mut rng = rng();

    // Historical memory
    let mut m_cr = vec![0.8; mem_size];
    let mut m_f = vec![0.3; mem_size];
    let last = mem_size - 1;
    m_cr[last] = 0.9;
    m_f[last] = 0.9;

    // Initialize population
    let mut pop: Vec<Vec<f64>> = (0..init_pop)
        .map(|_| (0..dim).map(|j| rng.random_range(lb[j]..ub[j])).collect())
        .collect();
    let mut fitness: Vec<f64> = pop.iter().map(|x| objective(x)).collect();
    let mut evals = init_pop;
    let mut mem_idx = 0;

    // Best value and solution
    let mut best_val = std::f64::INFINITY;
    let mut best_sol = vec![0.0; dim];
    let mut terminate = false;

    while evals < max_evals {
        // Population resizing
        let ratio = evals as f64 / max_evals as f64;
        let target =
            ((ratio * ((min_pop as f64) - (init_pop as f64))) + init_pop as f64).round() as usize;
        if target < pop.len() {
            let mut order: Vec<usize> = (0..pop.len()).collect();
            order.sort_by(|&i, &j| fitness[i].partial_cmp(&fitness[j]).unwrap());
            let keep = &order[..target];
            pop = keep.iter().map(|&i| pop[i].clone()).collect();
            fitness = keep.iter().map(|&i| fitness[i]).collect();
        }
        let cur_np = pop.len();
        let mut new_pop = pop.clone();
        let mut new_fit = fitness.clone();

        // For memory updates
        let mut s_cr = Vec::new();
        let mut s_f = Vec::new();
        let mut w_cr = Vec::new();
        let mut w_f = Vec::new();

        // Sort indices by fitness
        let mut sorted: Vec<usize> = (0..cur_np).collect();
        sorted.sort_by(|&i, &j| fitness[i].partial_cmp(&fitness[j]).unwrap());

        // Determine p_best count
        let p_rate = p_min + (p_max - p_min) * ratio;
        let mut npbest = (cur_np as f64 * p_rate).round() as usize;
        npbest = npbest.max(2).min(cur_np);

        // Build weighted index
        let weights: Vec<f64> = (1..=npbest)
            .map(|i| m_scale - (m_scale - 1.0) * ((i - 1) as f64) / ((npbest - 1) as f64))
            .collect();

        // Evolve each individual
        for i in 0..cur_np {
            // 1) Sample historic CR/F
            let r = rng.random_range(0..mem_size);
            let mu_cr = m_cr[r];
            let mu_f = m_f[r];

            // 2) Generate trial CR and F
            let cr = sample_normal(&mut rng, mu_cr, 0.1).clamp(0.0, 1.0);
            let mut f_i = sample_cauchy(&mut rng, mu_f, 0.1);
            while f_i <= 0.0 {
                f_i = sample_cauchy(&mut rng, mu_f, 0.1);
            }
            let f_i = f_i.min(1.0);

            // Adaptive F weighting
            let fw = if ratio < 0.2 {
                0.7 * f_i
            } else if ratio < 0.4 {
                0.8 * f_i
            } else {
                1.2 * f_i
            };

            // 3) Select p-best
            let p_idx = sample_weighted_index(&mut rng, &weights).unwrap();
            let x_p = &pop[sorted[p_idx]];

            // 4) Pick r1,r2
            let mut picks: Vec<usize> = (0..cur_np).filter(|&k| k != i).collect();
            picks.shuffle(&mut rng);
            let (r1, r2) = (picks[0], picks[1]);

            // Directed difference
            let diff: Vec<f64> = if fitness[r1] <= fitness[r2] {
                pop[r1].iter().zip(&pop[r2]).map(|(&a, &b)| a - b).collect()
            } else {
                pop[r2].iter().zip(&pop[r1]).map(|(&a, &b)| a - b).collect()
            };

            // 5) Mutation + Crossover
            let mut u = pop[i].clone();
            let jrand = rng.random_range(0..dim);
            for j in 0..dim {
                let mut v = pop[i][j] + fw * (x_p[j] - pop[i][j]) + f_i * diff[j];
                // boundary check
                v = if v < lb[j] {
                    0.5 * (lb[j] + pop[i][j])
                } else if v > ub[j] {
                    0.5 * (ub[j] + pop[i][j])
                } else {
                    v
                };
                if rng.random::<f64>() < cr || j == jrand {
                    u[j] = v;
                }
            }

            // 6) Selection
            let fit_u = objective(&u);
            evals += 1;
            if fit_u <= fitness[i] {
                new_pop[i] = u;
                new_fit[i] = fit_u;

                if fit_u < best_val {
                    best_val = fit_u;
                    best_sol = new_pop[i].clone();

                    if best_val < tolerance {
                        terminate = true;
                        break;
                    }
                }

                let w = (fitness[i] - fit_u).abs();
                s_cr.push(cr);
                w_cr.push(w);
                s_f.push(f_i);
                w_f.push(w);
            }
        }

        if terminate {
            break;
        }

        // 7) Memory update
        if !s_cr.is_empty() {
            let num_cr: f64 = s_cr.iter().zip(&w_cr).map(|(&c, &w)| w * c * c).sum();
            let den_cr: f64 = s_cr.iter().zip(&w_cr).map(|(&c, &w)| w * c).sum();
            let mean_cr = num_cr / den_cr;
            let num_f: f64 = s_f.iter().zip(&w_f).map(|(&f_, &w)| w * f_ * f_).sum();
            let den_f: f64 = s_f.iter().zip(&w_f).map(|(&f_, &w)| w * f_).sum();
            let mean_f = num_f / den_f;
            m_cr[mem_idx] = 0.5 * (mean_cr + m_cr[mem_idx]);
            m_f[mem_idx] = 0.5 * (mean_f + m_f[mem_idx]);
            mem_idx = (mem_idx + 1) % mem_size;
            m_cr[last] = 0.9;
            m_f[last] = 0.9;
        }

        pop = new_pop;
        fitness = new_fit;
    }

    // Return best solution, function value, and the number of evaluations
    (best_sol, best_val, evals as i32)
}
