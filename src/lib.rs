mod mjso;
pub use mjso::mjso_optimize;

#[cfg(test)]
mod tests {
    use super::*;

    /// Sphere function: f(x) = x1^2 + x2^2 + ... + xn^2
    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    #[test]
    fn test_sphere_minimization() {
        let dim = 30;
        // limit to a few dozen thousand evals so CI won't time out
        let max_evals = Some(50_000);
        // leave other parameters at their defaults
        let (best, val) = mjso_optimize(
            sphere, dim, None, None, max_evals, None, None, None, None, None, None,
        );

        // Near-zero function value should be found
        assert!(
            val < 1e-15,
            "Sphere value too large: got {} (expected < 1e-15)",
            val
        );

        // And each coordinate should be near 0
        println!("Best solution:");
        for xi in best {
            println!("  {}", xi);
            assert!(
                xi.abs() < 1e-5,
                "Coordinate too far from zero: got {} (expected < 1e-5)",
                xi
            );
        }

        // Print the best f value for debugging
        println!("Best value: {}", val);
    }
}
