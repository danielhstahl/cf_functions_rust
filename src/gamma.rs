use num_complex::Complex;

/// Returns characteristic function of a gamma distribution.
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let a = 0.5;
/// let b = 0.6;
/// let cf = cf_functions::gamma::gamma_cf(
///     &u, a, b
/// );
/// # }
/// ```
pub fn gamma_cf(u: &Complex<f64>, a: f64, b: f64) -> Complex<f64> {
    (1.0 - u * b).powf(-a)
}

//for gamma distribution
fn gamma_log(
    u: &Complex<f64>,
    t: f64,
    v0: f64,
    a: f64,
    sigma: f64,
    lambda: f64,
    correlation: f64,
    alpha: f64,
    beta: f64,
    num_steps: usize,
) -> Complex<f64> {
    crate::cir::cir_leverage_jump(
        u,
        &|u| gamma_cf(u, alpha, beta),
        t,
        v0,
        correlation,
        alpha * beta,
        a,
        sigma,
        lambda,
        num_steps,
    )
}

/// Returns log CF of an gamma jump diffusion when transformed by an affine process
/// and the process is correlated with the jump component of the Levy process.
///
/// # Remarks
/// The time change is assumed to be a single-dimensional CIR process with a
/// jump component with mean 1.
/// The correlation between the Levy process and the affine process is due
/// to sharing the same jumps (both the Levy process and the affine process
/// jump at the same time).
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let alpha=2.0;
/// let beta=3.0;
/// let alpha=1.1;
/// let lambda=100.0;
/// let correlation=0.9;
/// let a=0.4;
/// let sigma=0.4;
/// let t=1.0;
/// let v0=1.0;
/// let num_steps:usize=1024;
/// let cf=|u:&Complex<f64>|u.exp();
/// let cf=cf_functions::gamma::gamma_leverage(
///     t, v0, a, sigma, lambda,
///     correlation, alpha, beta, num_steps
/// );
/// let u=Complex::new(0.05, -0.5);
/// let value_of_cf=cf(&u);
/// # }
/// ```
pub fn gamma_leverage(
    t: f64,
    v0: f64,
    a: f64,
    sigma: f64,
    lambda: f64,
    correlation: f64,
    alpha: f64,
    beta: f64,
    num_steps: usize,
) -> impl Fn(&Complex<f64>) -> Complex<f64> {
    move |u| {
        gamma_log(
            u,
            t,
            v0,
            a,
            sigma,
            lambda,
            correlation,
            alpha,
            beta,
            num_steps,
        )
        .exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    #[test]
    fn test_gamma_leverage_expectation() {
        let t = 2.0;
        let num_steps = 1024;
        let v0 = 1.0;
        let a = 0.4;
        let sigma = 0.4;
        let lambda = 100.0;
        let correlation = 0.9;
        let alpha = 2.0;
        let beta = 4.0;
        let x_min = 0.0;
        let x_max = lambda * alpha * beta * beta * 5.0 * t;
        let num_u: usize = 256;
        let cf = gamma_leverage(t, v0, a, sigma, lambda, correlation, alpha, beta, num_steps);
        let discrete_cf = fang_oost::get_discrete_cf(num_u, x_min, x_max, &cf);
        let expectation = cf_dist_utils::get_expectation_discrete_cf(x_min, x_max, &discrete_cf);
        assert_abs_diff_eq!(expectation, lambda * alpha * beta * t, epsilon = 0.00001);
    }
    #[test]
    fn test_gamma_leverage_expectation_leq() {
        let t = 2.0;
        let num_steps = 1024;
        let v0 = 0.9;
        let a = 0.4;
        let sigma = 0.4;
        let lambda = 100.0;
        let correlation = 0.9;
        let alpha = 2.0;
        let beta = 4.0;
        let x_min = 0.0;
        let x_max = lambda * alpha * beta * beta * 5.0 * t;
        let num_u: usize = 256;
        let cf = gamma_leverage(t, v0, a, sigma, lambda, correlation, alpha, beta, num_steps);
        let discrete_cf = fang_oost::get_discrete_cf(num_u, x_min, x_max, &cf);
        let expectation = cf_dist_utils::get_expectation_discrete_cf(x_min, x_max, &discrete_cf);
        assert_eq!(expectation < lambda * alpha * beta * t, true);
    }
    #[test]
    fn test_gamma_leverage_expectation_geq() {
        let t = 2.0;
        let num_steps = 1024;
        let v0 = 1.1;
        let a = 0.4;
        let sigma = 0.4;
        let lambda = 100.0;
        let correlation = 0.9;
        let alpha = 2.0;
        let beta = 4.0;
        let x_min = 0.0;
        let x_max = lambda * alpha * beta * beta * 5.0 * t;
        let num_u: usize = 256;
        let cf = gamma_leverage(t, v0, a, sigma, lambda, correlation, alpha, beta, num_steps);
        let discrete_cf = fang_oost::get_discrete_cf(num_u, x_min, x_max, &cf);
        let expectation = cf_dist_utils::get_expectation_discrete_cf(x_min, x_max, &discrete_cf);
        assert_eq!(expectation > lambda * alpha * beta * t, true);
    }
}
