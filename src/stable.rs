use num_complex::Complex;
use std::f64::consts::PI;

fn compute_stable_phi(alpha: f64) -> f64 {
    (alpha * 0.5 * PI).tan()
}

fn stable_cf_memoize(
    u: &Complex<f64>,
    alpha: f64,
    mu: f64,
    beta: f64,
    c: f64,
    phi: f64,
) -> Complex<f64> {
    (u * mu - (u * Complex::new(0.0, -1.0) * c).powf(alpha) * Complex::new(1.0, -beta * phi)).exp()
}
/// Returns characteristic function of a stable distribution.
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let alpha = 0.5;
/// let mu = 0.5;
/// let beta = 0.3;
/// let c = 0.3;
/// let cf = cf_functions::stable::stable_cf(
///     &u, alpha, mu, beta, c
/// );
/// # }
/// ```
pub fn stable_cf(u: &Complex<f64>, alpha: f64, mu: f64, beta: f64, c: f64) -> Complex<f64> {
    let phi = compute_stable_phi(alpha);
    stable_cf_memoize(&u, alpha, mu, beta, c, phi)
}

const BETA_STABLE: f64 = 1.0; //to stay on the positive reals

//for stable distribution
fn alpha_stable_log(
    u: &Complex<f64>,
    t: f64,
    v0: f64,
    a: f64,
    sigma: f64,
    lambda: f64,
    correlation: f64,
    alpha: f64,
    mu: f64,
    c: f64,
    phi: f64,
    num_steps: usize,
) -> Complex<f64> {
    crate::cir::cir_leverage_jump(
        u,
        &|u| stable_cf_memoize(u, alpha, mu, BETA_STABLE, c, phi),
        t,
        v0,
        correlation,
        mu,
        a,
        sigma,
        lambda,
        num_steps,
    )
}

/// Returns log CF of an alpha stable process when transformed by an affine process
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
/// let mu=1300.0;
/// let c=100.0;
/// let alpha=1.1;
/// let lambda=100.0;
/// let correlation=0.9;
/// let a=0.4;
/// let sigma=0.4;
/// let t=1.0;
/// let v0=1.0;
/// let num_steps:usize=1024;
/// let cf=|u:&Complex<f64>|u.exp();
/// let cf=cf_functions::stable::alpha_stable_leverage(
///     t, v0, a, sigma, lambda,
///     correlation, alpha, mu, c, num_steps
/// );
/// let u=Complex::new(0.05, -0.5);
/// let value_of_cf=cf(&u);
/// # }
/// ```
pub fn alpha_stable_leverage(
    t: f64,
    v0: f64,
    a: f64,
    sigma: f64,
    lambda: f64,
    correlation: f64,
    alpha: f64,
    mu: f64,
    c: f64,
    num_steps: usize,
) -> impl Fn(&Complex<f64>) -> Complex<f64> {
    let phi = compute_stable_phi(alpha);
    move |u| {
        alpha_stable_log(
            u,
            t,
            v0,
            a,
            sigma,
            lambda,
            correlation,
            alpha,
            mu,
            c,
            phi,
            num_steps,
        )
        .exp()
    }
}
