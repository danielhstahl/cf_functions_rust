use num_complex::Complex;

/// Returns log of Poisson jump characteristic function with Gaussian jumps
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let lambda = 0.5; //jump frequency
/// let mu_l = 0.5; //mean of jump
/// let sigma_l = 0.3; //volatility of jump
/// let log_cf = cf_functions::merton::merton_log_cf(
///     &u, lambda, mu_l, sigma_l
/// );
/// # }
/// ```
pub fn merton_log_cf(u: &Complex<f64>, lambda: f64, mu_l: f64, sig_l: f64) -> Complex<f64> {
    lambda * (crate::gauss::gauss_cf(u, mu_l, sig_l) - 1.0)
}

/// Returns log of Merton jump diffusion characteristic function with Gaussian jumps, adjusted to be risk-neutral
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let lambda = 0.5; //jump frequency
/// let mu_l = 0.5; //mean of jump
/// let sigma_l = 0.3; //volatility of jump
/// let sigma = 0.3; //volatility of diffusion
/// let rate = 0.04; //risk free rate
/// let log_cf = cf_functions::merton::merton_log_risk_neutral_cf(
///     &u, lambda, mu_l, sigma_l, rate, sigma
/// );
/// # }
/// ```
pub fn merton_log_risk_neutral_cf(
    u: &Complex<f64>,
    lambda: f64,
    mu_l: f64,
    sig_l: f64,
    rate: f64,
    sigma: f64,
) -> Complex<f64> {
    let cmp_mu =
        rate - 0.5 * sigma.powi(2) - merton_log_cf(&Complex::new(1.0, 0.0), lambda, mu_l, sig_l);
    crate::gauss::gauss_log_cf_cmp(u, &cmp_mu, sigma) + merton_log_cf(u, lambda, mu_l, sig_l)
}

/// Returns log of time changed Merton jump diffusion characteristic function with Gaussian jumps with correlation between the diffusion of the time changed process and the underlying.
///
/// # Remarks
/// The time change is assumed to be a CIR process with long run mean of 1.0.
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let lambda = 0.5; //jump frequency
/// let mu_l = 0.5; //mean of jump
/// let sigma_l = 0.3; //volatility of jump
/// let sigma = 0.3; //volatility of underlying diffusion
/// let t = 0.5; //time horizon
/// let speed = 0.5; //speed of CIR process
/// let v0 = 0.9; //initial value of CIR process
/// let eta_v = 0.3; //volatility of CIR process
/// let rho = -0.5; //correlation between diffusions
/// let log_cf = cf_functions::merton::merton_time_change_log_cf(
///     &u, t, lambda, mu_l, sigma_l,
///     sigma, v0, speed, eta_v, rho
/// );
/// # }
/// ```
pub fn merton_time_change_log_cf(
    u: &Complex<f64>,
    t: f64,
    lambda: f64,
    mu_l: f64,
    sig_l: f64,
    sigma: f64,
    v0: f64,
    speed: f64,
    eta_v: f64,
    rho: f64,
) -> Complex<f64> {
    crate::affine_process::generic_leverage_diffusion(
        u,
        &|u| merton_log_risk_neutral_cf(u, lambda, mu_l, sig_l, 0.0, sigma),
        t,
        sigma,
        v0,
        speed,
        eta_v,
        rho,
    )
}

/// Returns cf function of a time changed Merton jump diffusion characteristic function with Gaussian jumps with correlation between the diffusion of the time changed process and the underlying, adjusted to be risk neutral.
///
/// # Remarks
/// The time change is assumed to be a CIR process with long run mean of 1.0.
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let lambda = 0.5; //jump frequency
/// let mu_l = 0.5; //mean of jump
/// let sigma_l = 0.3; //volatility of jump
/// let sigma = 0.3; //volatility of underlying diffusion
/// let t = 0.5; //time horizon
/// let rate = 0.05;
/// let speed = 0.5; //speed of CIR process
/// let v0 = 0.9; //initial value of CIR process
/// let eta_v = 0.3; //volatility of CIR process
/// let rho = -0.5; //correlation between diffusions
/// let cf = cf_functions::merton::merton_time_change_cf(
///     t, rate, lambda, mu_l, sigma_l,
///     sigma, v0, speed, eta_v, rho
/// );
/// let value_of_cf=cf(&Complex::new(0.05, -0.5));
/// # }
/// ```
pub fn merton_time_change_cf(
    t: f64,
    rate: f64,
    lambda: f64,
    mu_l: f64,
    sig_l: f64,
    sigma: f64,
    v0: f64,
    speed: f64,
    eta_v: f64,
    rho: f64,
) -> impl Fn(&Complex<f64>) -> Complex<f64> {
    move |u| {
        (rate * t * u
            + merton_time_change_log_cf(u, t, lambda, mu_l, sig_l, sigma, v0, speed, eta_v, rho))
        .exp()
    }
}

/// Returns volatility of Merton jump diffusion
pub fn jump_diffusion_vol(sigma: f64, lambda: f64, mu_l: f64, sig_l: f64, maturity: f64) -> f64 {
    ((sigma.powi(2) + lambda * (mu_l.powi(2) + sig_l.powi(2))) * maturity).sqrt()
}
