use num_complex::Complex;

/// Returns log of Gaussian characteristic function
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let mu = 0.5;
/// let sigma = 0.3;
/// let log_cf = cf_functions::gauss::gauss_log_cf(
///     &u, mu, sigma
/// );
/// # }
/// ```
pub fn gauss_log_cf(u: &Complex<f64>, mu: f64, sigma: f64) -> Complex<f64> {
    u * mu + u * u * 0.5 * sigma.powi(2)
}

/// Returns log of Gaussian characteristic function with complex mu
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let mu = Complex::new(1.0, 1.0);
/// let sigma = 0.3;
/// let log_cf = cf_functions::gauss::gauss_log_cf_cmp(
///     &u, &mu, sigma
/// );
/// # }
/// ```
pub fn gauss_log_cf_cmp(u: &Complex<f64>, mu: &Complex<f64>, sigma: f64) -> Complex<f64> {
    u * mu + u * u * 0.5 * sigma.powi(2)
}
/// Returns Gaussian characteristic function
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let mu = 0.5;
/// let sigma = 0.3;
/// let cf = cf_functions::gauss::gauss_cf(
///     &u, mu, sigma
/// );
/// # }
/// ```
pub fn gauss_cf(u: &Complex<f64>, mu: f64, sigma: f64) -> Complex<f64> {
    gauss_log_cf(u, mu, sigma).exp()
}

/// Returns Heston model log CF.  Heston is simply a time changed gaussian model.
///
/// # Remarks
/// The time change is assumed to be a CIR process.
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let sigma = 0.3; //square root of long run average
/// let t = 0.5; //time horizon
/// let speed = 0.5; //speed of mear reversion CIR process
/// let v0 = 0.25; //initial value of CIR process
/// let eta_v = 0.3; //volatility of CIR process (vol of vol)
/// let rho = -0.5; //correlation
/// let log_cf = cf_functions::gauss::heston_log_cf(
///     &u, t, sigma, v0,
///     speed, eta_v, rho
/// );
/// # }
/// ```
pub fn heston_log_cf(
    u: &Complex<f64>,
    t: f64,
    sigma: f64,
    v0: f64,
    speed: f64,
    eta_v: f64,
    rho: f64,
) -> Complex<f64> {
    let sigma_sq = sigma.powi(2);
    crate::cir::generic_leverage_diffusion(
        u,
        &|u| gauss_log_cf(u, -0.5 * sigma_sq, sigma),
        t,
        sigma,
        v0 / sigma_sq,
        speed,
        eta_v / sigma,
        rho,
    )
}

/// Returns Heston model log CF.  Heston is simply a time changed gaussian model.
///
/// # Remarks
/// The time change is assumed to be a CIR process
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let sigma = 0.3; //square root of long run average
/// let t = 0.5; //time horizon
/// let rate = 0.05;
/// let speed = 0.5; //speed of mean reversion of CIR process
/// let v0 = 0.29; //initial value of CIR process
/// let eta_v = 0.3; //volatility of CIR process (vol of vol)
/// let rho = -0.5; //correlation between diffusions
/// let cf = cf_functions::gauss::heston_cf(
///     t, rate,
///     sigma, v0, speed, eta_v, rho
/// );
/// let value_of_cf=cf(&Complex::new(0.05, -0.5));
/// # }
/// ```
pub fn heston_cf(
    t: f64,
    rate: f64,
    sigma: f64,
    v0: f64,
    speed: f64,
    eta_v: f64,
    rho: f64,
) -> impl Fn(&Complex<f64>) -> Complex<f64> {
    move |u| (rate * t * u + heston_log_cf(u, t, sigma, v0, speed, eta_v, rho)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cir_heston_2() {
        let t = 0.25;
        let k = 0.2;
        let v0 = 0.98;
        let sig = 0.2;
        let rho = -0.3;
        let sig_tot = 0.3;
        let u = Complex::new(0.5, 0.5);
        let neg_psi = 0.5 * sig_tot * sig_tot * (u - u * u);
        let k_star = k - u * rho * sig * sig_tot;
        let ada = (k_star * k_star + 2.0 * sig * sig * neg_psi as Complex<f64>).sqrt();
        let b_t = 2.0 * neg_psi * (1.0 - (-ada * t).exp())
            / (2.0 * ada - (ada - k_star) * (1.0 - (-ada * t).exp()));
        let c_t = (k / (sig * sig))
            * (2.0 * (1.0 - (1.0 - (-ada * t).exp()) * (ada - k_star) / (2.0 * ada)).ln()
                + (ada - k_star) * t);
        let cf_heston = (-b_t * v0 - c_t).exp().re;
        let approx_heston_cf =
            heston_cf(t, 0.0, sig_tot, v0 * sig_tot.powi(2), k, sig * sig_tot, rho)(&u).re;
        assert_eq!(cf_heston, approx_heston_cf);
    }
}
