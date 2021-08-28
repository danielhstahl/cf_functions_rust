use num_complex::Complex;
use special::Gamma;

/// Returns log of CGMY characteristic function
///
/// # Remarks
///
/// See [cgmy](http://finance.martinsewell.com/stylized-facts/distribution/CarrGemanMadanYor2002.pdf) pg 10
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let c = 0.5;
/// let g = 4.0;
/// let m = 3.0;
/// let y = 0.6;
/// let log_cf = cf_functions::cgmy::cgmy_log_cf(
///     &u, c, g, m, y
/// );
/// # }
/// ```
pub fn cgmy_log_cf(u: &Complex<f64>, c: f64, g: f64, m: f64, y: f64) -> Complex<f64> {
    if crate::utils::is_same(y, 1.0) {
        Complex::new(0.0, 0.0)
    } else if crate::utils::is_same(y, 0.0) {
        c * (1.0 - u / g).ln() * (1.0 + u / m)
    } else {
        c * (-y).gamma() * ((m - u).powf(y) + (g + u).powf(y) - m.powf(y) - g.powf(y))
    }
}

/// Returns log of CGMY-diffusion characteristic function adjusted to be risk neutral
///
/// # Remarks
///
/// See [cgmy](http://finance.martinsewell.com/stylized-facts/distribution/CarrGemanMadanYor2002.pdf) pg 12 and 13
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let c = 0.5;
/// let g = 4.0;
/// let m = 3.0;
/// let y = 0.6;
/// let rate = 0.05; //risk free rate
/// let sigma = 0.3; //volatility of diffusion
/// let log_cf = cf_functions::cgmy::cgmy_log_risk_neutral_cf(
///     &u, c, g, m, y, rate, sigma
/// );
/// # }
/// ```
pub fn cgmy_log_risk_neutral_cf(
    u: &Complex<f64>,
    c: f64,
    g: f64,
    m: f64,
    y: f64,
    rate: f64,
    sigma: f64,
) -> Complex<f64> {
    let cmp_mu = rate - sigma.powi(2) * 0.5 - cgmy_log_cf(&Complex::new(1.0, 0.0), c, g, m, y);
    crate::gauss::gauss_log_cf_cmp(u, &cmp_mu, sigma) + cgmy_log_cf(u, c, g, m, y)
}

/// Returns log of time changed CGMY characteristic function with correlation between the diffusion of the time changed process and the underlying.
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
/// let c = 0.5;
/// let g = 4.0;
/// let m = 3.0;
/// let y = 0.6;
/// let sigma = 0.3; //volatility of underlying diffusion
/// let t = 0.5; //time horizon
/// let speed = 0.5; //speed of CIR process
/// let v0 = 0.9; //initial value of CIR process
/// let eta_v = 0.3; //volatility of CIR process
/// let rho = -0.5; //correlation between diffusions
/// let log_cf = cf_functions::cgmy::cgmy_time_change_log_cf(
///     &u, t, c, g, m, y,
///     sigma, v0, speed, eta_v, rho
/// );
/// # }
/// ```
pub fn cgmy_time_change_log_cf(
    u: &Complex<f64>,
    t: f64,
    c: f64,
    g: f64,
    m: f64,
    y: f64,
    sigma: f64,
    v0: f64,
    speed: f64,
    eta_v: f64,
    rho: f64,
) -> Complex<f64> {
    crate::cir::generic_leverage_diffusion(
        u,
        &|u| cgmy_log_risk_neutral_cf(u, c, g, m, y, 0.0, sigma),
        t,
        sigma,
        v0,
        speed,
        eta_v,
        rho,
    )
}

//see https://poseidon01.ssrn.com/delivery.php?ID=737027111000006077113070089110095064016020050037028066000080065074127006086092092026061120060015055036110006010126103066122080108059078076004070004065091125021108014077028121011029092117112080127092065007111098070065099086069122086067104098093017117&EXT=pdf&INDEX=TRUE
// page 11
fn leverage_neutral_pure_jump_log_cf(
    u: &Complex<f64>,
    cf_negative: &dyn Fn(&Complex<f64>) -> Complex<f64>, //
    cf: &dyn Fn(&Complex<f64>) -> Complex<f64>,
    expected_value_jump: f64, //negative only
    speed: f64,
    eta_v: f64,
    sigma: f64,
    v0: f64,
    t: f64,
    num_steps: usize,
) -> Complex<f64> {
    let init_value_1 = Complex::new(0.0, 0.0);
    let init_value_2 = Complex::new(0.0, 0.0);
    let adjusted_kappa = 1.0 - eta_v * expected_value_jump; //get expectation equal to one
    let theta = speed / adjusted_kappa;
    let fx = move |_t: f64, _alpha_prev: &Complex<f64>, beta_prev: &Complex<f64>| {
        let u_sig = sigma * u;
        let u_extended = beta_prev * Complex::new(0.0, 1.0) * eta_v + u_sig;
        let beta = cf_negative(&u_extended) - cf_negative(&u_sig) - speed * beta_prev + cf(u);
        let alpha = speed * theta * beta_prev;
        (alpha, beta)
    };
    let (alpha, beta) =
        crate::cir::runge_kutta_complex_vector(&fx, init_value_1, init_value_2, t, num_steps);
    beta * v0 + alpha
}

/// /// Returns time changed (self-exciting) CGMY characteristic function with
/// correlation between the pure-jump time changed process and the underlying.
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
/// let c = 0.5;
/// let g = 4.0;
/// let m = 3.0;
/// let y = 0.6;
/// let sigma = 0.3; //volatility of underlying asset
/// let t = 0.5; //time horizon
/// let rate = 0.05;
/// let speed = 0.5; //speed of volatility process
/// let v0 = 0.9; //initial value of volatility process
/// let eta_v = 0.3; //volatility of volatility process
/// let cf = cf_functions::cgmy::cgmyse_time_change_cf(
///     t, rate, c, g, m, y,
///     sigma, v0, speed, eta_v
/// );
/// let value_of_cf=cf(&Complex::new(0.05, -0.5));
/// # }
/// ```
pub fn cgmyse_time_change_cf(
    t: f64,
    rate: f64,
    c: f64,
    g: f64,
    m: f64,
    y: f64,
    sigma: f64,
    v0: f64,
    speed: f64,
    eta_v: f64,
    num_steps: usize,
) -> impl Fn(&Complex<f64>) -> Complex<f64> {
    let expected_value_jump = cgmy_negative_jump_expectation(c, g, y);
    move |u: &Complex<f64>| {
        (leverage_neutral_pure_jump_log_cf(
            &u,
            &|u| cgmy_log_cf(&u, c, g, 0.0, y),
            &|u| cgmy_log_risk_neutral_cf(&u, c, g, m, y, 0.0, 0.0),
            expected_value_jump,
            speed,
            eta_v,
            sigma,
            v0,
            t,
            num_steps,
        ) + rate * u)
            .exp()
    }
}

/// Returns log of time changed CGMY characteristic function with correlation between the diffusion of the time changed process and the underlying.
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
/// let c = 0.5;
/// let g = 4.0;
/// let m = 3.0;
/// let y = 0.6;
/// let sigma = 0.3; //volatility of underlying diffusion
/// let t = 0.5; //time horizon
/// let rate = 0.05;
/// let speed = 0.5; //speed of CIR process
/// let v0 = 0.9; //initial value of CIR process
/// let eta_v = 0.3; //volatility of CIR process
/// let rho = -0.5; //correlation between diffusions
/// let cf = cf_functions::cgmy::cgmy_time_change_cf(
///     t, rate, c, g, m, y,
///     sigma, v0, speed, eta_v, rho
/// );
/// let value_of_cf=cf(&Complex::new(0.05, -0.5));
/// # }
/// ```
pub fn cgmy_time_change_cf(
    t: f64,
    rate: f64,
    c: f64,
    g: f64,
    m: f64,
    y: f64,
    sigma: f64,
    v0: f64,
    speed: f64,
    eta_v: f64,
    rho: f64,
) -> impl Fn(&Complex<f64>) -> Complex<f64> {
    move |u| {
        (rate * t * u + cgmy_time_change_log_cf(u, t, c, g, m, y, sigma, v0, speed, eta_v, rho))
            .exp()
    }
}

pub fn cgmy_negative_jump_expectation(c: f64, g: f64, y: f64) -> f64 {
    c * (1.0 - y).gamma() * g.powf(y - 1.0)
}

/// Returns volatility of CGMY
pub fn cgmy_diffusion_vol(sigma: f64, c: f64, g: f64, m: f64, y: f64, maturity: f64) -> f64 {
    ((sigma.powi(2) + c * (2.0 - y).gamma() * (m.powf(y - 2.0) + g.powf(y - 2.0))) * maturity)
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    //use cf_dist_utils::*;
    #[test]
    fn cgmy_expectation() {
        let t = 2.0;
        let num_steps = 1024;
        let x_min = -10.0;
        let x_max = 10.0;
        let c = 0.5;
        let g = 2.0;
        let y = 1.2;
        let cf = |u: &Complex<f64>| cgmy_log_cf(&u, c, g, 0.0, y).exp();
        let discrete_cf = fang_oost::get_discrete_cf(num_steps, x_min, x_max, &cf);
        let approx_expectation =
            cf_dist_utils::get_expectation_discrete_cf(x_min, x_max, &discrete_cf);
        let expectation = cgmy_negative_jump_expectation(c, g, y);
        assert_abs_diff_eq!(expectation, approx_expectation, epsilon = 0.00001);
    }
}
