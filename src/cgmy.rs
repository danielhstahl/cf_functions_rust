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
        c * (1.0 - u / g).ln() * (1.0 + u / m).ln()
    } else {
        c * (-y).gamma() * ((m - u).powf(y) + (g + u).powf(y) - m.powf(y) - g.powf(y))
    }
}

fn cgmy_log_cf_lower_side(u: &Complex<f64>, c: f64, m: f64, y: f64) -> Complex<f64> {
    if crate::utils::is_same(y, 1.0) {
        Complex::new(0.0, 0.0)
    } else if crate::utils::is_same(y, 0.0) {
        c * (1.0 + u / m).ln()
    } else {
        c * (-y).gamma() * ((m - u).powf(y) - m.powf(y))
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
// that paper defines khat*(theta-v), I define k*(1-v)+v*E[dJ]*eta_v
// to make the long run expectation equal to 1.
//solving, khat=k-E[dJ]*eta_v
//khat*theta=k
//so theta=k/khat
//try to depricate
fn depreciate_leverage_neutral_pure_jump_log_cf(
    u: &Complex<f64>,
    cf_negative: &dyn Fn(&Complex<f64>) -> Complex<f64>, //
    cf: &dyn Fn(&Complex<f64>) -> Complex<f64>,
    expected_value_jump: f64, //negative only
    speed: f64,               //"k"
    eta_v: f64,
    sigma: f64,
    v0: f64,
    t: f64,
    num_steps: usize,
) -> Complex<f64> {
    let init_value_1 = Complex::new(0.0, 0.0);
    let init_value_2 = Complex::new(0.0, 0.0);
    let khat = speed - eta_v * expected_value_jump; //get expectation equal to one

    let theta = if crate::utils::is_same(khat, 0.0) {
        0.0
    } else {
        speed / khat
    };
    println!("this is sigma in dlnpjlcf {}", sigma);
    println!("this is eta_v in dlnpjlcf {}", eta_v);
    println!("this is khat in dlnpjlcf {}", khat);
    println!("this is khat*theta in dlnpjlcf {}", khat * theta);
    //println!("this is h1 in dlnpjlcf {}", h1);
    let fx = move |_t: f64, _alpha_prev: &Complex<f64>, beta_prev: &Complex<f64>| {
        let u_sig = sigma * u;
        let u_extended = beta_prev * Complex::new(0.0, 1.0) * eta_v + u_sig;
        let beta = cf_negative(&u_extended) - cf_negative(&u_sig) - khat * beta_prev + cf(u);
        let alpha = khat * theta * beta_prev;
        (alpha, beta)
    };
    let (alpha, beta) =
        crate::cir::runge_kutta_complex_vector(&fx, init_value_1, init_value_2, t, num_steps);
    beta * v0 + alpha
}

fn leverage_neutral_pure_jump_log_cf(
    u: &Complex<f64>,
    cf_negative: &dyn Fn(&Complex<f64>) -> Complex<f64>, //
    cf: &dyn Fn(&Complex<f64>) -> Complex<f64>,
    expected_value_jump: f64, //negative only
    speed: f64,               //"k"
    eta_v: f64,
    sigma: f64,
    v0: f64,
    t: f64,
    num_steps: usize,
) -> Complex<f64> {
    let khat = speed - eta_v * expected_value_jump; //get expectation equal to one

    let theta = if crate::utils::is_same(khat, 0.0) {
        0.0
    } else {
        speed / khat
    };
    crate::cir::leverage_neutral_generic(
        &u,
        &cf_negative,
        &cf,
        0.0,
        0.0,
        khat * theta,
        -khat,
        v0,
        0.0,
        sigma,
        0.0,
        0.0,
        eta_v,
        t,
        num_steps,
    )
}

/// /// Returns time changed (self-exciting) CGMY characteristic function with
/// correlation between the pure-jump time changed process and the underlying.
///
/// # Remarks
/// The pure-jump time change is assumed to be an affine mean-reverting process
/// with long run mean of 1.0.  For more information, see https://poseidon01.ssrn.com/delivery.php?ID=737027111000006077113070089110095064016020050037028066000080065074127006086092092026061120060015055036110006010126103066122080108059078076004070004065091125021108014077028121011029092117112080127092065007111098070065099086069122086067104098093017117&EXT=pdf&INDEX=TRUE
/// page 11.  
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
///     sigma, v0, speed, eta_v, 128
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
    let expected_value_jump = cgmy_expectation(c, 0.0, m, y) * eta_v;
    move |u: &Complex<f64>| {
        (leverage_neutral_pure_jump_log_cf(
            &u,
            &|u| cgmy_log_cf_lower_side(&u, c, m, y),
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

/// Returns log of time changed CGMY characteristic function with correlation between the
/// diffusion of the time changed process and the underlying.
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

//http://www.math.chalmers.se/~palbin/YongqiangBu.pdf page 15
pub fn cgmy_expectation(c: f64, g: f64, m: f64, y: f64) -> f64 {
    if crate::utils::is_same(y, 1.0) {
        0.0
    } else if crate::utils::is_same(y, 0.0) {
        0.0
    } else {
        c * (1.0 - y).gamma() * (m.powf(y - 1.0) - g.powf(y - 1.0))
    }
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
    fn test_exp_cf(u: &Complex<f64>, lambda: f64) -> Complex<f64> {
        lambda / (lambda - u)
    }
    #[test]
    fn cgmy_expectation_test() {
        let num_steps = 1024;
        let x_min = -10.0;
        let x_max = 10.0;
        let c = 0.5;
        let m = 2.0;
        let g = 3.0; //g is larger than m, so distribution is right skewed; expectation is positive
        let y = 1.2;
        let cf = |u: &Complex<f64>| cgmy_log_cf(&u, c, g, m, y).exp();
        let discrete_cf = fang_oost::get_discrete_cf(num_steps, x_min, x_max, &cf);
        let approx_expectation = //0.2824205454935122
            cf_dist_utils::get_expectation_discrete_cf(x_min, x_max, &discrete_cf);
        let expectation = cgmy_expectation(c, g, m, y); //
        assert_abs_diff_eq!(expectation, approx_expectation, epsilon = 0.00001);
    }
    #[test]
    fn cgmy_partial_expectation_test() {
        let num_steps = 1024;
        let x_min = -10.0;
        let x_max = 10.0;
        let c = 0.5;
        let m = 2.0;
        let g = 0.0;
        let y = 1.2;
        let cf = |u: &Complex<f64>| cgmy_log_cf_lower_side(&u, c, m, y).exp();
        let discrete_cf = fang_oost::get_discrete_cf(num_steps, x_min, x_max, &cf);

        let approx_expectation =
            cf_dist_utils::get_expectation_discrete_cf(x_min, x_max, &discrete_cf);

        let expectation = cgmy_expectation(c, g, m, y);
        assert_abs_diff_eq!(expectation, approx_expectation, epsilon = 0.00001);
    }
    //these tests are to demonstrate, on an easy distribution,
    //how the correct expectation can be obtained even when
    //using domains over which the distribution is not
    //defined
    #[test]
    fn test_exp_normal() {
        let lambda = 2.0;
        let num_steps = 1024;
        let x_min = -20.0;
        let x_max = 20.0;
        let cf = |u: &Complex<f64>| test_exp_cf(&u, lambda);
        let discrete_cf = fang_oost::get_discrete_cf(num_steps, x_min, x_max, &cf);
        let approx_expectation =
            cf_dist_utils::get_expectation_discrete_cf(x_min, x_max, &discrete_cf);
        assert_abs_diff_eq!(0.5, approx_expectation, epsilon = 0.00001);
    }
    //these tests are to demonstrate, on an easy distribution,
    //how the correct expectation can be obtained even when
    //using domains over which the distribution is not
    //defined
    #[test]
    fn test_exp_negative() {
        let lambda = 2.0;
        let num_steps = 1024;
        let x_min = -20.0;
        let x_max = 20.0;
        let cf = |u: &Complex<f64>| test_exp_cf(&(-u), lambda);
        let discrete_cf = fang_oost::get_discrete_cf(num_steps, x_min, x_max, &cf);
        let approx_expectation =
            cf_dist_utils::get_expectation_discrete_cf(x_min, x_max, &discrete_cf);
        assert_abs_diff_eq!(-0.5, approx_expectation, epsilon = 0.00001);
    }
    #[test]
    fn cgmyse_option_price_test_special_case() {
        let sigma = 1.0; //to be able to compare apples to apples
        let c = 1.0;
        let g = 5.0;
        let m = 5.0;
        let y = 1.5;
        let speed = 0.0;
        let v0 = 1.0;
        let eta_v = 0.0;
        let strikes = vec![100.0];
        let num_u: usize = 256;
        let num_steps: usize = 256;
        let t = 1.0;
        let rate = 0.1;
        let asset = 100.0;
        let vol = cgmy_diffusion_vol(0.0, c, g, m, y, t);
        let max_strike = (10.0 * vol).exp() * asset;
        let cf_inst_se =
            cgmyse_time_change_cf(t, rate, c, g, m, y, sigma, v0, speed, eta_v, num_steps);
        let cf_inst = cgmy_time_change_cf(t, rate, c, g, m, y, 0.0, v0, 0.0, 0.0, 0.0);
        let results_se = fang_oost_option::option_pricing::fang_oost_call_price(
            num_u,
            asset,
            &strikes,
            max_strike,
            rate,
            t,
            &cf_inst_se,
        );
        let results = fang_oost_option::option_pricing::fang_oost_call_price(
            num_u, asset, &strikes, max_strike, rate, t, &cf_inst,
        );
        assert_abs_diff_eq!(results_se[0], results[0], epsilon = 0.00001);
    }
    #[test]
    fn cgmyse_option_price_test() {
        let sigma = 1.0; //to be able to compare apples to apples
        let c = 1.0;
        let g = 5.0;
        let m = 5.0;
        let y = 1.5;
        let speed = 0.3;
        let v0 = 1.0;
        let eta_v = 0.1;
        let strikes = vec![100.0];
        let num_u: usize = 256;
        let num_steps: usize = 256;
        let t = 1.0;
        let rate = 0.1;
        let asset = 100.0;
        let vol = cgmy_diffusion_vol(0.0, c, g, m, y, t);
        let max_strike = (10.0 * vol).exp() * asset;
        let cf_inst_se =
            cgmyse_time_change_cf(t, rate, c, g, m, y, sigma, v0, speed, eta_v, num_steps);
        let cf_inst = cgmy_time_change_cf(t, rate, c, g, m, y, 0.0, v0, 0.0, 0.0, 0.0);
        let results_se = fang_oost_option::option_pricing::fang_oost_call_price(
            num_u,
            asset,
            &strikes,
            max_strike,
            rate,
            t,
            &cf_inst_se,
        );
        let results = fang_oost_option::option_pricing::fang_oost_call_price(
            num_u, asset, &strikes, max_strike, rate, t, &cf_inst,
        );
        println!("this is results {}", results_se[0]);
        //negative correlation leads to lower call prices
        assert_eq!(results_se[0] < results[0], true);
    }
    #[test]
    fn cgmyse_option_price_expectation() {
        let sigma = 1.0; //to be able to compare apples to apples
        let c = 1.0;
        let g = 5.0;
        let m = 5.0;
        let y = 1.5;
        let speed = 0.3;
        let v0 = 1.0;
        let eta_v = 0.1;
        let num_u: usize = 256;
        let num_steps: usize = 256;
        let t = 1.0;
        let rate = 0.1;
        //let asset = 100.0;
        let vol = cgmy_diffusion_vol(0.0, c, g, m, y, t);
        let max_x = (10.0 * vol).exp();
        let cf_inst_se =
            cgmyse_time_change_cf(t, rate, c, g, m, y, sigma, v0, speed, eta_v, num_steps);
        let discrete_cf = fang_oost::get_discrete_cf(num_u, 0.0, max_x, &cf_inst_se);
        let expectation = cf_dist_utils::get_expectation_discrete_cf(0.0, max_x, &discrete_cf);
        assert_abs_diff_eq!(expectation, (rate * t).exp(), epsilon = 0.00001);
    }
    #[test]
    fn cgmy_option_price_test() {
        //https://mpra.ub.uni-muenchen.de/8914/4/MPRA_paper_8914.pdf pg 18
        //S0 = 100, K = 100, r = 0.1, q = 0, C = 1, G = 5, M = 5, T = 1, Y=1.98
        let sigma = 0.0;
        let c = 1.0;
        let g = 5.0;
        let m = 5.0;
        let y = 1.5;
        let speed = 0.0;
        let v0 = 1.0;
        let eta_v = 0.0;
        let rho = 0.0;
        let strikes = vec![100.0];
        let num_u: usize = 256;
        let t = 1.0;
        let rate = 0.1;
        let asset = 100.0;
        let vol = cgmy_diffusion_vol(sigma, c, g, m, y, t);
        let max_strike = (10.0 * vol).exp() * asset;
        let cf_inst = cgmy_time_change_cf(t, rate, c, g, m, y, sigma, v0, speed, eta_v, rho);
        let results = fang_oost_option::option_pricing::fang_oost_call_price(
            num_u, asset, &strikes, max_strike, rate, t, &cf_inst,
        );
        assert_abs_diff_eq!(results[0], 49.790905469, epsilon = 0.00001);
    }
}
