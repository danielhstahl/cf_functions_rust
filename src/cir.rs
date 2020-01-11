use num_complex::Complex;
/// Returns log of moment generating function for Cox Ingersoll Ross process evaluated at complex argument.
///
/// # Remarks
/// Useful for time changed levy processes.  "psi" can be a characteristic function of a levy process
/// evaluated at a given "u".
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let a = 0.3; //speed of mean reversion of CIR process
/// let kappa = 0.2; //kappa/a is the long run mean of CIR process
/// let sigma = 0.3; //volatility of CIR process
/// let t = 0.5; //time period of CIR process
/// let v0 = 0.7; //initial value of CIR process
/// let log_mgf = cf_functions::cir::cir_log_mgf(
///     &u, a, kappa, sigma, t, v0
/// );
/// # }
/// ```
pub fn cir_log_mgf(
    psi: &Complex<f64>,
    a: f64,
    kappa: f64,
    sigma: f64,
    t: f64,
    v0: f64,
) -> Complex<f64> {
    if crate::utils::is_same(kappa, 0.0) && crate::utils::is_same(sigma, 0.0) {
        return -psi * t;
    }
    let delta = (kappa.powi(2) + 2.0 * psi * sigma.powi(2)).sqrt();
    let exp_t = (-delta * t).exp();
    let delta_minus_kappa = delta - kappa;
    let b_t = 2.0 * psi * (1.0 - exp_t) / (delta + kappa + delta_minus_kappa * exp_t);
    let c_t = if sigma > 0.0 {
        (a / sigma.powi(2))
            * (2.0 * (1.0 - delta_minus_kappa * (1.0 - exp_t) / (2.0 * delta)).ln()
                + delta_minus_kappa * t)
    } else {
        psi * (t - (1.0 - exp_t) / kappa)
    };
    -b_t * v0 - c_t
}

/// Returns log of moment generating function for Cox Ingersoll Ross process
/// evaluated at complex argument and with complex kappa.
///
/// # Remarks
/// Useful for time changed levy processes.  "psi" can be a characteristic function of a levy
/// process evaluated at a given "u" with induced correlation used by "kappa".
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let a = 0.3; //speed of mean reversion of CIR process
/// let kappa = Complex::new(0.2, -0.3); //for leverage neutral measure
/// let sigma = 0.3; //volatility of CIR process
/// let t = 0.5; //time period of CIR process
/// let v0 = 0.7; //initial value of CIR process
/// let log_mgf = cf_functions::cir::cir_log_mgf_cmp(
///     &u, a, &kappa, sigma, t, v0
/// );
/// # }
/// ```
pub fn cir_log_mgf_cmp(
    psi: &Complex<f64>,
    a: f64,
    kappa: &Complex<f64>,
    sigma: f64,
    t: f64,
    v0: f64,
) -> Complex<f64> {
    if crate::utils::is_same_cmp(kappa, 0.0) && crate::utils::is_same(sigma, 0.0) {
        return -psi * t;
    }
    let delta = (kappa * kappa + 2.0 * psi * sigma.powi(2)).sqrt();
    let exp_t = (-delta * t).exp();
    let delta_minus_kappa = delta - kappa;
    let b_t = 2.0 * psi * (1.0 - exp_t) / (delta + kappa + delta_minus_kappa * exp_t);
    let c_t = if sigma > 0.0 {
        (a / sigma.powi(2))
            * (2.0 * (1.0 - delta_minus_kappa * (1.0 - exp_t) / (2.0 * delta)).ln()
                + delta_minus_kappa * t)
    } else {
        psi * (t - (1.0 - exp_t) / kappa)
    };
    -b_t * v0 - c_t
}
/// Returns moment generating function for Cox Ingersoll Ross process evaluated at complex argument.
///
/// # Remarks
/// Useful for time changed levy processes.  "psi" can be a characteristic function of a levy process
/// evaluated at a given "u".
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let a = 0.3; //speed of mean reversion of CIR process
/// let kappa = 0.2; //kappa/a is the long run mean of CIR process
/// let sigma = 0.3; //volatility of CIR process
/// let t = 0.5; //time period of CIR process
/// let v0 = 0.7; //initial value of CIR process
/// let mgf = cf_functions::cir::cir_mgf(
///     &u, a, kappa, sigma, t, v0
/// );
/// # }
/// ```
pub fn cir_mgf(
    psi: &Complex<f64>,
    a: f64,
    kappa: f64,
    sigma: f64,
    t: f64,
    v0: f64,
) -> Complex<f64> {
    cir_log_mgf(psi, a, kappa, sigma, t, v0).exp()
}
/// Returns moment generating function for Cox Ingersoll Ross process
/// evaluated at complex argument and with complex kappa.
///
/// # Remarks
/// Useful for time changed levy processes.  "psi" can be a characteristic function of a levy
/// process evaluated at a given "u" with induced correlation used by "kappa".
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_functions;
/// # fn main() {
/// let u = Complex::new(1.0, 1.0);
/// let a = 0.3; //speed of mean reversion of CIR process
/// let kappa = Complex::new(0.2, -0.3); //for leverage neutral measure
/// let sigma = 0.3; //volatility of CIR process
/// let t = 0.5; //time period of CIR process
/// let v0 = 0.7; //initial value of CIR process
/// let mgf = cf_functions::cir::cir_mgf_cmp(
///     &u, a, &kappa, sigma, t, v0
/// );
/// # }
/// ```
pub fn cir_mgf_cmp(
    psi: &Complex<f64>,
    a: f64,
    kappa: &Complex<f64>,
    sigma: f64,
    t: f64,
    v0: f64,
) -> Complex<f64> {
    cir_log_mgf_cmp(psi, a, kappa, sigma, t, v0).exp()
}

pub fn generic_leverage_diffusion(
    u: &Complex<f64>,
    get_cf: &dyn Fn(&Complex<f64>) -> Complex<f64>,
    t: f64,
    sigma: f64,
    v0: f64,
    speed: f64,
    eta_v: f64,
    rho: f64,
) -> Complex<f64> {
    //implies that long run mean is one
    let ln_m = speed - eta_v * rho * u * sigma;
    let cf_fn_rn = -get_cf(u);
    cir_log_mgf_cmp(&cf_fn_rn, speed, &ln_m, eta_v, t, v0)
}

//needed to solve ODE for duffie MGF
fn runge_kutta_complex_vector(
    fx: &dyn Fn(f64, &Complex<f64>, &Complex<f64>) -> (Complex<f64>, Complex<f64>),
    mut init_value_1: Complex<f64>,
    mut init_value_2: Complex<f64>,
    t: f64,
    num_steps: usize,
) -> (Complex<f64>, Complex<f64>) {
    let dt = t / (num_steps as f64);
    let hfdt = dt * 0.5;
    let sixthdt = dt / 6.0;
    for index in 0..num_steps {
        let t_curr = dt * (index as f64);
        let (k11, k12) = fx(t_curr, &init_value_1, &init_value_2);
        let (k21, k22) = fx(
            t_curr + hfdt,
            &(init_value_1 + k11 * hfdt),
            &(init_value_2 + k12 * hfdt),
        );
        let (k31, k32) = fx(
            t_curr + hfdt,
            &(init_value_1 + k21 * hfdt),
            &(init_value_2 + k22 * hfdt),
        );
        let (k41, k42) = fx(
            t_curr + dt,
            &(init_value_1 + k21 * dt),
            &(init_value_2 + k22 * dt),
        );
        init_value_1 = init_value_1 + (k11 + 2.0 * k21 + 2.0 * k31 + k41) * sixthdt;
        init_value_2 = init_value_2 + (k12 + 2.0 * k22 + 2.0 * k32 + k42) * sixthdt;
    }
    (init_value_1, init_value_2)
}

//helper for ODE, http://web.stanford.edu/~duffie/dps.pdf
//since with respect to T-t, this is the opposite sign as the paper
fn alpha_or_beta(
    rho: f64,
    k: f64,
    h: f64,
    l: f64,
) -> impl (Fn(&Complex<f64>, &Complex<f64>) -> Complex<f64>) {
    move |ode_val: &Complex<f64>, cf_val: &Complex<f64>| {
        -rho + k * ode_val + 0.5 * ode_val * ode_val * h + l * cf_val
    }
}

fn duffie_mgf_increment(
    u: &Complex<f64>,
    ode_val_2: &Complex<f64>,
    rho0: f64,
    rho1: f64,
    k0: f64,
    k1: f64,
    h0: f64,
    h1: f64,
    l0: f64,
    l1: f64,
    cf: &dyn Fn(&Complex<f64>) -> Complex<f64>,
) -> (Complex<f64>, Complex<f64>) {
    let cf_part = cf(u) - 1.0;
    let beta = alpha_or_beta(rho1, k1, h1, l1);
    let alpha = alpha_or_beta(rho0, k0, h0, l0);
    (alpha(ode_val_2, &cf_part), beta(ode_val_2, &cf_part))
}

//jump leverage...http://web.stanford.edu/~duffie/dps.pdf page 10
pub fn generic_leverage_jump(
    u: &Complex<f64>,
    cf: &dyn Fn(&Complex<f64>) -> Complex<f64>,
    t: f64,
    v0: f64,
    correlation: f64,
    expected_value_of_cf: f64,
    rho0: f64,
    rho1: f64,
    k0: f64,
    k1: f64,
    h0: f64,
    h1: f64,
    l0: f64,
    l1: f64,
    num_steps: usize,
) -> Complex<f64> {
    let init_value_1 = Complex::new(0.0, 0.0);
    let init_value_2 = Complex::new(0.0, 0.0);
    let delta = if l1 > 0.0 && expected_value_of_cf > 0.0 {
        correlation / (expected_value_of_cf * l1)
    } else {
        0.0
    };
    let fx = move |_t: f64, _curr_val_1: &Complex<f64>, curr_val_2: &Complex<f64>| {
        duffie_mgf_increment(
            &(u + delta * curr_val_2),
            curr_val_2,
            rho0,
            rho1,
            k0,
            k1,
            h0,
            h1,
            l0,
            l1,
            cf,
        )
    };
    let (alpha, beta) = runge_kutta_complex_vector(&fx, init_value_1, init_value_2, t, num_steps);
    beta * v0 + alpha
}

//From page 8 and 9 of my ops risk paper
//https://github.com/phillyfan1138/OpsRiskPaper/blob/master/OpsRiskForRiskNet.pdf
//The expectation is E[e^{lambda*(1-E[e^uiL])\int v_s ds}]
//Using the duffie ODE formula, rho0=0, rho1=lambda*(1-E[e^uiL]),
//k0=a, k1=-a*kappahat (where kappahat=1+correlation/a
//and and correlation=delta*E[L]*lambda), h0=0,
//h1=sigma*sigma, l0=0, l1=lambda.  However, this can be
//simplified so that rho1=0 by adjusting the cf of the
//jump to have (u-i*delta*beta) instead of just u.
//See equation 8 in my ops risk paper.
pub fn cir_leverage_jump(
    u: &Complex<f64>,
    cf: &dyn Fn(&Complex<f64>) -> Complex<f64>,
    t: f64,
    v0: f64,
    correlation: f64,
    expected_value_of_cf: f64,
    a: f64,
    sigma: f64,
    lambda: f64,
    num_steps: usize,
) -> Complex<f64> {
    let kappa = 1.0 + correlation / a; //to stay expectation of 1
    generic_leverage_jump(
        u,
        cf,
        t,
        v0,
        correlation,
        expected_value_of_cf,
        0.0,
        0.0,
        a,
        -a * kappa,
        0.0,
        sigma.powi(2),
        0.0,
        lambda,
        num_steps,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    #[test]
    fn runge_kutta() {
        let t = 2.0;
        let num_steps = 1024;
        let init_value_1 = Complex::new(1.0, 0.0);
        let init_value_2 = Complex::new(1.0, 0.0);
        let (res1, res2) = runge_kutta_complex_vector(
            &|t: f64, val1: &Complex<f64>, val2: &Complex<f64>| (val1 * t, val2 * t),
            init_value_1,
            init_value_2,
            t,
            num_steps,
        );
        assert_abs_diff_eq!(res1.re, (2.0 as f64).exp(), epsilon = 0.00001);
        assert_abs_diff_eq!(res2.re, (2.0 as f64).exp(), epsilon = 0.00001);
    }
    #[test]
    fn duffie_mgf_compare_cir() {
        let sigma = 0.3;
        let a = 0.3;
        let b = 0.05;
        let r0 = 0.05;
        let h = (a * a + 2.0 * sigma * sigma as f64).sqrt();
        let t = 0.25;
        let a_num = 2.0 * h * ((a + h) * t * 0.5).exp();
        let a_den = 2.0 * h + (a + h) * ((t * h).exp() - 1.0);
        let a_t_tm = (a_num / a_den).powf(2.0 * a * b / (sigma * sigma));
        let b_num = 2.0 * ((t * h).exp() - 1.0);
        let b_den = a_den;
        let b_t_tm = b_num / b_den;
        let bond_price = a_t_tm * ((-r0 * b_t_tm).exp());
        let rho0 = 0.0;
        let rho1 = 1.0;
        let k0 = a * b;
        let k1 = -a;
        let h0 = 0.0;
        let h1 = sigma * sigma;
        let l0 = 0.0;
        let l1 = 0.0;
        let num_steps: usize = 1024;
        let cf = |u: &Complex<f64>| u.exp();
        let correlation = 0.0;
        let expected_value_of_cf = 1.0; //doesnt matter
        let u = Complex::new(1.0, 0.0);
        let result = generic_leverage_jump(
            &u,
            &cf,
            t,
            r0,
            correlation,
            expected_value_of_cf,
            rho0,
            rho1,
            k0,
            k1,
            h0,
            h1,
            l0,
            l1,
            num_steps,
        );
        assert_abs_diff_eq!(result.re.exp(), bond_price, epsilon = 0.000001);
    }
    #[test]
    fn cir_analytical() {
        let sigma = 0.3;
        let a = 0.3;
        let b = 0.05;
        let r0 = 0.05;
        let h = (a * a + 2.0 * sigma * sigma as f64).sqrt();
        let t = 0.25;
        let a_num = 2.0 * h * ((a + h) * t * 0.5).exp();
        let a_den = 2.0 * h + (a + h) * ((t * h).exp() - 1.0);
        let a_t_tm = (a_num / a_den).powf(2.0 * a * b / (sigma * sigma));
        let b_num = 2.0 * ((t * h).exp() - 1.0);
        let b_den = a_den;
        let b_t_tm = b_num / b_den;
        let bond_price = a_t_tm * ((-r0 * b_t_tm).exp());
        assert_eq!(
            bond_price,
            cir_mgf(&Complex::new(1.0, 0.0), a * b, a, sigma, t, r0).re
        );
    }
    #[test]
    fn cir_with_zeros() {
        let t = 1.0;
        let r0 = 0.04;
        let approx_bond_price = cir_mgf(&Complex::new(1.0, 0.0), 0.0, 0.0, 0.0, t, r0).re;
        assert_eq!(approx_bond_price.is_nan(), false);
    }
    #[test]
    fn cir_heston() {
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
        let approx_heston_cf = cir_mgf_cmp(&neg_psi, k, &k_star, sig, t, v0).re;
        assert_eq!(cf_heston, approx_heston_cf);
    }
}
