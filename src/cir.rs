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
pub fn runge_kutta_complex_vector(
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
pub fn alpha_or_beta(
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
    //note I can only use this because the self-exciting time change still returns a (complex valued) poisson distribution
    //else I would need cf(u+beta*delta*i)-cf(u)
    let cf_part = cf(u) - 1.0;
    let beta = alpha_or_beta(rho1, k1, h1, l1);
    let alpha = alpha_or_beta(rho0, k0, h0, l0);
    (alpha(ode_val_2, &cf_part), beta(ode_val_2, &cf_part))
}

//jump leverage...http://web.stanford.edu/~duffie/dps.pdf page 10
//try to depricate
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

//http://web.stanford.edu/~duffie/dps.pdf page 10
pub fn leverage_neutral_generic(
    u: &Complex<f64>,
    cf_jump: &impl Fn(&Complex<f64>) -> Complex<f64>, //
    cf: &impl Fn(&Complex<f64>) -> Complex<f64>,
    //expected_value_jump: f64,
    r0: f64, //defined in Duffie et al
    r1: f64,
    k0: f64,
    k1: f64,
    //h1: f64, //
    //l1: f64, incorporated in CF
    v0: f64,
    sigma0: f64, //constant multiplying BM part of asset,
    sigma1: f64, //constant multiplying pure jump part of asset
    rho: f64,    //correlation for BM part
    eta0: f64,   //constant multiplying BM part of time-change process, the square of this is h1
    eta1: f64,   //constant multiplying pure jump part of time-change process

    t: f64,
    num_steps: usize,
) -> Complex<f64> {
    let init_value_1 = Complex::new(0.0, 0.0);
    let init_value_2 = Complex::new(0.0, 0.0);
    let h1 = eta0.powi(2);
    /*println!("this is sigma0 in lng {}", sigma0);
    println!("this is sigma1 in lng {}", sigma1);
    println!("this is k0 in lng {}", k0);
    println!("this is k1 in lng {}", k1);
    println!("this is h1 in lng {}", h1);
    println!("this is eta1 in lng {}", eta1);*/
    let fx = move |_t: f64, _alpha_prev: &Complex<f64>, beta_prev: &Complex<f64>| {
        let u_sig = sigma1 * u;
        let u_extended = beta_prev * eta1 + u_sig;
        let k_extended = k1 + u * eta0 * rho * sigma0; //note that k1 is typically negative
        let beta = cf_jump(&u_extended) - cf_jump(&u_sig) + k_extended * beta_prev - r1
            + cf(u)
            + beta_prev * beta_prev * h1 * 0.5;
        let alpha = k0 * beta_prev - r0;
        (alpha, beta)
    };
    let (alpha, beta) =
        crate::cir::runge_kutta_complex_vector(&fx, init_value_1, init_value_2, t, num_steps);
    beta * v0 + alpha
}

//From page 8 and 9 of my ops risk paper
//https://github.com/danielhstahl/OpsRiskPaper/releases/download/0.1.0/main.pdf
//The expectation is E[e^{lambda*(E[e^uiL]-1)\int v_s ds}]
//Using the duffie ODE formula, rho0=0, rho1=lambda*(1-E[e^uiL]),
//k0=a, k1=-a*kappahat (where kappahat=1+correlation/a
//and and correlation=delta*E[L]*lambda), h0=0,
//h1=sigma*sigma, l0=0, l1=lambda.  However, this can be
//simplified so that rho1=0 by adjusting the cf of the
//jump to have (u-i*delta*beta) instead of just u.
//See equation 8 in my ops risk paper.
pub fn cir_leverage_jump(
    u: &Complex<f64>,
    cf_jump: &impl Fn(&Complex<f64>) -> Complex<f64>,
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
    let delta = correlation / (lambda * expected_value_of_cf);
    let full_cf = |u: &Complex<f64>| lambda * (cf_jump(u) - 1.0);
    let jump_cf_extended = |u: &Complex<f64>| lambda * cf_jump(u);
    leverage_neutral_generic(
        u,
        &full_cf,
        &jump_cf_extended,
        0.0,
        0.0,
        a,
        -a * kappa,
        v0,
        0.0,
        1.0,
        0.0,
        sigma,
        delta,
        t,
        num_steps,
    )

    /*
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
    )*/
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
    fn duffie_mgf_compare_cir_new() {
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
        let cf = |u: &Complex<f64>| Complex::new(0.0, 0.0);
        let correlation = 0.0;
        let expected_value_of_cf = 1.0; //doesnt matter
        let u = Complex::new(1.0, 0.0);
        let result = leverage_neutral_generic(
            &u, &cf, &cf, rho0, rho1, k0, k1, r0, 0.0, 0.0, 0.0, sigma, 0.0, t, num_steps,
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

    #[test]
    fn leverage_jump() {
        let u = Complex::new(0.5, 0.5);
        let example_lambda = 0.5; //for jump distribution
        let lambda = 0.6; //lambda for poisson
        let cfjump = |u: &Complex<f64>| lambda * (example_lambda / (example_lambda - u));
        let t = 0.5;
        let v0 = 1.2;
        let correlation = 0.5;
        let expected_value_of_cf = 1.0 / example_lambda;
        let a = 0.4;
        let sigma = 0.3;
        let num_steps: usize = 256;
        let kappa = 1.0 + correlation / a; //to stay expectation of 1
        let delta = correlation / (lambda * expected_value_of_cf);
        let cf = |u: &Complex<f64>| cfjump(u) - lambda;
        let result = leverage_neutral_generic(
            &u,
            &cfjump,
            &cf,
            0.0,
            0.0,
            a,
            -a * kappa,
            v0,
            0.0,
            1.0,
            0.0,
            sigma,
            delta,
            t,
            num_steps,
        ); /*
           let result = cir_leverage_jump(
               &u,
               &cf,
               t,
               v0,
               correlation,
               expected_value_of_cf,
               a,
               sigma,
               lambda,
               num_steps,
           );*/

        // let cf1 = |u| example_lambda / (example_lambda - u);
        let resultprev = generic_leverage_jump(
            &u,
            &cf,
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
        );
        assert_eq!(result.re, resultprev.re);
        assert_eq!(result.im, resultprev.im);
    }
    #[test]
    fn test_heston_full_generic() {
        //https://mpra.ub.uni-muenchen.de/8914/4/MPRA_paper_8914.pdf pg 15
        let b: f64 = 0.0398;
        let a = 1.5768;
        let c = 0.5751;
        let rho = -0.5711;
        let v0 = 0.0175;

        let sigma = b.sqrt();
        let speed = a;
        let eta_v = c;
        let strikes = vec![100.0];
        let num_u: usize = 256;
        let t = 1.0;
        let rate = 0.0;
        let asset = 100.0;
        let max_strike = (10.0 * sigma).exp() * asset;
        let num_steps: usize = 256;
        let cf_inst = |u: &Complex<f64>| {
            (leverage_neutral_generic(
                &u,
                &|_u: &Complex<f64>| Complex::new(0.0, 0.0),
                &|u| crate::gauss::gauss_log_cf(u, -0.5 * sigma * sigma, sigma),
                0.0,
                0.0,
                speed,
                -speed,
                v0 / b,
                sigma,
                0.0,
                rho,
                eta_v / sigma,
                0.0,
                t,
                num_steps,
            ) + rate * u * t)
                .exp()
        };

        let results = fang_oost_option::option_pricing::fang_oost_call_price(
            num_u, asset, &strikes, max_strike, rate, t, &cf_inst,
        );
        assert_abs_diff_eq!(results[0], 5.78515545, epsilon = 0.0001);
    }
}
