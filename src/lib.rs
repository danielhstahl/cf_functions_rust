//! Provides several common characteristic functions for
//! option pricing.  All of the characteristic functions
//! are with respect to "ui" instead of "u".  
extern crate num_complex;

use num_complex::Complex;

use std::f64::consts::PI;

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
/// let log_cf = cf_functions::gauss_log_cf(
///     &u, mu, sigma
/// );
/// # }
/// ```
pub fn gauss_log_cf(
    u:&Complex<f64>,
    mu:f64,
    sigma:f64
)->Complex<f64>
{
    u*mu+u*u*0.5*sigma.powi(2)
}

fn gauss_log_cf_cmp(
    u:&Complex<f64>,
    mu:&Complex<f64>,
    sigma:f64
)->Complex<f64>
{
    u*mu+u*u*0.5*sigma.powi(2)
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
/// let cf = cf_functions::gauss_cf(
///     &u, mu, sigma
/// );
/// # }
/// ```
pub fn gauss_cf(
    u:&Complex<f64>,
    mu:f64,
    sigma:f64
)->Complex<f64>
{
    gauss_log_cf(u, mu, sigma).exp()
}

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
/// let log_cf = cf_functions::merton_log_cf(
///     &u, lambda, mu_l, sigma_l
/// );
/// # }
/// ```
pub fn merton_log_cf(
    u:&Complex<f64>,
    lambda:f64,
    mu_l:f64,
    sig_l:f64
)->Complex<f64>
{
    lambda*(gauss_cf(u, mu_l, sig_l)-1.0)
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
/// let log_cf = cf_functions::merton_log_risk_neutral_cf(
///     &u, lambda, mu_l, sigma_l, rate, sigma
/// );
/// # }
/// ```
pub fn merton_log_risk_neutral_cf(
    u:&Complex<f64>,
    lambda:f64,
    mu_l:f64,
    sig_l:f64,
    rate:f64,
    sigma:f64
)->Complex<f64>{
    let cmp_mu=rate-0.5*sigma.powi(2)-merton_log_cf(&Complex::new(1.0, 0.0), lambda, mu_l, sig_l);
    gauss_log_cf_cmp(
        u, 
        &cmp_mu,
        sigma
    )+merton_log_cf(u, lambda, mu_l, sig_l)
}
fn is_same(
    num:f64,
    to_compare:f64
)->bool{
    (num-to_compare).abs()<=std::f64::EPSILON
}
fn is_same_cmp(
    num:&Complex<f64>,
    to_compare:f64
)->bool{
    (num.re-to_compare).abs()<=std::f64::EPSILON
}
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
/// let log_mgf = cf_functions::cir_log_mgf(
///     &u, a, kappa, sigma, t, v0
/// );
/// # }
/// ```
pub fn cir_log_mgf(
    psi:&Complex<f64>,
    a:f64,
    kappa:f64,
    sigma:f64,
    t:f64,
    v0:f64
)->Complex<f64>{
    if is_same(kappa, 0.0) && is_same(sigma, 0.0){
        return -psi*t;
    }
    let delta=(kappa.powi(2)+2.0*psi*sigma.powi(2)).sqrt();
    let exp_t=(-delta*t).exp();
    let delta_minus_kappa=delta-kappa;
    let b_t=2.0*psi*(1.0-exp_t)/(delta+kappa+delta_minus_kappa*exp_t);
    let c_t=if sigma>0.0 {
        (a/sigma.powi(2))*(2.0*(1.0-delta_minus_kappa*(1.0-exp_t)/(2.0*delta)).ln()+delta_minus_kappa*t)
    } else {
        psi*(t-(1.0-exp_t)/kappa)
    };
    -b_t*v0-c_t
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
/// let log_mgf = cf_functions::cir_log_mgf_cmp(
///     &u, a, &kappa, sigma, t, v0
/// );
/// # }
/// ```
pub fn cir_log_mgf_cmp(
    psi:&Complex<f64>,
    a:f64,
    kappa:&Complex<f64>,
    sigma:f64,
    t:f64,
    v0:f64
)->Complex<f64>{
    if is_same_cmp(kappa, 0.0) && is_same(sigma, 0.0){
        return -psi*t;
    }
    let delta=(kappa*kappa+2.0*psi*sigma.powi(2)).sqrt();
    let exp_t=(-delta*t).exp();
    let delta_minus_kappa=delta-kappa;
    let b_t=2.0*psi*(1.0-exp_t)/(delta+kappa+delta_minus_kappa*exp_t);
    let c_t=if sigma>0.0 {
        (a/sigma.powi(2))*(2.0*(1.0-delta_minus_kappa*(1.0-exp_t)/(2.0*delta)).ln()+delta_minus_kappa*t)
    } else {
        psi*(t-(1.0-exp_t)/kappa)
    };
    -b_t*v0-c_t
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
/// let mgf = cf_functions::cir_mgf(
///     &u, a, kappa, sigma, t, v0
/// );
/// # }
/// ```
pub fn cir_mgf(
    psi:&Complex<f64>,
    a:f64,
    kappa:f64,
    sigma:f64,
    t:f64,
    v0:f64
)->Complex<f64>{
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
/// let mgf = cf_functions::cir_mgf_cmp(
///     &u, a, &kappa, sigma, t, v0
/// );
/// # }
/// ```
pub fn cir_mgf_cmp(
    psi:&Complex<f64>,
    a:f64,
    kappa:&Complex<f64>,
    sigma:f64,
    t:f64,
    v0:f64
)->Complex<f64>{
    cir_log_mgf_cmp(psi, a, kappa, sigma, t, v0).exp()
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
/// let cf = cf_functions::stable_cf(
///     &u, alpha, mu, beta, c
/// );
/// # }
/// ```
pub fn stable_cf(
    u:&Complex<f64>,
    alpha:f64,
    mu:f64,
    beta:f64,
    c:f64
)->Complex<f64>{
    let phi=(alpha*0.5*PI).tan();
    (u*mu-(u*Complex::new(0.0, -1.0)*c).powf(alpha)*Complex::new(1.0, -beta*phi)).exp()
}

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
/// let cf = cf_functions::gamma_cf(
///     &u, a, b
/// );
/// # }
/// ```
pub fn gamma_cf(
    u:&Complex<f64>,
    a:f64,
    b:f64
)->Complex<f64>{
    (1.0-u*b).powf(-a)
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
/// let log_cf = cf_functions::merton_time_change_log_cf(
///     &u, t, lambda, mu_l, sigma_l, 
///     sigma, v0, speed, eta_v, rho
/// );
/// # }
/// ```
pub fn merton_time_change_log_cf(
    u:&Complex<f64>,
    t:f64,
    lambda:f64,
    mu_l:f64,
    sig_l:f64,
    sigma:f64,
    v0:f64,
    speed:f64,
    eta_v:f64,
    rho:f64    
)->Complex<f64>{
    let cf_rn=-merton_log_risk_neutral_cf(u, lambda, mu_l, sig_l, 0.0, sigma);
    let ln_m=speed-eta_v*rho*u*sigma;
    cir_log_mgf_cmp(
        &cf_rn, 
        speed,
        &ln_m,
        eta_v,
        t, 
        v0
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
/// let cf = cf_functions::merton_time_change_cf(
///     t, rate, lambda, mu_l, sigma_l, 
///     sigma, v0, speed, eta_v, rho
/// );
/// let value_of_cf=cf(&Complex::new(0.05, -0.5));
/// # }
/// ```
pub fn merton_time_change_cf(
    t:f64,
    rate:f64,
    lambda:f64,
    mu_l:f64,
    sig_l:f64,
    sigma:f64,
    v0:f64,
    speed:f64,
    ada_v:f64,
    rho:f64  
)->impl Fn(&Complex<f64>)->Complex<f64>
{
    move |u|(rate*t*u+merton_time_change_log_cf(
        u, t, lambda, mu_l, sig_l, 
        sigma, v0, speed, ada_v, rho)
    ).exp()
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cir_analytical() {
        let sigma=0.3;
        let a=0.3;
        let b=0.05;
        let r0=0.05;
        let h=(a*a+2.0*sigma*sigma as f64).sqrt();
        let t=0.25;
        let a_num=2.0*h*((a+h)*t*0.5).exp();
        let a_den=2.0*h+(a+h)*((t*h).exp()-1.0);
        let a_t_T=(a_num/a_den).powf(2.0*a*b/(sigma*sigma));
        let b_num=2.0*((t*h).exp()-1.0);
        let b_den=a_den;
        let b_t_T=b_num/b_den;
        let bond_price=a_t_T*((-r0*b_t_T).exp());
        assert_eq!(bond_price, cir_mgf(&Complex::new(1.0, 0.0), a*b, a, sigma, t, r0).re);
    }
    #[test]
    fn cir_with_zeros(){
        let t=1.0;
        let r0=0.04;
        let approx_bond_price=cir_mgf(&Complex::new(1.0, 0.0), 0.0, 0.0, 0.0, t, r0).re;
        assert_eq!(approx_bond_price.is_nan(), false);
    }
    #[test]
    fn cir_heston(){
        let t=0.25;
        let k=0.2;
        let v0=0.98;
        let sig=0.2;
        let rho=-0.3;
        let sig_tot=0.3;
        let u=Complex::new(0.5, 0.5);
        let neg_psi=0.5*sig_tot*sig_tot*(u-u*u);
        let k_star=k-u*rho*sig*sig_tot;
        let ada=(k_star*k_star+2.0*sig*sig*neg_psi as Complex<f64>).sqrt();
        let b_t=2.0*neg_psi*(1.0-(-ada*t).exp())/(2.0*ada-(ada-k_star)*(1.0-(-ada*t).exp()));
        let c_t=(k/(sig*sig))*(2.0*(1.0-(1.0-(-ada*t).exp())*(ada-k_star)/(2.0*ada)).ln()+(ada-k_star)*t);
        let cf_heston=(-b_t*v0-c_t).exp().re;
        let approx_heston_cf=cir_mgf_cmp(&neg_psi, k, &k_star, sig, t, v0).re;
        assert_eq!(cf_heston, approx_heston_cf);
    }
}

