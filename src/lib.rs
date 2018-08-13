extern crate num_complex;

use num_complex::Complex;
use special::Gamma;

use std::f64::consts::PI;
/**All CFs are with respect to the complex u (ie, ui) */
pub fn gauss_log_cf(
    u:&Complex<f64>,
    mu:f64,
    sigma:f64
)->Complex<f64>
{
    u*mu+u*u*0.5*sigma.powi(2)
}
//I hate rust's inability to do generics properly
fn gauss_log_cf_cmp(
    u:&Complex<f64>,
    mu:&Complex<f64>,
    sigma:f64
)->Complex<f64>
{
    u*mu+u*u*0.5*sigma.powi(2)
}

pub fn gauss_cf(
    u:&Complex<f64>,
    mu:f64,
    sigma:f64
)->Complex<f64>
{
    gauss_log_cf(u, mu, sigma).exp()
}

pub fn merton_log_cf(
    u:&Complex<f64>,
    lambda:f64,
    mu_l:f64,
    sig_l:f64
)->Complex<f64>
{
    lambda*(gauss_cf(u, mu_l, sig_l)-1.0)
}

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

//see http://finance.martinsewell.com/stylized-facts/distribution/CarrGemanMadanYor2002.pdf pg 10
pub fn cgmy_log_cf(
    u:&Complex<f64>,
    c:f64,
    g:f64,
    m:f64,
    y:f64
)->Complex<f64>{
    if is_same_cmp(y, 1.0) {
        Complex::new(0.0, 0.0)
    }
    else if is_same_cmp(y, 0.0) {
        c*(1.0-u/g).ln()*(1.0+u/m)
    }
    else {
        c*(-y).gamma()*(m-u).powf(y)+(g+u).powf(y)-m.powf(y)-g.powf(y)
    }
}
//see http://finance.martinsewell.com/stylized-facts/distribution/CarrGemanMadanYor2002.pdf pg 12 and 13
pub fn cgmy_log_risk_neutral_cf(
    u:&Complex<f64>,
    c:f64,
    g:f64,
    m:f64,
    y:f64,
    r:f64,
    sigma:f64
)->Complex<f64>{
    gauss_log_cf_cmp(
        u, 
        r-sigma.powi(2)*0.5-cgmy_log_cf(complex::new(1.0, 0.0), c, g, m, y),
        sigma
    )+cgmy_log_cf(u, c, g, m, y)
}



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
//hate Rusts lack of good generics
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

pub fn gamma_cf(
    u:&Complex<f64>,
    a:f64,
    b:f64
)->Complex<f64>{
    (1.0-u*b).powf(-a)
}

pub fn merton_time_change_log_cf(
    u:&Complex<f64>,
    t:f64,
    lambda:f64,
    mu_l:f64,
    sig_l:f64,
    sigma:f64,
    v0:f64,
    speed:f64,
    ada_v:f64,
    rho:f64    
)->Complex<f64>{
    let cf_rn=-merton_log_risk_neutral_cf(u, lambda, mu_l, sig_l, 0.0, sigma);
    let ln_m=speed-ada_v*rho*u*sigma;
    cir_log_mgf_cmp(
        &cf_rn, 
        speed,
        &ln_m,
        ada_v,
        t, 
        v0
    )
}

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

