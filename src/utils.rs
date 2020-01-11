use num_complex::Complex;
pub fn is_same(num: f64, to_compare: f64) -> bool {
    (num - to_compare).abs() <= std::f64::EPSILON
}
pub fn is_same_cmp(num: &Complex<f64>, to_compare: f64) -> bool {
    (num.re - to_compare).abs() <= std::f64::EPSILON
}
