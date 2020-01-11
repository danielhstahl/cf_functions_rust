#[macro_use]
extern crate criterion;
extern crate cf_functions;
extern crate num_complex;
use criterion::{Criterion, ParameterizedBenchmark};
use num_complex::Complex;

fn bench_mgf(c: &mut Criterion) {
    c.bench(
        "cir_mgf_analytical_numerical",
        ParameterizedBenchmark::new(
            "analytical",
            |b, _i| {
                b.iter(|| {
                    let sigma = 0.3;
                    let a = 0.3;
                    let b = 0.05;
                    let r0 = 0.05;
                    let t = 0.25;
                    cf_functions::cir::cir_mgf(&Complex::new(1.0, 0.0), a * b, a, sigma, t, r0).re
                });
            },
            vec![128, 256, 512, 1024],
        )
        .with_function("numeric", |b, i| {
            b.iter(|| {
                let sigma = 0.3;
                let a = 0.3;
                let b = 0.05;
                let r0 = 0.05;
                let t = 0.25;
                let rho0 = 0.0;
                let rho1 = 1.0;
                let k0 = a * b;
                let k1 = -a;
                let h0 = 0.0;
                let h1 = sigma * sigma;
                let l0 = 0.0;
                let l1 = 0.0;
                let cf = |u: &Complex<f64>| u.exp();
                let correlation = 0.0;
                let expected_value_of_cf = 1.0; //doesnt matter
                let u = Complex::new(1.0, 0.0);
                cf_functions::cir::generic_leverage_jump(
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
                    *i,
                )
                .re
            });
        }),
    );
}

criterion_group!(benches, bench_mgf);
criterion_main!(benches);
