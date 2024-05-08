mod grad_descent;
mod newton;

use crate::grad_descent::grad_descent;
use crate::newton::newton;

use std::time;

fn exp(x : f64) -> f64{
    std::f64::consts::E.powf(x)
}

fn pow(x: f64, d : f64) -> f64{
    x.powf(d)
}

fn main() {

    //関数
    let f1 = |x1, x2| { x1 - exp(x2) };
    let f2 = |x1, x2| { x1 - pow(x2, 3.) };
    let functions : Vec<fn(f64, f64)->f64> = vec![f1, f2];

    //関数の偏微分
    let f1_dx1 = |x1, x2| { 1. };
    let f1_dx2 = |x1, x2| { - exp(x2) };
    let f2_dx1 = |x1, x2| { 1. };
    let f2_dx2 = |x1, x2| { - 3. * pow(x2, 2.) };
    let d_functions : Vec<Vec<fn(f64, f64)->f64>> = vec![vec![f1_dx1, f1_dx2], vec![f2_dx1, f2_dx2]];

    //誤差関数
    let loss_fn = |x1 : f64, x2 : f64| { pow(x1 - exp(x2), 2.) + pow(x1 - x2.powf(3.), 2.)};

    //誤差関数の偏微分
    let loss_fn_dx1 = |x1, x2|{ 4.*x1 - 2.*exp(x2) - 2.*pow(x2, 3.) };
    let loss_fn_dx2 = |x1, x2|{ -2.*x1*exp(x2) + 2.*exp(2.*x2) - 6.*x1*pow(x2, 2.) + 6.*pow(x2, 5.)};
    let d_loss_fn : Vec<fn(f64, f64)->f64> = vec![loss_fn_dx1, loss_fn_dx2];

    //初期値
    let x0 = vec![3., 3.];

    //勾配降下法
    println!("Using grad descent method...");
    let start_gd = time::Instant::now();
    let ans1 = grad_descent(&loss_fn, &d_loss_fn, &x0);
    let end_gd = start_gd.elapsed();
    println!("Time elapsed : {}.{:03} sec", end_gd.as_secs(), end_gd.subsec_nanos() / 1_000_000);
    println!("x1 : {}, x2 : {}", ans1.0, ans1.1);
    println!("-------------------------");

    println!("using newton method...");
    let start_nt = time::Instant::now();
    let ans2 = newton(&functions, &d_functions, &x0);
    let end_nt = start_nt.elapsed();
    println!("Time elapsed : {}.{:03} sec", end_nt.as_secs(), end_nt.subsec_nanos() / 1_000_000);
    println!("x1 : {}, x2 : {}", ans2.0, ans2.1);
    println!("-------------------------");
}


