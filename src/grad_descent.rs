pub fn grad_descent<F1, F2>(loss_fn : &F1, d_loss_fn : &Vec<F2>, x0 : &Vec<f64>) -> (f64, f64)
where F1 : Fn(f64, f64) -> f64, F2 : Fn(f64, f64) -> f64{

    let mut counter     = 0;
    let mut x           = [x0[0], x0[1]];
    let     lr          = 1e-4;
    let     threshould  = 1e-8;

    loop {
        let _error = loss_fn(x[0], x[1]);

        if _error < threshould{
            println!("iteration end : loop {counter} times");
            break
        }
        
        for (i, d_fn) in d_loss_fn.iter().enumerate(){
            let _grad = d_fn(x[0], x[1]);
            x[i] = x[i] - lr * _grad;
        }
        counter += 1;
    }

    return (x[0], x[1])
}