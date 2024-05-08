use nalgebra::{vector, matrix};

pub fn newton<F1, F2>(functions : &Vec<F1>, d_functions : &Vec<Vec<F2>>, x0 : &Vec<f64>) -> (f64, f64)
where F1 : Fn(f64, f64) -> f64, F2 : Fn(f64, f64) -> f64{

	let mut counter = 0;
	let mut x1 = x0[0];
	let mut x2 = x0[0];
	let threshould = 1e-8;
	loop {
		let y1 = functions[0](x1, x2);
		let y2 = functions[1](x1, x2);

		if (y1.powf(2.) + y2.powf(2.)).abs() < threshould{
			println!("iteration end : loop {counter} times");
			break;
		}

		let y1_dx1 = d_functions[0][0](x1, x2);
		let y1_dx2 = d_functions[0][1](x1, x2);
		let y2_dx1 = d_functions[1][0](x1, x2);
		let y2_dx2 = d_functions[1][1](x1, x2);

        let x_old	= vector![x1, x2];
        let nabla 	= vector![y1, y2];
        let nabla2  = matrix![y1_dx1, y1_dx2;
        				 	  y2_dx1, y2_dx2;];
        let inv_nabla2 = (1. / (y1_dx1*y2_dx2 - y1_dx2*y2_dx1))* matrix![y2_dx2, -y1_dx2;
        						   										-y2_dx1, y1_dx1];
        //let inv_nabla2 = nabla2.try_inverse().unwrap();

        let x_new = x_old - inv_nabla2 * nabla;

        x1 = x_new[0];
        x2 = x_new[1];

        counter = counter +  1;
	}

	return (x1, x2)
}