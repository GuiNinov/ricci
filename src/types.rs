use tch::Tensor;



#[derive(Debug)]
pub struct Dataset {
    pub x_train: Tensor,
    pub y_train: Tensor,
    pub x_eval: Tensor,
    pub y_eval: Tensor,
    pub x_test: Tensor,
    pub y_test: Tensor,
}

trait Compute {
    fn forward (&self,  mem: &Memory, input: &Tensor) -> Tensor;
}