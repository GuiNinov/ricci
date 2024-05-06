use tch::{Device, Tensor};

use crate::types::Dataset;

pub fn split_dataset(
    raw_dataset: Vec<usize>,
    splits_params: (f64, f64, f64),
    device: Device,
) -> Dataset {

    let (train_split, eval_split, test_split) = splits_params;

    let train_size = (raw_dataset.len() as f64 * train_split).round() as usize;

    let eval_size = (raw_dataset.len() as f64 * eval_split).round() as usize;

    let test_size = (raw_dataset.len() as f64 * test_split).round() as usize;

    let x_train: Vec<u8> = raw_dataset[0..train_size].to_vec().iter().map(| v | *v as u8).collect();
    let y_train: Vec<u8> = raw_dataset[1..train_size + 1].to_vec().iter().map(| v | *v as u8).collect();

    let x_eval: Vec<u8> = raw_dataset[train_size..train_size + eval_size].to_vec().iter().map(| v | *v as u8).collect();
    let y_eval: Vec<u8> = raw_dataset[train_size + 1..train_size + eval_size + 1].to_vec().iter().map(| v | *v as u8).collect();

    let x_test: Vec<u8> = raw_dataset[train_size + eval_size..train_size + eval_size + test_size].to_vec().iter().map(| v | *v as u8).collect();
    let y_test: Vec<u8> = raw_dataset[train_size + eval_size + 1..train_size + eval_size + test_size + 1].to_vec().iter().map(| v | *v as u8).collect();

    // println!("x_train: {:?}", &x_train.iter().map(| v | v.try_into().unwrap()).collect());
    let x_train_tensor = Tensor::from_slice(&x_train);
    let y_train_tensor = Tensor::from_slice(&y_train);

    let x_eval_tensor = Tensor::from_slice(&x_eval);
    let y_eval_tensor = Tensor::from_slice(&y_eval);

    let x_test_tensor = Tensor::from_slice(&x_test);
    let y_test_tensor = Tensor::from_slice(&y_test);

    let dataset = Dataset {
        x_train: x_train_tensor,
        y_train: y_train_tensor,
        x_eval: x_eval_tensor,
        y_eval: y_eval_tensor,
        x_test: x_test_tensor,
        y_test: y_test_tensor,
    };
    
    return dataset
}