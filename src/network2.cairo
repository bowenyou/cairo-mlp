use debug::PrintTrait;
use array::{ArrayTrait, SpanTrait};
use option::OptionTrait;
use orion::operators::{
    tensor::{
        core::{Tensor, TensorTrait, ExtraParams},
        implementations::impl_tensor_fp::{
            Tensor_fp, FixedTypeTensorMul, FixedTypeTensorSub, FixedTypeTensorDiv
        },
    },
    nn::{core::NNTrait, implementations::impl_nn_fp::NN_fp}
};
use orion::numbers::fixed_point::{
    core::{FixedTrait, FixedType, FixedImpl},
    implementations::fp16x16::core::{FP16x16Impl, FP16x16Div, FP16x16PartialOrd}
};
use cairo_mlp::gradients::{linear_weights_grad, linear_bias_grad, softmax_grad};

// Layer struct with parameters: LayerType, weights, bias, gradient
struct Layer {
    layer_type: LayerType,
    old_weights: Option<Tensor<FixedType>>,
    weights: Tensor<FixedType>,
    bias: Tensor<FixedType>,
    gradient: Tensor<FixedType>,
}

// implementation with previous functions + update weights function
#[derive(Drop)]
enum LayerType {
    // (Tensor<FixedType>, Tensor<FixedType>) -> (weights, bias)
    Linear: (Tensor<FixedType>, Tensor<FixedType>),
    // Tensor<FixedType> -> (weights)
    Softmax: (Tensor<FixedType>),
}

impl Layer {
    fn initialize(layer_type: LayerType, input_size: u32, output_size: u32) -> Self {
        match layer_type {
            LayerType::Linear((
                dim_in, dim_out
            )) => {
                let mut i = 0_u32;
                let mut bias_data = ArrayTrait::<FixedType>::new();
                let mut weights_data = ArrayTrait::<FixedType>::new();

                loop {
                    if i >= dim_in {
                        break ();
                    }
                    bias_data.append(FixedTrait::new(0, false));
                    let mut j = 0_u32;
                    loop {
                        if j >= dim_out {
                            break ();
                        }
                        weights_data.append(FixedTrait::new(0, false));
                        j += 1;
                    };
                    i += 1;
                };

                let weights = TensorTrait::<FixedType>::new(
                    shape: array![dim_in, dim_out].span(),
                    data: weights_data.span(),
                    extra: Option::Some(extra)
                );

                let bias = TensorTrait::<FixedType>::new(
                    shape: array![dim_in].span(), data: bias_data.span(), extra: Option::Some(extra)
                );

                Layer {
                    layer_type: LayerType::Linear(weights.clone(), bias.clone()),
                    old_weights: None,
                    weights,
                    bias,
                    gradient: Tensor::new_zeros(output_size),
                }
            }
            LayerType::Softmax => {
                let weights = dummy_weights(input_size, output_size);
                Layer {
                    layer_type: LayerType::Softmax(weights.clone()),
                    old_weights: None,
                    weights,
                    bias: Tensor::new_zeros(output_size), 
                    gradient: Tensor::new_zeros(output_size),
                }
            }
        }
    }

    fn forward(&self, input: Tensor<FixedType>) -> Tensor<FixedType> {
        match &self.layer_type {
            LayerType::Linear(weights, bias) => NNTrait::linear(input, weights, bias),
            LayerType::Softmax(weights) => NNTrait::softmax(input.matmul(weights)),
        }
    }

    fn gradient(&mut self, dloss: Tensor<FixedType>) {
        match &self.layer_type {
            LayerType::Linear(weights, bias) => {
                self.gradient = dloss * linear_weights_grad(weights);
            }
            LayerType::Softmax(weights) => {
                self.gradient = dloss * softmax_grad(weights); 
            }
        }
    }

    fn backprop(&mut self, dloss: Tensor<FixedType>, learning_rate: f64) {
        self.gradient(dloss);

        match &mut self.layer_type {
            LayerType::Linear(weights, bias) => {
                self.old_weights = Some(weights.clone());
                weights -= learning_rate * linear_weights_grad(weights);
                bias -= learning_rate * linear_bias_grad(bias));
            }
            LayerType::Softmax(weights) => {
                self.old_weights = Some(weights.clone());
                weights -= learning_rate * self.gradient;
            }
        }
    }
}

fn dummy_weights(input_size: u32, output_size: u32) -> Tensor<FixedType> {
    let mut value = 0.0;
    
    let mut weights_data = Vec::new();
    for _ in 0..(input_size * output_size) {
        weights_data.push(FixedType::from(value)); 
        
        value += 0.01;  
    }

    Tensor::new(weights_data, (input_size, output_size))  
}
