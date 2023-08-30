use debug::PrintTrait;
use array::{ArrayTrait, SpanTrait};
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

#[derive(Drop)]
enum Layer {
    // (Tensor<FixedType>, Tensor<FixedType>) -> (weights, bias)
    Linear: (Tensor<FixedType>, Tensor<FixedType>),
    // Tensor<FixedType> -> (weights)
    Softmax: (Tensor<FixedType>),
}

#[derive(Drop)]
enum InitializeLayer {
    // (dim_in, dim_out)
    Linear: (u32, u32),
    Softmax: (),
}

trait LayerTrait {
    fn forward(self: Layer, input: Tensor<FixedType>) -> Tensor<FixedType>;
    fn initialize(self: InitializeLayer) -> Layer;
    fn gradient(self: Layer, input: Tensor<FixedType>) -> Array<Tensor<FixedType>>;
    fn backprop(self: Layer) -> Array<Tensor<FixedType>>;
}

impl LayerImpl of LayerTrait {
    fn forward(self: Layer, input: Tensor<FixedType>) -> Tensor<FixedType> {
        match self {
            Layer::Linear((
                theta, bias
            )) => {
                "linear layer forward pass".print();
                return NNTrait::linear(input, theta, bias);
            },
            Layer::Softmax((softmax_input)) => {
                "softmax layer forward pass".print();
                return NNTrait::softmax(softmax_input, 0);
            }
        }
    }

    fn initialize(self: InitializeLayer) -> Layer {
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
        match self {
            InitializeLayer::Linear((
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

                return Layer::Linear((weights, bias));
            },
            InitializeLayer::Softmax((softmax_input)) => {
                return Layer::Softmax((softmax_input));
            }
        }
    }

    fn gradient(self: Layer, input: Tensor<FixedType>) -> Array<Tensor<FixedType>> {
        match self {
            Layer::Linear((
                weights, bias
            )) => {
                return array![linear_weights_grad(weights), linear_bias_grad(bias)];
            },
            Layer::Softmax((softmax_input)) => {
                return array![softmax_grad(softmax_input)];
            }
        }
    }

    fn backprop(self: Layer, dloss: Tensor<FixedType>) -> Array<Tensor<FixedType>> {

    }
    
}


fn network() -> Array<Layer> {
    let mut layers = ArrayTrait::<Layer>::new();
    layers.append(InitializeLayer::Linear((2, 3)).initialize());
    layers.append(InitializeLayer::Softmax(()).initialize());
    layers.append(InitializeLayer::Linear((3, 2)).initialize());
    layers.append(InitializeLayer::Softmax(()).initialize());

    return layers;
}
