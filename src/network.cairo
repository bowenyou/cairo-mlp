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

use cairo_mlp::gradients::{sigmoid_grad, linear_weights_grad, linear_bias_grad};
use cairo_mlp::utils::print_shape;


#[derive(Copy, Drop)]
enum Layer {
    // (Tensor<FixedType>, Tensor<FixedType>) -> (weights, bias)
    Linear: (Tensor<FixedType>, Tensor<FixedType>),
    Sigmoid: (),
}

#[derive(Copy, Drop)]
enum InitializeLayer {
    // (dim_in, dim_out)
    Linear: (u32, u32),
    Sigmoid: (),
}

trait LayerTrait {
    fn forward(self: Layer, input: Tensor<FixedType>) -> Tensor<FixedType>;
    fn initialize(self: InitializeLayer) -> Layer;
    fn gradient(self: Layer, input: Tensor<FixedType>) -> Array<Tensor<FixedType>>;
}

impl LayerImpl of LayerTrait {
    fn forward(self: Layer, input: Tensor<FixedType>) -> Tensor<FixedType> {
        match self {
            Layer::Linear((
                theta, bias
            )) => {
                'linear layer forward pass'.print();
                return NNTrait::linear(input, theta, bias);
            },
            Layer::Sigmoid(()) => {
                'sigmoid layer forward pass'.print();
                return NNTrait::sigmoid(@input);
            },
        }
    }

    fn initialize(self: InitializeLayer) -> Layer {
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
        match self {
            InitializeLayer::Linear((
                dim_out, dim_in
            )) => {
                let mut i = 0_u32;
                let mut bias_data = ArrayTrait::<FixedType>::new();
                let mut weights_data = ArrayTrait::<FixedType>::new();

                loop {
                    if i >= dim_out {
                        break ();
                    }
                    bias_data.append(FixedTrait::new(0, false));
                    let mut j = 0_u32;
                    loop {
                        if j >= dim_in {
                            break ();
                        }
                        weights_data.append(FixedTrait::new(0, false));
                        j += 1;
                    };
                    i += 1;
                };

                let weights = TensorTrait::<FixedType>::new(
                    shape: array![dim_out, dim_in].span(),
                    data: weights_data.span(),
                    extra: Option::Some(extra)
                );

                let bias = TensorTrait::<FixedType>::new(
                    shape: array![dim_out].span(),
                    data: bias_data.span(),
                    extra: Option::Some(extra)
                );

                return Layer::Linear((weights, bias));
            },
            InitializeLayer::Sigmoid(()) => {
                return Layer::Sigmoid(());
            },
        }
    }

    fn gradient(self: Layer, input: Tensor<FixedType>) -> Array<Tensor<FixedType>> {
        match self {
            Layer::Linear((
                weights, bias
            )) => {
                'got linear gradient'.print();
                return array![linear_weights_grad(input), linear_bias_grad(input)];
            },
            Layer::Sigmoid(()) => {
                'got sigmoid gradient'.print();
                return array![sigmoid_grad(input)];
            },
        }
    }
}


fn network() -> Array<Layer> {
    let mut layers = ArrayTrait::<Layer>::new();

    // 3x1
    layers.append(InitializeLayer::Linear((3, 3)).initialize());
    // 2x1
    layers.append(InitializeLayer::Sigmoid(()).initialize());

    return layers;
}
