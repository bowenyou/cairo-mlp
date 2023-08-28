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

#[derive(Drop)]
enum Layer {
    // (Tensor<FixedType>, Tensor<FixedType>) -> (weights, bias)
    Linear: (Tensor<FixedType>, Tensor<FixedType>),
    Sigmoid: ()
}

#[derive(Drop)]
enum InitializeLayer {
    Linear: (u32, u32),
    Sigmoid: ()
}

trait LayerTrait {
    fn forward(self: Layer, input: Tensor<FixedType>) -> Tensor<FixedType>;
    fn initialize(self: InitializeParams) -> Layer;
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
            }
        }
    }
    fn initialize(self: InitializeParams) -> Layer {
        match self {
            InitializeLayer::Linear((
                dim_in, dim_out
            )) => {
                let mut i = 0_u32;
                let mut bias_data = ArrayTrait::<FixedTrait>::new();
                let mut weights_data = ArrayTrait::<FixedTrait>::new();

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
            InitializeLayer::Sigmoid(()) => {
                return Layer::Sigmoid(());
            }
        }
    }
}


fn network() -> ArrayTrait::<Layers> {}
