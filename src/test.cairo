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

trait ForwardTrait {
    fn forward(self: Layer, input: Tensor<FixedType>) -> Tensor<FixedType>;
}

impl ForwardImpl of ForwardTrait {
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
}

fn example() -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

    // We instantiate inputs here.
    let inputs = TensorTrait::<FixedType>::new(
        shape: array![3].span(),
        data: array![
            FixedTrait::new(71, true), FixedTrait::new(38, false), FixedTrait::new(62, false),
        ]
            .span(),
        extra: Option::Some(extra)
    );

    // We instantiate weights here.
    let weights = TensorTrait::<FixedType>::new(
        shape: array![2, 3].span(),
        data: array![
            FixedTrait::new(8, true),
            FixedTrait::new(64, false),
            FixedTrait::new(40, false),
            FixedTrait::new(33, true),
            FixedTrait::new(34, true),
            FixedTrait::new(20, true),
        ]
            .span(),
        extra: Option::Some(extra)
    );

    let weights2 = TensorTrait::<FixedType>::new(
        shape: array![3, 2].span(),
        data: array![
            FixedTrait::new(8, true),
            FixedTrait::new(64, false),
            FixedTrait::new(40, false),
            FixedTrait::new(33, true),
            FixedTrait::new(34, true),
            FixedTrait::new(20, true),
        ]
            .span(),
        extra: Option::Some(extra)
    );

    // We instantiate bias here.
    let bias = TensorTrait::<FixedType>::new(
        shape: array![2].span(),
        data: array![FixedTrait::new(61, false), FixedTrait::new(61, true),].span(),
        extra: Option::Some(extra)
    );

    let bias2 = TensorTrait::<FixedType>::new(
        shape: array![3].span(),
        data: array![
            FixedTrait::new(61, false), FixedTrait::new(61, true), FixedTrait::new(61, true)
        ]
            .span(),
        extra: Option::Some(extra)
    );

    let mut layers = ArrayTrait::<Layer>::new();
    layers.append(Layer::Linear((weights, bias)));
    layers.append(Layer::Linear((weights2, bias2)));
    layers.append(Layer::Sigmoid(()));

    let n_layers = layers.span().len();

    let mut i = 0_u32;

    let mut current_output = inputs;

    loop {
        if i >= n_layers {
            break ();
        }

        let current_layer = layers.pop_front().unwrap();
        current_output = current_layer.forward(current_output);
        i += 1;
    };

    return current_output;
}


#[test]
#[available_gas(99999999999999999)]
fn test() {
    example();
}
