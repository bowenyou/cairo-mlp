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

use cairo_mlp::network::{network, LayerImpl};

fn mse_loss_grad(y: Tensor<FixedType>, y_pred: Tensor<FixedType>) -> Array<Tensor<FixedType>> {
    return array![y - y_pred];
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

    let y = TensorTrait::<FixedType>::new(
        shape: array![3].span(),
        data: array![
            FixedTrait::new(71, true), FixedTrait::new(38, false), FixedTrait::new(62, false),
        ]
            .span(),
        extra: Option::Some(extra)
    );

    let mut layers = network();

    let mut layers_span = layers.span();
    let n_layers = layers.span().len();

    let mut i = 0_u32;

    let mut cached_output = ArrayTrait::<Tensor<FixedType>>::new();
    let mut current_output = inputs;

    loop {
        if i >= n_layers {
            break ();
        }

        let current_layer = layers_span.pop_front().unwrap();
        current_output = (*current_layer).forward(current_output);
        i += 1;
    };

    layers_span = layers.span();

    let mut current_gradients = ArrayTrait::<Array<Tensor<FixedType>>>::new();

    current_gradients.append(mse_loss_grad(y, current_output));
    let err = y - current_output;
    loop {
        if layers_span.len() == 0 {
            break ();
        }

        let current_layer = layers_span.pop_back().unwrap();
        current_gradients.append((*current_layer).gradient(err));
    }; 

    return current_output;
}

#[test]
#[available_gas(99999999999999999)]
fn test() {
    
    example();
}
