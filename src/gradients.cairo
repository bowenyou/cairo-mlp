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

fn sigmoid_grad(input: Tensor<FixedType>) -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let one = TensorTrait::new(
        shape: array![1].span(),
        data: array![FixedTrait::new(65536, false)].span(),
        extra: Option::Some(extra),
    );

    return NNTrait::sigmoid(@input) * (one - NNTrait::sigmoid(@input));
}

fn linear_weights_grad(weights: Tensor<FixedType>) -> Tensor<FixedType> {
    return weights.transpose(axes: array![1, 0].span());
}

fn linear_bias_grad(bias: Tensor<FixedType>) -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

    let tensor_size = bias.data.len();
    let mut ones_data = ArrayTrait::<FixedType>::new();
    let mut i = 0_u32;
    loop {
        if i >= tensor_size {
            break ();
        }
        ones_data.append(FixedTrait::new(65536, false));
        i += 1;
    };
    let ones = TensorTrait::new(
        shape: array![1].span(), data: ones_data.span(), extra: Option::Some(extra),
    );

    return ones;
}
