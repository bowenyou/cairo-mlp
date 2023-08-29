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


fn print_shape(input: Tensor<FixedType>) {
    let mut shape = input.shape;
    loop {
        if shape.len() == 0 {
            break ();
        }

        let cur_shape = shape.pop_front().unwrap();
        (*cur_shape).print();
    };
}

