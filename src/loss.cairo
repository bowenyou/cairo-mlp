use array::{ArrayTrait, SpanTrait};

// cross-entropy loss function
// fn cross_entropy_loss(y_pred: &[f64], y_true: &[f64]) -> f64 {
//     if y_pred.len() != y_true.len() {
//         panic!("y_pred and y_true must have the same length");
//     }

//     let mut loss = 0.0;
//     let mut index = 0;
//     loop {
//         if index >= y_pred.len() {
//             break;
//         }
//         let pred = y_pred[index];
//         let true_val = y_true[index];

//         if pred != 0.0 {
//             loss -= true_val * pred.log(std::f64::consts::E);
//         }

//         index += 1;
//     }
//     loss
// }