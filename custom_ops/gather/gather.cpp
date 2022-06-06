#include <paddle/extension.h>
#include <vector>
#define PADDLE_WITH_CUDA
#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

// cuda function declaration
std::vector<paddle::Tensor> gather_cuda_forward(const paddle::Tensor& input, const paddle::Tensor& index);

std::vector<paddle::Tensor> gather_cuda_backward(const paddle::Tensor& input, const paddle::Tensor& index, const paddle::Tensor& doutput);

std::vector<paddle::Tensor> gather_cuda_double_backward(const paddle::Tensor& input, const paddle::Tensor& index, const paddle::Tensor& ddx);


// use gpu or cpu impl? now only support gpu.
std::vector<paddle::Tensor> GatherForward(const paddle::Tensor& input, const paddle::Tensor& index) {
    CHECK_INPUT(input);
    CHECK_INPUT(index);
    return gather_cuda_forward(input, index);
}

std::vector<paddle::Tensor> GatherBackward(const paddle::Tensor& input, const paddle::Tensor& index, const paddle::Tensor& doutput) {
    CHECK_INPUT(input);
    CHECK_INPUT(index);
    CHECK_INPUT(doutput);
    return gather_cuda_backward(input, index, doutput);
}

std::vector<paddle::Tensor> GatherDoubleBackward(const paddle::Tensor& input, const paddle::Tensor& index, const paddle::Tensor& ddx) {
    CHECK_INPUT(input);
    CHECK_INPUT(index);
    CHECK_INPUT(ddx);
    return gather_cuda_double_backward(input, index, ddx);
}

//
std::vector<std::vector<int64_t>> gather_forward_InferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& index_shape) {
    std::vector<int64_t> output_shape;
    output_shape.push_back(index_shape[0]);
    for (int i = 1; i < input_shape.size(); i++) {
        output_shape.push_back(input_shape[i]);
    }
    return {output_shape};
}

std::vector<std::vector<int64_t>> gather_backward_InferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& index_shape,
    const std::vector<int64_t>& doutput_shape) {
    return {input_shape};
}

std::vector<std::vector<int64_t>> gather_double_backward_InferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& index_shape,
    const std::vector<int64_t>& ddx_shape) {
    std::vector<int64_t> output_shape;
    output_shape.push_back(index_shape[0]);
    for (int i = 1; i < input_shape.size(); i++) {
        output_shape.push_back(input_shape[i]);
    }
    return {output_shape};
}


//
std::vector<paddle::DataType> gather_forward_InferDtype(paddle::DataType input_dtype, paddle::DataType index_dtype) {
    return {input_dtype};
}

std::vector<paddle::DataType> gather_backward_InferDtype(paddle::DataType input_dtype, paddle::DataType index_dtype, paddle::DataType doutput_dtype) {
    return {input_dtype};
}

std::vector<paddle::DataType> gather_double_backward_InferDtype(paddle::DataType input_dtype, paddle::DataType index_dtype, paddle::DataType ddx_dtype) {
    return {input_dtype};
}


PD_BUILD_OP(gather_op)
    .Inputs({"Input", "Index"})
    .Outputs({"Output"})
    .SetKernelFn(PD_KERNEL(GatherForward))
    .SetInferShapeFn(PD_INFER_SHAPE(gather_forward_InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(gather_forward_InferDtype));

PD_BUILD_GRAD_OP(gather_op)
    .Inputs({"Input", "Index", paddle::Grad("Output")})
    .Outputs({paddle::Grad("Input")})
    .SetKernelFn(PD_KERNEL(GatherBackward));

PD_BUILD_DOUBLE_GRAD_OP(gather_op)
    .Inputs({"Input", "Index", paddle::Grad(paddle::Grad("Input"))})
    .Outputs({paddle::Grad(paddle::Grad("Output"))})
    .SetKernelFn(PD_KERNEL(GatherDoubleBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(gather_double_backward_InferShape));
