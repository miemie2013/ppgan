from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='custom_gather',
    ext_modules=CUDAExtension(
        sources=['gather.cpp', 'gather.cu']
    )
)