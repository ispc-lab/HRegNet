from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='point_utils',
    ext_modules=[
        CUDAExtension('point_utils_cuda', [
            'src/point_utils_api.cpp',

            'src/furthest_point_sampling.cpp',
            'src/furthest_point_sampling_gpu.cu',
        ],
        extra_compile_args={
            'cxx':['-g'],
            'nvcc': ['-O2']
        })
    ],
    cmdclass={'build_ext':BuildExtension}
)