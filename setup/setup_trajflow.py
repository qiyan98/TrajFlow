# Copyright (c) 2025-present, Qi Yan.
# Copyright (c) Shaoshuai Shi.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Motion Transformer (https://arxiv.org/abs/2209.13508) implementation
# from https://github.com/sshaoshuai/MTR by Shaoshuai Shi, Li Jiang, Dengxin Dai, Bernt Schiele
####################################################################################


import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(FILE_PATH)
os.chdir(PROJ_DIR)


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.0.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'trajflow/version.py')

    setup(
        name='TrajFlow',
        version=version,
        description='TrajFlow: Multi-modal Motion Prediction via Flow Matching',
        author='Qi Yan, Brian Zhang, Yutong Zhang, Daniel Yang, Joshua White, Di Chen, Jiachao Liu, Langechuan Liu, Binnan Zhuang, Shaoshuai Shi, Renjie Liao',
        license='Apache License 2.0',
        packages=find_packages(exclude=['runner', 'data', 'output']),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='knn_cuda',
                module='trajflow.mtr_ops.knn',
                sources=[
                    'src/knn.cpp',
                    'src/knn_gpu.cu',
                    'src/knn_api.cpp',
                ],
            ),
            make_cuda_ext(
                name='attention_cuda',
                module='trajflow.mtr_ops.attention',
                sources=[
                    'src/attention_api.cpp',
                    'src/attention_func_v2.cpp',
                    'src/attention_func.cpp',
                    'src/attention_value_computation_kernel_v2.cu',
                    'src/attention_value_computation_kernel.cu',
                    'src/attention_weight_computation_kernel_v2.cu',
                    'src/attention_weight_computation_kernel.cu',
                ],
            ),
        ],
    )
