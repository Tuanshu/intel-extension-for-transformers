//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/types.h>
#include "jblas/jit_blas_wrapper.h"

torch::Tensor jblas_mmbf16_packwei(torch::Tensor& weight, bool transpose);
void jblas_mmbf16(torch::Tensor& activation, torch::Tensor& weight, torch::Tensor& output);

torch::Tensor jblas_mmfp32_avx2_packwei(torch::Tensor& weight, bool transpose);
void jblas_mmfp32_avx2(torch::Tensor& activation, torch::Tensor& weight, torch::Tensor& output);