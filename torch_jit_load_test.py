import torch

model="runtime_outs/ne_chatglm_q_int8_jblas_cbf16_g32.bin"
res=torch.jit.load(model)

print(res)
