# Pair of PyTorch and Triton code from open-sourced kernel libraries

**Notice**:

This repository is just a copy and paste of the ***<torch, triton>*** pair in some popular open-sourced kernel repositories such like ***DLBLAS/Liger-Kernel/FlagGems***, to help newbies like me to understand the functionality of each triton kernel better and easier.

## Current Status

| Repository     | Kernel                                     | torch-triton-pair
|----------------|--------------------------------------------|--------------------|
| DLBLAS         | Context_Attention                          | <[torch](dlblas/torch/Context_Attention.py), [triton](dlblas/triton/Context_Attention.py)>    |
| DLBLAS         | Fill_KVCache                               | <[torch](dlblas/torch/Fill_KVCache.py), [triton](dlblas/triton/Fill_KVCache.py)>    |
| DLBLAS         | FlashAttention                             | <[torch](dlblas/torch/FlashAttention.py), [triton](dlblas/triton/FlashAttention.py)>    |
| DLBLAS         | GroupedGemm                                | <[torch](dlblas/torch/GroupedGemm.py), [triton](dlblas/triton/GroupedGemm.py)>    |
| DLBLAS         | GRPOLoss                                   | <[torch](dlblas/torch/GRPOLoss.py), [triton](dlblas/triton/GRPOLoss.py)>    |
| DLBLAS         | LayerNorm_Gated                            | <[torch](dlblas/torch/LayerNorm_Gated.py), [triton](dlblas/triton/LayerNorm_Gated.py)>    |
| DLBLAS         | Matmul_FP8                                 | <[torch](dlblas/torch/Matmul_FP8.py), [triton](dlblas/triton/Matmul_FP8.py)>    |
| DLBLAS         | PagedAttention                             | <[torch](dlblas/torch/PagedAttention.py), [triton](dlblas/triton/PagedAttention.py)>    |
| DLBLAS         | Partial_RotaryEmbed                        | <[torch](dlblas/torch/Partial_RotaryEmbed.py), [triton](dlblas/triton/Partial_RotaryEmbed.py)>    |
| DLBLAS         | Selective_Scan                             | <[torch](dlblas/torch/Selective_Scan.py), [triton](dlblas/triton/Selective_Scan.py)>    |
| DLBLAS         | Silu_Matmul                                | <[torch](dlblas/torch/Silu_Matmul.py), [triton](dlblas/triton/Silu_Matmul.py)>    |
| Liger-Kernel   | Dynamic_Tanh                               | <[torch](liger-kernel/torch/Dynamic_Tanh.py), [triton](liger-kernel/triton/Dynamic_Tanh.py)>    |
| Liger-Kernel   | Fuse_Linear_CorssEntropyLoss               | <[torch](liger-kernel/torch/Fuse_Linear_CorssEntropyLoss.py), [triton](liger-kernel/triton/Fuse_Linear_CorssEntropyLoss.py)>    |
| Liger-Kernel   | Fuse_Linear_Jensen_Shannon_Distance_Loss   | <[torch](liger-kernel/torch/Fuse_Linear_Jensen_Shannon_Distance_Loss.py), [triton](liger-kernel/triton/Fuse_Linear_Jensen_Shannon_Distance_Loss.py)>    |
| Liger-Kernel   | Fuse_Neighborhood_Attention                | <[torch](liger-kernel/torch/Fuse_Neighborhood_Attention.py), [triton](liger-kernel/triton/Fuse_Neighborhood_Attention.py)>    |
| Liger-Kernel   | Jensen_Shannon_Distance                    | <[torch](liger-kernel/torch/Jensen_Shannon_Distance.py), [triton](liger-kernel/triton/Jensen_Shannon_Distance.py)>    |
| Liger-Kernel   | Matmul_Int8Int2                            | <[torch](liger-kernel/torch/Matmul_Int8Int2.py), [triton](liger-kernel/triton/Matmul_Int8Int2.py)>    |
| Liger-Kernel   | SparseMax                                  | <[torch](liger-kernel/torch/SparseMax.py), [triton](liger-kernel/triton/SparseMax.py)>    |
| Liger-Kernel   | Swiglu                                     | <[torch](liger-kernel/torch/Swiglu.py), [triton](liger-kernel/triton/Swiglu.py)>    |
| Liger-Kernel   | Total_Variation_Distance_Loss              | <[torch](liger-kernel/torch/Total_Variation_Distance_Loss.py), [triton](liger-kernel/triton/Total_Variation_Distance_Loss.py)>    |
