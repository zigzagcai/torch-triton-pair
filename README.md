# Pair of PyTorch and Triton code from open-sourced kernel libraries

**Notice**:

This repository is just a copy and paste of the ***<torch, triton>*** pair in some popular open-sourced kernel repositories such like ***DLBLAS/Liger-Kernel/FlagGems***, to help newbies like me to understand the functionality of each triton kernel better and easier.

## Current Status

| Repository     |   Kernel                                   |
|----------------|--------------------------------------------|
| DLBLAS         | Context_Attention                          |
| DLBLAS         | Fill_KVCache                               |
| DLBLAS         | FlashAttention                             |
| DLBLAS         | GroupedGemm                                |
| DLBLAS         | GRPOLoss                                   |
| DLBLAS         | LayerNorm_Gated                            |
| DLBLAS         | Matmul_FP8                                 |
| DLBLAS         | PagedAttention                             |
| DLBLAS         | Partial_RotaryEmbed                        |
| DLBLAS         | Selective_Scan                             |
| DLBLAS         | Silu_Matmul                                |
| Liger-Kernel   | Dynamic_Tanh                               |
| Liger-Kernel   | Fuse_Linear_CorssEntropyLoss               |
| Liger-Kernel   | Fuse_Linear_Jensen_Shannon_Distance_Loss   |
| Liger-Kernel   | Fuse_Neighborhood_Attention                |
| Liger-Kernel   | Jensen_Shannon_Distance                    |
| Liger-Kernel   | Matmul_Int8Int2                            |
| Liger-Kernel   | SparseMax                                  |
| Liger-Kernel   | Swiglu                                     |
| Liger-Kernel   | Total_Variation_Distance_Loss              |