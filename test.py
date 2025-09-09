import os
import argparse
import torch
import importlib
from torch import nn

device = 'cuda'

def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)

def load_original_model_and_inputs(
    model_original_src: str, context: dict, entry_point:str="Model"
) -> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """

    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None

    try:
        exec(model_original_src, context)  # expose to current namespace
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get(entry_point)
    return (Model, get_init_inputs_fn, get_inputs_fn)

def load_custom_model_with_tempfile(model_custom_file, entry_point="Model"):
    """
    This is a hack that is needed for triton code as compile / exec do not play well
    with the @triton.jit decorator.
    """
    # Create a module specification pointing to our temp file
    spec = importlib.util.spec_from_file_location("temp_module", model_custom_file)
    # Create a new module based on that spec
    temp_module = importlib.util.module_from_spec(spec)
    # Execute the code in the module's namespace
    spec.loader.exec_module(temp_module)

    Model = getattr(temp_module, entry_point)

    # Return the object (class, function, etc.) that was defined in the code
    return Model

def gen_output(Model, get_init_inputs, get_inputs):
    with torch.no_grad():
        torch.cuda.synchronize(device=device)
        set_seed(42)
        inputs = get_inputs()
        set_seed(42)
        init_inputs = get_init_inputs()
        inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
        init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in init_inputs
        ]

        # Initialize PyTorch model, use this for eager mode execution
        model = Model(*init_inputs).to(device)
        outputs = model(*inputs)
    return outputs

def is_tensor_and_0d(obj):
    return isinstance(obj, torch.Tensor) and obj.dim() == 0

def compare_kernel(current_path, kernel_item):
    # get torch Model, get_init_inputs, get_inputs
    ref_src_path = os.path.join(current_path, kernel_item[0], "torch", kernel_item[1]+".py")
    with open(ref_src_path, 'r', encoding='utf-8') as f:
        ref_code = f.read()
    ref_context = {}
    ref_Model, ref_get_init_inputs, ref_get_inputs = load_original_model_and_inputs(ref_code, ref_context)
    ref_outputs = gen_output(ref_Model, ref_get_init_inputs, ref_get_inputs)

    # get triton Model, get_init_inputs, get_inputs
    triton_src_path = os.path.join(current_path, kernel_item[0], "triton", kernel_item[1]+".py")
    Model = load_custom_model_with_tempfile(triton_src_path)
    triton_outputs = gen_output(Model, ref_get_init_inputs, ref_get_inputs)

    # ensure list
    if not isinstance(ref_outputs, (tuple, list)):
        ref_outputs = [ref_outputs]
    if not isinstance(triton_outputs, (tuple, list)):
        triton_outputs = [triton_outputs]
    
    # compare outputs
    if len(ref_outputs) != len(triton_outputs):
        print(f"Return mismatch for {kernel_item[0]}/{kernel_item[1]}.", flush=True)
        return False
    rtol, atol = (3e-2, 5e-2)
    for ref, triton in zip(ref_outputs, triton_outputs):
        if ref.dtype != triton.dtype:
            print(f"Dtype mismatch for {kernel_item[0]}/{kernel_item[1]}.", flush=True)
            return False
        elif ref.shape != triton.shape:
            print(f"Shape mismatch for {kernel_item[0]}/{kernel_item[1]}.", flush=True)
            return False
        elif not torch.allclose(ref.to(device), triton.to(device), atol=atol, rtol=rtol):
            print(f"Acc mismatch for {kernel_item[0]}/{kernel_item[1]}. max diff: {(ref - triton).abs().max().item()}, min diff: {(ref - triton).abs().mean().item()}", flush=True)
            return False
    
    return True


def main(current_path):
    test_map = {
        "dlblas": [
            "Context_Attention", "Fill_KVCache", "FlashAttention", "GroupedGemm", "GRPOLoss", 
            "LayerNorm_Gated", "Matmul_FP8", "PagedAttention", "Partial_RotaryEmbed", 
            "Selective_Scan", "Silu_Matmul"
            ],
        "liger-kernel": [
            "Dynamic_Tanh", "Fuse_Linear_CorssEntropyLoss", "Fuse_Linear_Jensen_Shannon_Distance_Loss", 
            "Fuse_Neighborhood_Attention", "Jensen_Shannon_Distance", "Matmul_Int8Int2", 
            "SparseMax", "Swiglu", "Total_Variation_Distance_Loss"
            ],
    }
    test_list = []
    succ_kernel_list = []
    fail_kernel_list = []
    for repo_name, kernel_filename in test_map.items():
        for fn in kernel_filename:
            test_list.append([repo_name, fn])
    for kernel_item in test_list:
        result = compare_kernel(current_path, kernel_item)
        if result:
            succ_kernel_list.append(kernel_item[0]+"/"+kernel_item[1])
            
        else:
            fail_kernel_list.append(kernel_item[0]+"/"+kernel_item[1])
    print(f"Succ Kernels: {succ_kernel_list}", flush=True)
    print(f"Fail Kernels: {fail_kernel_list}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test kernel pairs")
    parser.add_argument("--current_path", type=str, help="Current path for repo kernels.")
    args = parser.parse_args()
    main(args.current_path)
