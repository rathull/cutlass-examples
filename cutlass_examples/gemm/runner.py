import modal
from .modal_images import cuda_base_hopper_image
from .bench import bench

app = modal.App("gemm-runner", image=cuda_base_hopper_image)
gpu = "H100"

def print_gpu_info():
    import torch
    
    print(f"GPU is available: {torch.cuda.is_available()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU capability: {torch.cuda.get_device_capability(0)}")
    print(f"GPU properties: {torch.cuda.get_device_properties(0)}")
    print(f"GPU total memory: {torch.cuda.get_device_properties(0).total_memory}")
    free_memory, total_memory = torch.cuda.mem_get_info(0)
    used_memory = total_memory - free_memory
    print(f"GPU free memory: {free_memory}")
    print(f"GPU used memory: {used_memory}")
    print(f"GPU total memory from mem_get_info: {total_memory}")

def gemm_runner_fn():
    print(f"Running benchmark on {gpu}...")
    print_gpu_info()
    
    results = bench(
        kernel_names=["ref", "torch_ref", "cuda"],
        warmup_iters=5,
        time_iters=50,
        timing_mode="avg",
        shape=(2048, 2048, 2048),
        rtol=1e-4,   # Reasonable tolerance for FP32 GEMM with different accumulation orders
        atol=1e-6,   # Typical for custom kernels
        use_tf32=False,
        verbose=True,
    )
    
    for _, result in results.items():
        print("\n" + str(result))

@app.function(gpu=gpu)
def gemm_runner_remote():
    gemm_runner_fn()

@app.local_entrypoint()
def main():
    print("Running benchmark locally...")
    gemm_runner_fn()
    print("Local benchmark completed successfully!")

if __name__ == "__main__":
    main()