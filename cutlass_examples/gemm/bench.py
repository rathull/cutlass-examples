# cutlass_examples/gemm/bench.py
import torch
from .kernels.kernels import kernels_map, make_args, get_flop_count
from typing import Collection, Literal
from dataclasses import dataclass

@dataclass
class Result:
    kernel_name: str
    correct: bool
    flop_count: int
    time_ms: float | None = None
    time_ms_list: list[float] | None = None

    def __str__(self) -> str:
        if not self.correct:
            return f"=== {self.kernel_name} ===\n" \
                   f"Incorrect result!\n"
        if self.time_ms is not None:
            return f"=== {self.kernel_name} ===\n" \
                   f"Correct:    {self.correct}\n" \
                   f"Time:       {self.time_ms} ms\n" \
                   f"Flop count: {self.flop_count}\n" \
                   f"GFLOPs/s:   {self.flop_count / self.time_ms * 1e-9}\n"
        else:
            assert self.time_ms_list is not None, "No timing results available"
            avg_time_ms = sum(self.time_ms_list) / len(self.time_ms_list)
            std_time_ms = sum((x - avg_time_ms) ** 2 for x in self.time_ms_list) / len(self.time_ms_list) ** 0.5
            min_time_ms = min(self.time_ms_list)
            max_time_ms = max(self.time_ms_list)
            return f"=== {self.kernel_name} ===\n" \
                   f"Correct:            {self.correct}\n" \
                   f"Average Time:       {avg_time_ms} ms\n" \
                   f"Standard Deviation: {std_time_ms} ms\n" \
                   f"Minimum Time:       {min_time_ms} ms\n" \
                   f"Maximum Time:       {max_time_ms} ms\n" \
                   f"Flop count:         {self.flop_count}\n" \
                   f"GFLOPs/s:           {self.flop_count / avg_time_ms * 1e-9}\n"

@torch.inference_mode()
def bench(
    kernel_names: Collection[str],
    warmup_iters: int = 10,
    time_iters: int = 100,
    timing_mode: Literal["avg", "list"] = "avg",
    shape: tuple[int, int, int] = (1024, 1024, 1024),
    rtol: float = 1e-5,
    atol: float = 1e-8,
    use_tf32: bool = False,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
) -> dict[str, Result]:
    torch.backends.fp32_precision = "tf32" if use_tf32 else "ieee"
    torch.backends.cuda.matmul.fp32_precision = "tf32" if use_tf32 else "ieee"
    torch.backends.cudnn.fp32_precision = "tf32" if use_tf32 else "ieee"
    torch.backends.cudnn.conv.fp32_precision = "tf32" if use_tf32 else "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "tf32" if use_tf32 else "ieee"
    
    results: dict[str, Result] = {}
    
    # Prepare arguments
    args, out_tensor = make_args(shape, device, dtype)
    flops = get_flop_count(shape)
    
    # Check correctness of all implementations against `ref`
    kernels_map["ref"](*args, out_tensor)
    torch.cuda.synchronize()  # Make reference ready
    ref_result = out_tensor.detach().clone()
    
    correct_kernel_names: list[str] = []
    for kernel_name in kernel_names:
        if kernel_name == "ref":
            continue
        
        # Run implementation
        kernel = kernels_map[kernel_name]
        kernel(*args, out_tensor)
        torch.cuda.synchronize()
        
        # Compare out_tensor against ref_result
        is_close = torch.allclose(
            out_tensor, ref_result,
            rtol=rtol, atol=atol,
        )
        
        if is_close:
            if verbose:
                print(f"{kernel_name} implementation is correct!")
            correct_kernel_names.append(kernel_name)
        else:
            if verbose:
                print(f"{kernel_name} implementation is incorrect!")
                
                # Debug statistics
                diff = torch.abs(out_tensor - ref_result)
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                min_diff = diff.min().item()
                std_diff = diff.std().item()

                max_diff_idx = torch.argmax(diff)
                max_diff_row = max_diff_idx // out_tensor.shape[1]
                max_diff_col = max_diff_idx % out_tensor.shape[1]
                                
                num_failing_elements = (diff > atol + rtol * torch.abs(ref_result)).sum().item()
                num_total_elements = out_tensor.numel()
                percentage_failing = num_failing_elements / num_total_elements * 100
                required_rtol = torch.max(diff / (torch.abs(ref_result) + 1e-12)).item()
                required_atol = torch.max(diff).item()
                
                # Debug output
                print(f"  Max absolute difference:  {max_diff:.6e}")
                print(f"  Mean absolute difference: {mean_diff:.6e}")
                print(f"  Min absolute difference:  {min_diff:.6e}")
                print(f"  Stdev of difference:      {std_diff:.6e}")
                print(f"  Max diff location:        row={max_diff_row.item()}, col={max_diff_col.item()}")
                print(f"  Reference value:          {ref_result[max_diff_row, max_diff_col].item():.6f}")
                print(f"  Kernel value:             {out_tensor[max_diff_row, max_diff_col].item():.6f}")
                print(f"  Num failing elements:     {num_failing_elements}")
                print(f"  % of failing elements:    {percentage_failing:.2f}%")
                print(f"  Required rtol:            {required_rtol:.6e}")
                print(f"  Required atol:            {required_atol:.6e}")
            results[kernel_name] = Result(kernel_name=kernel_name, correct=False, flop_count=flops)
    
    # Timing
    kernels_to_time: set[str] = set(correct_kernel_names + ["ref"])
    
    for kernel_name in kernels_to_time:
        kernel = kernels_map[kernel_name]
        if verbose:
            print(f"Timing {kernel_name} implementation...")
        
        # Warmup without timing
        torch.cuda.synchronize()
        for _ in range(warmup_iters):    
            kernel(*args, out_tensor)
        torch.cuda.synchronize()
        
        # Timing loop
        if timing_mode == "avg":
            # Average timing
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start.record()
            for _ in range(time_iters):
                kernel(*args, out_tensor)
            end.record()
            end.synchronize()
            
            total_ms = start.elapsed_time(end)
            avg_ms = total_ms / time_iters
            
            results[kernel_name] = Result(
                kernel_name=kernel_name,
                correct=True,
                time_ms=avg_ms,
                flop_count=flops,
            )
        else:
            # Per-iter timing (heavier due to per-iter sync overhead)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            time_ms_list = []
            
            torch.cuda.synchronize()
            for _ in range(time_iters):
                start.record()
                kernel(*args, out_tensor)
                end.record()
                end.synchronize()
                time_ms_list.append(start.elapsed_time(end))
            
            results[kernel_name] = Result(
                kernel_name=kernel_name,
                correct=True,
                time_ms_list=time_ms_list,
                flop_count=flops,
            )
        
    return results