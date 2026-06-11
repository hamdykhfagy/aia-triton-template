import torch
import triton
import triton.language as tl

DEVICE = "cuda"


@triton.jit
def tiled_matmul_kernel(a_ptr, b_ptr, c_ptr, N,
                        BLOCK_M: tl.constexpr,
                        BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr,
                        GROUP_SIZE_M: tl.constexpr):
    pid       = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id         = pid // num_pid_in_group
    first_pid_m      = group_id * GROUP_SIZE_M
    group_size_m     = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, N, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * N + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)

    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


def tiled_matmul(a, b, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_SIZE_M=8):
    N = a.shape[0]
    c = torch.empty((N, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    tiled_matmul_kernel[grid](
        a, b, c, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=4,
    )
    return c


N = 2048
a = torch.rand((N, N), device=DEVICE, dtype=torch.float32)
b = torch.rand((N, N), device=DEVICE, dtype=torch.float32)

# warmup — forces Triton JIT compilation before ncu starts capturing
for _ in range(3):
    tiled_matmul(a, b)
torch.cuda.synchronize()

# profiling region — ncu captures these
torch.cuda.cudart().cudaProfilerStart()
for _ in range(5):
    tiled_matmul(a, b)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("done")
