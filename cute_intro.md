# Developing CUDA Kernels for Accelerated Matrix Multiplication on NVIDIA Hopper Architecture using the CUTLASS Library



# 使用 cuda 编写矩阵乘法
## 版本1 naive版本 
```Python
for i in range(0, M):
  for j in range(0, N):
     for k in range(0, K):
         c[i][j] += A[i][k] * B[k][j]
```      
## 版本2 shared memory + 合并访问存储 + float4
![image](https://github.com/user-attachments/assets/0cf59666-50e1-48da-8644-88b123975d79)
### 访存方式1，无法连续访存
```
for (int ph = 0; ph < width; ph++)
{

    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_k = 0; index_k < BK; index_k++)
        {
            if (indA + index_q < M && index_k + ph * BK < K)
            {
                SA[(threadIdx.x * TM + index_q) * BK + index_k] = dA[(indA + index_q) * K + index_k + ph * BK];
            }
            else
            {
                SA[(threadIdx.x * TM + index_q) * BK + index_k] = 0.0f;
            }
        }
    }
```
### 访存方式2，可以连续访存
```
int indA = TM * (blockIdx.x * blockDim.x);
int indB = TN * (blockIdx.y * blockDim.y);
int width = (K + BK - 1) / BK;
float tmp[TM * TN] = {0.0f};
int tid = threadIdx.x + threadIdx.y * blockDim.x;
int smem_a_m = tid % 128;
int smem_a_k = tid / 128;
int smem_b_k = tid % 8;
int smem_b_n = tid / 8;
for (int ph = 0; ph < width; ph++)
{

    if (indA + smem_a_m < M && smem_a_k + ph * BK < K)
    {
        SA[smem_a_m * BK + smem_a_k] = dA[(indA + smem_a_m) * K + smem_a_k + ph * BK];
    }
    else
    {
        SA[smem_a_m * BK + smem_a_k] = 0.0f;
    }
    if (indB + smem_b_n < N && smem_b_k + ph * BK < K)
    {

        SB[smem_b_k * BN + smem_b_n] = dB[(smem_b_k + ph * BK) * N + indB + smem_b_n];
    }
    else
    {
        SB[smem_b_k * BN + smem_b_n] = 0.0f;
    }

```
访存方式1和访存方式2对比 
![image](https://github.com/user-attachments/assets/38f90267-9a5c-4074-83c2-b0567ca9132f)

## 版本3 - shard memory bank conflict
``` C++
(float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
for (int id = 0; id < 4; id++)
{
    if (indA + smem_a_m >= M || ph * BK + 4 * smem_a_k + id >= K)
    {
        SA[(4 * smem_a_k + id) * BM + smem_a_m] = 0.0f;
    }
    else
    {
        SA[(4 * smem_a_k + id) * BM + smem_a_m] = a[id];
    }
}
```

## 版本4 向量内积 vs 向量外积
### 向量内积
```
for (int index_q = 0; index_q < TM; index_q++)
{
    for (int index_v = 0; index_v < TN; index_v++)
    {
        int reg_c_m = threadIdx.x * TM + index_q;
        int reg_c_n = threadIdx.y * TN + index_v;
        for (int index_k = 0; index_k < BK; index_k++)
        {
            tmp[index_q * TN + index_v] += SA[index_k * BM + reg_c_m] * SB[index_k * BN + reg_c_n];
        }
    }
}
```
### 向量外积
```
for (int index_k = 0; index_k < BK; index_k++)
{
    (float4 &)com_a[0] = (float4 &)SA[index_k * BM + threadIdx.x * TM];
    (float4 &)com_a[4] = (float4 &)SA[index_k * BM + threadIdx.x * TM + 4];
    (float4 &)com_b[0] = (float4 &)SB[index_k * BN + threadIdx.y * TN];
    (float4 &)com_b[4] = (float4 &)SB[index_k * BN + threadIdx.y * TN + 4];
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            tmp[index_q * TN + index_v] += com_a[index_q] * com_b[index_v];
        }
    }
}
```
将SA和SB的数据按照行，加载到寄存器，后续计算从寄存器加载
## 版本5 加载数据和矩阵运算进行流水
加载下次要计算的数据，然后进行本次的mm

###  原因
- 本质是GPU存储的层次结构，访问不同层级的存储速度不同。
- 矩阵运算的特点，C矩阵中的每个元素需要，依赖访问A矩阵行以及B矩阵中的列。


## cutlass 
![image](https://github.com/user-attachments/assets/7dc4e00a-25b4-48ef-9454-6f9d401b03d2)

![image](https://github.com/user-attachments/assets/1bd4a1a6-2454-44ba-bf61-6a08725c1e17)


With reference to the functionality exposed by CUTLASS, performance is sensitive to many parameters such as:

决定矩阵乘法的因素包含以下

  Shape of matrices (tall, skinny, or square);
- Layout of matrices (row or column);
- Number of warps per thread block;
- Thread block shape;
- Thread cluster shape;
- Number of pipeline stages in the software pipelining optimization;
- Chosen precision (TF32 vs FP32 vs FP16 vs FP8);
- Usage of special MMA instructions like WGMMA or TMA.

## Cutalss API 层次

A basic listing of CUTLASS-based Matmul is described in Listing 1. Apart from CuTe, CUTLASS has the
following 3 important APIs for GEMM, each corresponding to a distinct level of the GPU memory hierarchy [12]:


### (1) Device API;
The Device API is the highest-level API. It is invoked from the Host (i.e., CPU) and does not have any detail
about the specifics of the Matmul implementation. This API is used by host-side .cu code to invoke CUTLASS’s
GEMM kernels, much like cuBLAS API.

设备API是最高级别的API。它从主机（即CPU）调用，不涉及Matmul实现的具体细节。此API由主机端的.cu代码调用，用于调用CUTLASS的GEMM内核，类似于cuBLAS API。
在主机上启动kernel

### (2) Kernel API;
The Kernel API embodies the entire grid. It thus schedules the collectives and is responsible for tiling the input
matrices into row and column panels, loading the references to them and invoking the GEMM and the epilogues.
Fusion of epilogues with GEMM happens at the Kernel API level.

内核API体现了整个网格。因此，它负责调度集体操作，并将输入矩阵切分为行和列面板，加载这些面板的引用并调用GEMM（通用矩阵乘法）和尾声操作。GEMM与尾声操作的融合在内核API级别发生。

### (3) Collective API.
The Collective API embodies a thread block or a cluster of thread blocks (from Hopper architecture onwards).
Collective APIs can be used to construct a GEMM as well as the epilogue to be fused with GEMM. The default
epilogue simply writes out the accumulator of GEMM from register memory to global memory. CUTLASS defines
several other typical operations such as linear scaling and clamping; other device-side function call operators
may also be used to perform custom operations.

Collective API 体现了一个线程块或一个线程块集群（从Hopper架构开始）。Collective API 可用于构建 GEMM 以及与 GEMM 融合的尾部操作。默认的尾部操作简单地将 GEMM 的累加器从寄存器内存写入全局内存。CUTLASS 定义了几种其他典型操作，如线性缩放和限制；还可以使用其他设备端函数调用操作符来执行自定义操作。


```c++
using ElementA = float; // Element type for A matrix operand
using ElementB = float; // Element type for B matrix operand
using ElementC = float; // Element type for C and D matrix operands
using ArchTag = cutlass::arch::Sm90; // Tag indicating the SM
using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag
using TileShape = Shape<_128,_128,_32>; // Threadblock-level tile size
using ClusterShape = Shape<_1,_2,_1>; // Shape of the threadblocks in a cluster
```
## Collective API
```
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag, OperatorClass,
  ElementA, RowMajor, 4,
  Developing CUDA Kernels for GEMM on Hopper using CUTLASS • 9
  ElementB, ColumnMajor, 4,
  ElementAccumulator,
  TileShape, ClusterShape,
  cutlass::gemm::collective::StageCountAuto,
  cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
using CollectiveEpilogue = typename cutlass::epilogue::collective::
  CollectiveBuilder<
  cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
  TileShape, ClusterShape,
  cutlass::epilogue::collective::EpilogueTileAuto,
  ElementC, ElementC,
  ElementC, ColumnMajor, 4,
  ElementC, ColumnMajor, 4,
  cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;
```
## Kernel API
```c++
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  Shape<int,int,int>, // Indicates ProblemShape
  CollectiveMainloop,
  CollectiveEpilogue
>;
```
## Device API
```c++
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

## Tiled Matrix Multiplication Using CuTe
CuTe is another API within the CUTLASS API that provides even more flexibility to develop GEMM kernels. It specifically introduces the concept of Shapes and Layouts, using which programmers can define the different levels of tiling explicitly. Additionally, it provide APIs to:

使用cute进行矩阵乘法的workflow，提供了以下的API
- (a) Convert matrices in to tensors and partition them;
- (b) Access the tiles of a tensor that belong to a thread block (local_tiles);
- (c) Make a local partition of a tensor that belongs to a thread within a thread block (local_partition);
- (d) Copy between GEMM, SMEM and RMEM (copy);
- (e) Multiply tensors with special Matmul instructions like WGMMA (gemm);
- (f) Synchronize between thread clusters;
- (g) Make special swizzle layouts for shared memory.


###  plain SIMT mul-add
 
```
template <class MShape, class NShape, class KShape,
class TA, class AStride, class ABlockLayout, class AThreadLayout,
class TB, class BStride, class BBlockLayout, class BThreadLayout,
class TC, class CStride, class CBlockLayout, class CThreadLayout,
class Alpha, class Beta>
__global__ static
void
gemm_device(MShape M, NShape N, KShape K,
TA const* A, AStride dA, ABlockLayout blockA, AThreadLayout tA,
TB const* B, BStride dB, BBlockLayout blockB, BThreadLayout tB,
TC * C, CStride dC, CBlockLayout blockC, CThreadLayout tC,
Alpha alpha, Beta beta)
{
using namespace cute;
using X = Underscore;
// Shared memory buffers.
__shared__ TA smemA[cosize_v<ABlockLayout>];
__shared__ TB smemB[cosize_v<BBlockLayout>];
auto sA = make_tensor(make_smem_ptr(smemA), blockA);
auto sB = make_tensor(make_smem_ptr(smemB), blockB);
// Represent the full tensors.
auto mA = make_tensor(make_gmem_ptr(A), make_shape(M,K), dA);
auto mB = make_tensor(make_gmem_ptr(B), make_shape(N,K), dB);
auto mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);
// Get the appropriate blocks for this thread block.
auto MT = size<0>(sA);
auto NT = size<0>(sB);
auto KT = size<1>(sB);
auto gA = local_tile(mA, make_shape(MT, KT), make_coord(blockIdx.x, _));
auto gB = local_tile(mB, make_shape(NT, KT), make_coord(blockIdx.y, _));
auto gC = local_tile(mC, make_shape(MT, NT), make_coord(blockIdx.x, blockIdx.y);
// Define partitioned views of GMEM and SMEM for COPY

auto tAgA = local_partition(gA, tA, threadIdx.x);
auto tAsA = local_partition(sA, tA, threadIdx.x);
auto tBgB = local_partition(gB, tB, threadIdx.x);
auto tBsB = local_partition(sB, tB, threadIdx.x);
// Define partitioned views of SMEM for GEMM.
// Partition sA (M,K) by the rows of tC.
auto tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
// Partition sB (N,K) by the cols of tC.
auto tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});
// Partition gC (M,N) by the tile of tC.
auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});
// Allocate the accumulators (RMEM).
auto tCrC = make_fragment_like(tCgC);
// Clear the accumulators
clear(tCrC);


// Data is copied from GMEM to SMEM using the COPY views.
// gemm(.) operates on the GEMM views.
auto k_max = size<2>(tAgA);
for (int k = 0; k < k_max; ++k) {
// Copy GMEM to SMEM.
copy(tAgA(_,_,k), tAsA);
copy(tBgB(_,_,k), tBsB);
cp_async_fence();
cp_async_wait<0>();
__syncthreads();
// Compute GEMM on SMEM.
// Accumulate to registers.
gemm(tCsA, tCsB, tCrC);
__syncthreads();
}
 
// Epilogue fusion goes here.
for (int i = 0; i < size(tCgC); ++i)
{
tCgC(i) = tCrC(i);
}
}
```

矩阵乘法计算的关键部分在循环中列出。该 Kernel 计算矩阵 𝐴 和 𝐵 的乘积，结果为 𝐶。矩阵 𝐴 和 𝐵 如图 4 所示进行了切块处理。对于矩阵 𝐶 的切块处理，在 naive Matmul 实现中已作过讨论。naive Matmul 和此 Matmul 之间的主要区别有：

(1) 矩阵元素首先通过异步复制操作从全局内存（GMEM）带入共享内存（SMEM）；
(2) 结果矩阵 𝐶 存储在寄存器内存（RMEM）中，并最终在尾声阶段写回全局内存（GMEM）；
(3) 计算也沿着 𝐾 维度进行了切块处理。这使得第 (1) 步成为可能，因为与整个行或列面板相比，𝐴 和 𝐵 的切块足够小，可以适应共享内存。

在列表 2 中需要强调的 CuTe 的关键 API 是：

(1) local_tile：将线程块本地的切块提取到张量中。
(2) local_partition：将线程块中线程本地的元素提取到张量中。
(3) make_fragment_like: 申请寄存器，
(4) make_tensor：基于显存地址，构建tensor
(5) make_coord：构建访问坐标
(6) copy：将数据从全局显存拷贝到共享显存，不同架构的硬件设备由不同的实现
(7) gemm：矩阵乘法，不同架构的硬件设备由不同的实现，例如SIMT 以及 Tensor Core

一旦使用 CuTe API 获取了本地切块，就可以使用 GEMM API 将对应的 𝐴 和 𝐵 切块相乘，并将结果累加到 𝐶 矩阵（在寄存器中）。在沿着 𝐾 维度处理完最后一个切块后，结果 𝐶 随后被写入 GMEM。 
CuTe 中的一个重要特性是张量的视图（在 C++ 概念的意义上）。在复制操作期间，从全局内存读取数据到共享内存时，输入张量使用基于 AThreadLayoutA (tA) 和 BThreadLayout (tB) 的视图。例如，这样的视图是为了改善全局内存加载的合并效果。然而，在 GEMM 操作期间，使用的是基于 CThreadLayout (tC) 的视图。这种线程到数据的映射可以提高矩阵乘法计算的性能。但可能不会导致对全局内存的合并存储。原始共享内存可以使用**不同的视图进行读取和写入**。因此，复制和 GEMM 操作的线程布局是解耦的，以便用户可以为每个操作选择最佳的选项
CuTe 访问全局存储时尽可能合并访存，将数据存储在共享存储时避免再次访问是bank conflict。


###  Incorporating TMA and WGMMA instructions from NVIDIA Hopper Architecture

The copy API call should be changed to include the TMA copy atom;
The gemm API call should be changed to include the MMA atom – for Hopper, we choose WGM

```c++
....
for (int k = 0; k < size<1>(tAgA); ++k)
{
.....
//copy A and B from GMEM to SMEM using COPY views.
if (threadIdx.x == 0)
{
/// Initialize shared memory barrier
....
copy(tma_copy_a, tAgA(_,k), tAsA);
copy(tma_copy_b, tBgB(_,k), tBsB);
}
__syncthreads();
warpgroup_fence_operand(tCrC);
cute::gemm(wmma_atom, tCrA, tCrB, tCrC);
warpgroup_commit_batch();
warpgroup_wait<1>();
__syncthreads();
}
```


### ADDITIONAL OPTIMIZATIONS

从第2节回顾，最佳的CUTLASS核对于SGEMM的性能约为280 TFLOPS，而cuBLAS的性能约为215 TFLOPS。CUTLASS通过实现更多的优化来达到这一优越的性能水平。以下是文档中提到的一些优化措施[11]：

(1) 软件流水线 – 软件流水线是一种隐藏内存延迟的技术，它通过让内存访问和数学指令并发执行来实现，同时始终考虑这些步骤之间的依赖关系。CUTLASS实现使用了多个缓冲区，既包括线程块层面，也包括warp层面。

(2) Warp专业化 – 在软件流水线等优化下，不同的线程或线程组自然具有不同的角色。一些线程是生产者，负责加载数据，而其他线程是消费者，负责执行MMA指令。warp专业化的思想是将线程块中的warps空间上划分为两组，分别作为生产者和消费者。

(3) 持久核 – 持久核是一种CUDA设计模式，旨在通过让核在GPU上持续存在以避免内核启动和配置的开销。在CUTLASS中，这意味着持久线程块在其生命周期内计算多个输出块。

(4) 两个协作消费warp组 – WGMMA允许操作数𝐴的块存放在寄存器内存中，而不是共享内存中。然而，这限制了𝐴的块大小，因为寄存器空间有限。将块大小在𝑀维度上拆分并分配给两个不同的消费warp组，可以允许更大的块大小并减轻寄存器压力。

(5) Warp专业化持久ping-pong核 – 从(4)中的两个消费warp组各自分配给不同的输出块。这允许一个消费warp组的最终红利与另一个消费warp组的数学操作重叠，从而最大化张量核心的利用率。同时在生产者warp组之间进行同步。

根据我们的实验研究，特别是(5)这一点在第2图的第四列与280 TFLOPS最佳测量CUTLASS核之间的差距上起到了重要作用。


### cute 扩展性
 BATCHED-GEMM
The AI workflow that we are targeting does not involve multiplying large square matrices. Instead, it involves
large square matrices decomposed as products of matrices with small 𝐾 (e.g., 64 or 128), and with batch count
𝐿 > 1 (e.g., 64 or 96); cf. [1, §2.2]. Such a scheme is popularly known as Batched-GEMM. Our CuTe program can
be extended to handle Batched-GEMM by simply setting the third dimension of the grid to be 𝐿. We then use
blockIdx.z when using the local_tile operation inside the CUDA kernel, as shown in listing 4.
```
auto gA = local_tile(mA, make_shape(MT, KT), make_coord(blockIdx.x, _, blockIdx.
z));
auto gB = local_tile(mB, make_shape(NT, KT), make_coord(blockIdx.y, _, blockIdx.
z));
auto gC = local_tile(mC, make_shape(MT, NT), make_coord(blockIdx.x, blockIdx.y,
blockIdx.z);
```
Listing 4. Batched-GEMM kernel using CuTe
Performance of such a Batched-GEMM using CuTe is shown in Figure 6. Surprisingly, the CuTe program
outperforms both cuBLAS and CUTLASS, even though it does not use any of the additional optimizations that
CUTLASS uses as listed in §5.



Based on the layout of the shared memory and the mma instruction being used - we change the swizzle pattern to ensure both the stores and loads to and from shared memory are bank conflict free.
