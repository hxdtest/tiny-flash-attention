Developing CUDA Kernels for Accelerated Matrix Multiplication on NVIDIA Hopper Architecture using the CUTLASS Library



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

## 版本4 - 向量内积->向量外积
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

## 原因
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
