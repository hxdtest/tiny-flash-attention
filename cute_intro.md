Developing CUDA Kernels for Accelerated Matrix Multiplication on NVIDIA Hopper Architecture using the CUTLASS Library

## 影响矩阵乘法的因素
With reference to the functionality exposed by CUTLASS, performance is sensitive to many parameters such as:

决定 cuda 矩阵乘法的因素包含以下

  Shape of matrices (tall, skinny, or square);
- Layout of matrices (row or column);
- Number of warps per thread block;
- Thread block shape;
- Thread cluster shape;
- Number of pipeline stages in the software pipelining optimization;
- Chosen precision (TF32 vs FP32 vs FP16 vs FP8);
- Usage of special MMA instructions like WGMMA or TMA.
  
A basic listing of CUTLASS-based Matmul is described in Listing 1. Apart from CuTe, CUTLASS has the
following 3 important APIs for GEMM, each corresponding to a distinct level of the GPU memory hierarchy [12]:

Cutalss API 层次
## (1) Device API;
The Device API is the highest-level API. It is invoked from the Host (i.e., CPU) and does not have any detail
about the specifics of the Matmul implementation. This API is used by host-side .cu code to invoke CUTLASS’s
GEMM kernels, much like cuBLAS API.

设备API是最高级别的API。它从主机（即CPU）调用，不涉及Matmul实现的具体细节。此API由主机端的.cu代码调用，用于调用CUTLASS的GEMM内核，类似于cuBLAS API。
在主机上启动kernel

## (2) Kernel API;
The Kernel API embodies the entire grid. It thus schedules the collectives and is responsible for tiling the input
matrices into row and column panels, loading the references to them and invoking the GEMM and the epilogues.
Fusion of epilogues with GEMM happens at the Kernel API level.

内核API体现了整个网格。因此，它负责调度集体操作，并将输入矩阵切分为行和列面板，加载这些面板的引用并调用GEMM（通用矩阵乘法）和尾声操作。GEMM与尾声操作的融合在内核API级别发生。

## (3) Collective API.
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
