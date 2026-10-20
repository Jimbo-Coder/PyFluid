# Precision investigation plan

Status: planning only, Cycle460. No solver changes, builds, simulations or
compiler-policy changes are authorized by this request. CPU FP64 remains the
reference; existing CPU/CUDA FP32 is already a supported build policy with
scoped numerical acceptance. Metal remains device FP32-only.

## Formats and hardware

Use bit counts in names: FP32 means four bytes; FP64 means eight bytes.
If “4/8” refers to bytes, those are the existing two policies. FP4/FP8 mean
four/eight bits and are separate, much coarser formats. Describe storage,
working arithmetic and accumulation independently when they differ.

The checked-in workstation profile targets RTX3070Ti, compute capability8.6.
NVIDIA documents128 FP32 versus2 FP64 cores per SM and the corresponding
native arithmetic throughput: FP64 peak is approximately1/64 of FP32.
This ratio does not predict whole-solver throughput. Bandwidth, recovery,
iteration counts, synchronization and startup still matter. The existing
wsl_cuda profile already selects FP32 with precise math.

| Format/policy | Proposed role | Readiness |
| --- | --- | --- |
| FP64 | Numerical reference and demanding runs | Existing build policy; CPU is the practical reference here |
| FP32 | Main GPU production candidate | Existing build policy; acceptance depends on problem/resolution/time |
| Mixed FP32/FP64 | FP32 bulk work with measured sensitive operations widened | First new candidate; CUDA/CPU capability required, not available as device FP64 on Metal |
| FP16 / BF16 | Selected storage/work buffers with FP32 arithmetic/accumulation, if a measured bandwidth bottleneck justifies them | Exploratory; neither is an AMReX Real build option |
| TF32 | Possible future matrix-multiply-heavy component | Tensor Core mode, not a general stencil scalar type |
| FP8 / FP4, including scaled vendor variants | Optional bounded storage/compression investigation on suitable future hardware | No native FP8/FP4 arithmetic acceleration on the current Ampere card; not a proposed global evolution type |

CUDA format headers/conversions do not establish native arithmetic support on
a device. INT4/INT8 support also does not imply FP4/FP8 support. Lower storage
width need not improve total runtime after conversion and packing costs.

## Investigation sequence when resumed

1. Establish an FP32 acceptance envelope using existing analytic/reference
   controls: smooth hydro/RHD entropy convergence, Sod, MHD Alfvén/Brio-Wu
   AMR+CT, representative RMHD recovery, diffusion and gravity. Sweep a bounded
   set of resolutions and physical durations. Include the existing failed
   strongly magnetized hybrid-SLy FP32 preflight and cancellation-sensitive
   cold stellar cases rather than discarding them. Start with small cases;
   authorize longer runs only if the small controls discriminate candidates.
2. Compare error against both an analytic/high-resolution oracle and CPU FP64.
   Keep CFL, methods, physical setup and conservation contracts fixed. Record
   physical admissibility, recovery failures/iterations, floors/resets,
   mass/energy/momentum drift, divB, elliptic residuals, AMR levels/headroom,
   throughput, end-to-end time and memory. Find roundoff plateaus and failure
   boundaries; stability alone is insufficient. Compare time/memory at equal
   accepted physical error, allowing different resolutions when justified.
3. Profile the failing or expensive operation before choosing mixed precision.
   Candidates include recovery intermediates, cancellation-prone expressions,
   time accumulation, reductions and elliptic residual/correction work.
   Existing compensated reductions and affine stage updates are established
   protections, not missing features. Widen only the operation supported by
   evidence. Wider computation cannot recover information already lost when
   a conserved state was stored in FP32; retain or reformulate sensitive state
   quantities only when the numerical analysis and held-out cases justify it.
4. If profiling proves a bandwidth/storage bottleneck, try FP16/BF16 only at an
   explicitly bounded storage boundary, with FP32 computation and measured
   conversion overhead. State/face-B/AMR transfer/checkpoint support is a larger
   design than a scratch-buffer prototype. Preserve flux conservation and CT
   ownership; do not quantize stages or coarse-fine transfers independently.
   Tensor Core use requires a real matrix workload and an end-to-end comparison.
5. Promote only candidates that preserve the physical acceptance gates and
   improve time or memory at equal accuracy. Document unsupported formats and
   capability failures explicitly. Do not globally loosen tolerances, enable
   fast math, repair failed states or silently demote requested precision.

## Implementation constraints for a later design

AMReX/PyFluid currently select SINGLE or DOUBLE for amrex::Real. FP16/BF16/FP8/
FP4 are not extra values to add to that CMake enum. Lower-width storage would
need explicit representation, typed loads/stores, arithmetic ownership,
communication and I/O metadata. Shared physics and dimension projections must
remain shared; vendor format adapters belong at representation boundaries.

Do not instantiate every precision combination across every kernel. Begin with
one source owner or isolated representation boundary; retain existing300-second
compile deadlines and measure edit cost as well as runtime. CUDA's unresolved
30--60s development goal is part of the adoption decision. Capability profiles
must separate a known dtype from a supported operation and a validated problem.

## Source references

- Current contracts and measured limits: [precision_and_metal.md](precision_and_metal.md).
- NVIDIA [native arithmetic throughput and Ampere architecture](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#arithmetic-instructions).
- NVIDIA [TF32 matrix-mode semantics](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#alternate-floating-point).
- NVIDIA [GA10x architecture and supported Tensor Core formats](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf).
