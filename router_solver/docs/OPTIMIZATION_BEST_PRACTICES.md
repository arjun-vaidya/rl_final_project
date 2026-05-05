# MANTRA Optimization Best Practices

**Updated**: 2026-04-20  
**Scope**: project-wide performance and memory optimization playbook  
**Goal**: preserve correctness while removing time and memory bottlenecks

> [!WARNING]
> Do not optimize blindly. Measure first, then change one variable at a time, then verify parity.

---

## Executive Summary

MANTRA performance improvements cluster into a small number of repeatable classes:

1. **Algebraic reframing** to reduce dimensionality and passes
2. **Parsimony-first data access** to avoid loading unused genes and intermediates
3. **GPU residency and zero-copy reuse** to eliminate PCIe churn and repeated allocations
4. **Kernel fusion in Triton** to avoid materializing large intermediates and reduce bandwidth pressure
5. **High throughput I/O** to survive watchdog environments using sorted access patterns
6. **Storage-contract optimization** for workflows where source parsing dominates runtime
7. **Sparse-first training kernels** to remove dense discovery intermediates
8. **Vectorized reductions** to remove Python overhead in training and evaluation loops

This document is the clean summary of those patterns, with pointers to the primary white papers and the concrete source files that implement them.

---

## Foundational Mechanics

These are the mechanics that keep showing up in every win. They are not MANTRA specific. MANTRA is simply an aggressive application of them.

### Vectorization

Vectorization replaces explicit iteration with operations that act on whole tensors.

Why it wins:

- it removes Python interpreter overhead
- it activates kernel level fusion paths in PyTorch
- it converts fragmented work into contiguous memory access

Where it appears:

- bin means and per bucket reductions using scatter add
- masking and top k selection expressed as tensor ops

### Parallelism

Parallelism runs independent work concurrently.

Rules of thumb:

- CPU cores are good for orchestration and heavy I O
- GPU cores are good for dense math and repeated arithmetic over large tensors

Where it appears:

- GPU scoring and projection
- bootstrap loops that keep invariant tensors resident on device

### Batching

Batching is logistical efficiency.

Why it wins:

- PCIe latency punishes many small transfers
- high bandwidth memory needs a large workload to saturate throughput

Best practice:

- choose batch sizes that keep GPU occupancy high
- stage host reads so the GPU is never starved

### Latent space arithmetic

This is a project signature.

If you need a weighted gene space combination and then a projection, use linearity and do it in program space.

This identity is defined once in Optimization Class 1 and reused throughout the project.

Impact:

- replace gene dimension work with program dimension work
- make soft binning and conditioning feasible at atlas scale

Primary references:

- `docs/GOD_FLOW_WHITE_PAPER.md`
- `docs/GPU_OPT.md`

---

## Canonical Principles

### Measure, then optimize

- Establish a baseline wall clock for each phase using explicit timers.
- Profile only after you have a stable baseline.
- Use the smallest unit test that reproduces the slowdown.

### Reduce passes before you micro-optimize

Each full pass over a matrix has a fixed cost dominated by memory bandwidth and cache misses. Removing one pass usually beats making one pass faster.

### Change the storage contract when parsing dominates

If a workflow repeatedly replays gzip decompression, ASCII tokenization, dense-to-sparse conversion, or COO-to-CSR sorting, the fastest kernel is usually the wrong fix.

Best practice:

- measure decompression, parse, transfer, sparse construction, and load phases separately
- convert source text into the exact binary structure the consumer needs
- preselect the realized cell or feature surface at conversion time when the surface is stable
- pre-sort sparse arrays into the runtime orientation so reads need only array loads and a matrix constructor

### Keep data on device across loops

If a loop runs many iterations, allocate and upload once, reuse buffers, and avoid per-iteration host sync.

### Prefer stable numerics over fragile speed

- Use float16 for storage and bandwidth.
- Accumulate in float32.
- Apply clamping and nan guards where needed.

### Versioning rule

When changing a critical pipeline, clone the original script into a new versioned file name and keep the previous version runnable.

---

## Optimization Classes

### 1. Algebraic reframing and latent-space arithmetic

This class turns gene-space work into program-space work.

**Key pattern**: project first, then interpolate.

If you need a weighted combination of gene vectors and then a projection, use linearity:

$$
\left[\sum_i w_i x_i\right] P = \sum_i w_i \left[x_i P\right]
$$

**Impact**: replace gene dimension work with program dimension work, often a forty fold reduction when genes are around two thousand and programs are around fifty.

**Where it appears**:

- Audit target construction and soft binning aggregation
- Precomputed treated projection reused across bootstrap iterations

Primary references:

- `docs/GOD_FLOW_WHITE_PAPER.md`
- `docs/GPU_OPT.md`

### 2. Parsimony-first data access

Load only what you need, when you need it.

Rules:

- Keep full gene space only where it is required.
- Materialize a delta gene subset for projection when the target space is defined on that subset.
- Avoid forming dense intermediates that scale with cells by programs unless you can reuse them.

Concrete examples:

- Persist a float16 delta matrix in VRAM for all treated cells.
- Keep control bin means computed from controls, but do not keep the full cells by genes matrix in VRAM unless it is necessary.

Primary references:

- `docs/GPU_OPT.md`
- `docs/v14_SPEED_RUN_WHITE_PAPER.md`

### 3. GPU residency, zero-copy reuse, and buffer discipline

This class eliminates repeated transfers and allocations.

Rules:

- Allocate once outside the loop.
- Reuse output buffers across iterations.
- Keep invariant tensors resident on GPU across bootstraps.
- Convert to float16 for residency, cast to float32 for compute hot spots.

Canonical examples in MANTRA:

- Persist scores, labels, centers, projection matrix, and treated projection across bootstraps.
- Use dataset objects that build persistent buffers once and expose views for downstream phases.

Primary references:

- `docs/GPU_OPT.md`
- `src/mantra/programs/v6/gated_data_v7.py`

### 4. Kernel fusion in Triton

This class fuses operations that were previously multiple GPU kernels plus intermediate tensors.

When to use Triton:

- An intermediate is large and short-lived.
- You have a tight loop with repeated gather and scatter patterns.
- You are bandwidth bound and kernel launch bound.

Core patterns implemented in MANTRA:

- **Triton KMeans assignment**  
  Fused distance and argmin without a full distance matrix.
  Reference: `docs/TRITON_KMEANS_KERNEL.md`

- **GPU-resident KMeans loop**
  Fully asynchronous KMeans implementation that eliminates all CPU-GPU synchronization (normalization, relocation, and loop control).
  Reference: `src/mantra/programs/v6/kmeans_singularity_gpu.py`

- **Fused centered projection**  
  Computes `(X - Mu[labels]) @ P` inside the Triton kernel, so the dense `X_centered` matrix is never materialized. Use this when a large cell-by-gene matrix needs bin-centered projection into a smaller program space.
  Reference: `docs/GOD_FLOW_WHITE_PAPER.md`

- **Fused soft-bin residual aggregation**  
  Computes the neighbor-weighted `Mu_proj` correction, subtracts it from projected cell coordinates, and accumulates perturbation-by-bin outputs in one pass. This replaces a random neighbor gather, a residual tensor, and a separate groupby aggregation.
  Reference: `docs/GOD_FLOW_WHITE_PAPER.md`

- **Fused refit aggregation**  
  For refit audits where centers change per bootstrap, computes center-distance weights and residual aggregation in the same launch. The explicit audit labels remain the grouping key; the refit prediction only supplies correction weights.
  Reference: `docs/GOD_FLOW_WHITE_PAPER.md`

Kernel design rules:

- Avoid materializing a cells by bins distance matrix when possible.
- Use min-shifted exponentials for stable soft weights.
- Accumulate in float32 even if inputs are float16.
- Reduce atomic pressure using block-local reductions when possible.

Primary references:

- `docs/GOD_FLOW_WHITE_PAPER.md`
- `docs/TRITON_KMEANS_KERNEL.md`
- `src/mantra/programs/v6/soft_bin_gpu.py`
- `src/mantra/programs/v6/kmeans_gpu.py`

### 5. High throughput I/O

This class optimizes large reads that must finish within a bounded runtime window.

Core idea: sorted fancy indexing in HDF5 is drastically faster than unsorted access.

Rules:

- Sort requested indices before the read.
- Perform a one-shot read on sorted indices.
- Permute back in memory after the read.
- If needed, mask interrupts for the brief I/O burst.

Primary reference:

- `docs/v14_SPEED_RUN_WHITE_PAPER.md`
- `src/mantra/utils/fast_h5_loader.py`

### 6. Storage-contract optimization for ingestion

This class applies when the source format is optimized for distribution rather than repeated computation.

Core idea: if a workflow repeatedly pays for gzip decompression, ASCII tokenization, dense-to-sparse conversion, or COO-to-CSR sorting, change the persisted representation so runtime reads the structure it actually consumes.

Rules:

- Do not repeatedly tokenize raw gzip text in training, benchmarking, or surface construction loops.
- Do not materialize full decompressed shards in `/dev/shm` unless the measured decompressed size is below available tmpfs space with margin.
- Do not store a dense binary cache when the realized matrix is sparse and the consumer needs sparse rows.
- Do not rely on GPU kernels to rescue a bad transfer contract that ships mostly zeros across PCIe.
- Write the cache in the final row order when that order is known, including any inverse permutation needed by consumers.
- For shared source files that fan out into multiple logical shards, read the source once and emit all child shards in one pass.
- Record source paths, selection cache path, selected cell count, gene count, nonzero count, dtype, compression, wall time, and disk footprint next to the cache.
- Preserve a slow reference path and verify shape, gene order, nonzero count, and sampled element parity.

Implementation pattern:

1. Read the source gzip once.
2. Parse raw counts into integer arrays.
3. Select the stable cell surface during conversion.
4. Accumulate only nonzero triples.
5. Sort once by runtime row id.
6. Persist `csr_data`, `csr_indices`, and `csr_indptr` with fast binary compression.
7. Load with direct CSR construction, no dense allocation, no COO intermediary, and no sort.

Where it appears:

- `hoptf/scripts/convert_tf_atlas_to_zarr_csr.py`
- `hoptf/scripts/convert_tf_atlas_s01_s04_to_zarr_csr.py`
- `hoptf/src/hoptf/tf_atlas_streaming_surface_binary.py`
- `hoptf/data_contracts/tf_atlas/H5AD_SHARD_CONTRACT.md`
- `hoptf/data_contracts/tf_atlas/PREPROCESSING_CONTRACT.md`

### 7. Sparse-first training to remove the discovery memory wall

This class removes the dense intermediate that scales with cells by programs during training and discovery.

Core idea: do not materialize X times W. Compute sparse dot products on the fly inside the solver loop.

In MANTRA this is the Ghost solver, a fused sparse NNLS kernel that iterates over CSR rows directly.

Primary reference:

- `docs/GOD_FLOW_WHITE_PAPER.md`
- `src/mantra/programs/v6/ghost_solver.py`

### 8. Vectorized training and evaluation to remove the Python overhead wall

This class removes Python level loops that scale with cell count and turn GPU epochs into CPU bound bookkeeping.

Core idea: everything that looks like per example aggregation should be expressed as a vector operation over an index key, then reduced by scatter add on device.

Primary reference:

- `src/mantra/programs/docs/GATING_OPTIMIZATION_WHITE_PAPER.md`

#### 7.1 Vectorized aggregation of norms by bucket

Many gating metrics aggregate norms over a bucket defined by perturbation and state bin.

Instead of iterating over examples and appending to Python dicts, pack a bucket id and use GPU scatter add.

Let the packed key be:

$$
pb_i = p_i \cdot B + b_i
$$

Let S be predicted programs for the batch and T be target programs for the batch.

Aggregate batch norms in one pass:

$$
N_{pb} = \sum_{i \in pb} \lVert S_i \rVert_2
$$

$$
M_{pb} = \sum_{i \in pb} \lVert T_i \rVert_2
$$

In code this is `torch.norm` followed by `index_add_` on the packed key.

#### 7.2 Log norm scale correlation without Python loops

Scale correlation is the Pearson correlation of log mean norms across bins for each perturbation.

The pipeline avoids per perturbation Python loops by computing the correlation formula in batch with masks for inactive bins.

Best practice:

- exclude perturbations with too few active bins so the statistic is meaningful
- log norms with a small epsilon to avoid nan values

#### 7.3 Fast prototype memory bank construction

Training throughput collapses if memory bank construction uses a DataLoader loop.

Instead:

- index directly into the dataset tensors that already live in memory
- build prototypes in one pass using vectorized indexing
- move to GPU in chunks to avoid VRAM spikes

This pattern is implemented in the gating training scripts as a fast memory bank builder, with a tunable chunk size.

#### 7.4 Vectorized stratified splitting with packed keys

Training splits are a common hidden bottleneck when they enforce stratification across multiple axes.

Best practice:

- pack a stratification key into a single int64
- use `np.unique` and argsort to group and slice at O N log N
- keep split determinism and parity checks

This replaces nested Python loops over perturbations, bins, and responder strata.

---

## GPU Acceleration Pipeline

This section reclaims the historical story in a compact form so future work has a map of what mattered.

The canonical arc in v5 is a five phase pipeline that took bootstraps from seconds to fractions of a second while preserving parity.

### Phase map

| Phase    | Innovation                      | Time per bootstrap | Cumulative |
| -------- | ------------------------------- | ------------------ | ---------- |
| Baseline | CPU KMeans and NumPy            | about 12 seconds   | 1          |
| 1        | GPU scoring                     | about 4 seconds    | 3          |
| 2        | GPU KMeans and preload          | about 2 seconds    | 6          |
| 3        | full device residency           | about 1.3 seconds  | 9          |
| 4        | GPU bootstrap loop              | about 0.4 seconds  | 27         |
| 5        | zero copy reuse and GPU metrics | about 0.23 seconds | 50         |

### What to copy when you want the same speedup

- build a dataset object that establishes persistent buffers once
- precompute invariants and reuse them across bootstraps
- never materialize a cells by bins distance matrix
- fuse the hot path in Triton when intermediates are large and short lived

Primary references:

- `docs/GOD_FLOW_WHITE_PAPER.md`
- `docs/GPU_OPT.md`
- `src/mantra/programs/v6/gated_data_v7.py`
- `src/mantra/programs/v6/bin_stability_audit_v8.py`

---

## Implementation Protocol

### Interfaces and shapes

Always document tensor shapes in prose using words, not inline shape tuples, to avoid ambiguity and keep docs readable.

Example:

- X has shape cells by genes
- W has shape genes by programs
- scores has shape cells by state features

### Always ship diagnostics with the optimization

An optimization is incomplete if it does not preserve observability.

Minimum diagnostics to log for major kernels and sweeps:

- wall clock by phase
- VRAM peak
- parity error versus reference on real data using the proposed optimization
- quality metrics relevant to the stage, such as cosine, relative MSE, coverage, ARI, NMI

### Prefer reproducible pipelines

For any SOTA artifact, pin:

- git commit
- dataset path
- program artifact hashes
- script hashes
- exact reproduction commands

Reference:

- `docs/PROJECT_MANIFEST.md`

---

## Profiling Strategy

### Timing first

Use explicit timers around phases and inside long loops.

### CPU profiling

Commands:

```bash
python -m cProfile -o run.prof your_script.py
python -m pstats run.prof
```

### GPU profiling

Rules:

- profile one iteration first
- then profile a steady-state window
- avoid profiling with frequent synchronization inside the hot loop

Tools:

- PyTorch profiler for operator timelines
- NVTX ranges for readable traces

---

## Verification and Parity

### Numerical parity

Define a reference path and test the optimized path against it.

Example parity command pattern:

```bash
python scripts/programs/benchmark/soft_binning_sweep_full_dynamic.py
```

### Determinism

If you claim determinism, set seeds and document which operations remain nondeterministic on GPU.

### Stress tests

Stress tests should target:

- larger K
- large cell counts
- edge cases such as empty bins and low-count groups

---

## Failure Modes and Mitigations

- **Out of memory**  
  Mitigation: parsimony-first gene subsets, float16 residency, chunked compute, avoid dense intermediates.

- **Numerical instability in float16**  
  Mitigation: float32 accumulation, clamping, nan guards, stable exponentials.

- **Watchdog kill during I/O**  
  Mitigation: sorted indexing, one-shot reads, short critical sections.

- **Silent drift from parity**  
  Mitigation: invariant metrics and bit-perfect tests where possible, plus saved parity artifacts.

---

## Workflow Specific Patterns

This section captures optimizations that are easy to forget because they live at the workflow boundary between data and math.

### Gating training pipeline

Core wins:

- vectorized metric computation using packed keys and scatter add
- prototype memory bank construction without DataLoader loops
- stratified splitting without nested Python loops

Primary reference:

- `src/mantra/programs/docs/GATING_OPTIMIZATION_WHITE_PAPER.md`

### Bin stability audits

These are pragmatic rules that have paid off repeatedly.

1. Precompute and cache the delta gene matrix once  
   Avoid repeated densify and slice inside the bootstrap loop.

2. Pick the sparse format that matches the access pattern  
   CSC is best for repeated column slicing. CSR is best for repeated row slicing.

3. Use vectorized bin means  
   Prefer scatter add on GPU or sparse group matrix times dense matrix on CPU.

4. Use MiniBatch KMeans where it is valid  
   It is often sufficient for bootstraps when the score space is stable.

5. Cache invariants  
   Delta gene indices, valid bin masks, and canonical centers should not be recomputed per grid point.

6. Hard bin fast path  
   If tau is absent or M is one, avoid distance computation entirely.

Primary references:

- `src/mantra/programs/v6/bin_stability_audit_v8.py`
- `src/mantra/programs/v6/soft_bin_gpu.py`

---

## Additional Optimization Classes

### 9. Hybrid CPU-GPU Ensembles for Ragged Workloads

This class handles problems where data is partially shared (large, common) and partially ragged (small, unique).

**Key pattern**: Keep the shared component resident on GPU. Stream only the ragged indices or small deltas.

**Where it appears**:

- **Supervisor Fitting**: Logistic Regression where 50k control cells are shared across 5k perturbation problems.
  - $X_{shared}$ stays on GPU.
  - Solver iterates over perturbation indices.
  - Result: Zero data movement during the inner loop.

**Impact**:

- Determines feasibility. Without this, $O(N_{problems} \times N_{shared})$ memory usage causes immediate OOM.

Primary reference:

- `src/mantra/programs/v6/supervised_fitting_gpu.py`

### 10. Compiler-accelerated fusion

This class uses `torch.compile` to fuse standard PyTorch operations when custom kernels are overkill but Python overhead is fatal.

**When to use**:

- You have a tight loop of small matrix operations (e.g., $20 \times 20$ batched inversions or metadata aggregations).
- The "overhead-to-compute" ratio is high ($> 50\%$).
- Maintaining C++ or CUDA extensions is not justified for the workload.

**Core pattern**:
Wrap the inner computational block in a static function and decorate:

```python
@torch.compile(mode="reduce-overhead")
def _compute_block(A, B):
    return torch.einsum('nij,njk->nik', A, B)
```

**Impact**:

- **82x speedup** on Batched Logistic Regression by fusing the Hessian accumulation loop.
- Effectively removes the Python interpreter from the hot path.

Primary reference:

- `src/mantra/programs/v6/supervised_fitting_gpu.py`

---

### 11. Batched Newton-Solvers for Composition Matching

This class addresses the scaling bottleneck of performing thousands of independent Logistic Regression problems against a large, shared control set.

**Key pattern**: Solve $N$ problems in parallel using a batched second-order optimizer (Newton-Raphson) where the negative class is common across all $N$ problems.

**Implementation detail**:

- Use `torch.einsum` to accumulate the shared component of the Hessian across the batch in a single pass.
- Maintain the shared control set resident on GPU.
- Ragged positive sets are concatenated and indexed via a batch ID tensor to avoid padding.

**Impact**:

- Memory optimization: Reduces memory requirements from $O(N_{problems} \times N_{shared})$ to $O(N_{shared})$.
- Speed: Newton's method converges in 5-20 iterations, providing a 10x speedup over sequential `sklearn` runs on L4 GPUs.

**Where it appears**:

- Composition matching for ablation sweeps (`fit_c_cm_shr_gpu`).

### 12. GPU-Resident Regressive Training (Zero-H2D Inner Loop)

This class eliminates the Python-to-GPU synchronization bottleneck in regressive training loops (e.g., training small MLPs or Attention heads within a larger script).

**Core idea**: Move the entire training orchestration—batch indexing, RNG, optimization step, and data—to the GPU.

**Rules**:

- Generate batch indices on-device using `torch.randint` or GPU-resident RNGs.
- Keep the `AdamW` or `SGD` state on-device.
- Avoid all `.cpu()`, `.item()`, or `print(loss)` calls inside the inner step loop.

**Impact**:

- Removes the "latency tax" of PCIe transfers for small batches.
- Prevents the CPU from becoming the bottleneck for low-parameter models.

**Where it appears**:

- `OpAttentionEmbedding` and `OpKitchenSinkMLP` training.
- `student_meta_mlp` regressive fitting.

### 13. Direct Normal Equation Solvers for High-Count Baselines

This class replaces general-purpose iterative solvers with direct linear algebra when the problem structure allows (e.g., Ridge Regression).

**Core idea**: Explicitly solve $(X^T X + \alpha I) W = X^T Y$ using `torch.linalg.solve` on the GPU.

**When to use**:

- Inputs $(X, Y)$ are already GPU-resident.
- Feature dimension $D$ is small enough for $X^T X$ to fit in memory (e.g., $D < 4000$).
- `sklearn` or `scipy.optimize` initialization overhead dominates runtime.

**Impact**:

- Turning a 10-second CPU-bound loop into a sub-millisecond GPU operation.

**Where it appears**:

- Baseline Ridge variants across the attention ablation suite.

### 14. Vectorized Centroid Calculation via Scatter-Add

This class replaces explicit Python loops over perturbations with vectorized GPU operations when computing centroids or means for thousands of groups.

**Key pattern**: Use `torch.scatter_add_` (or `.index_add_`) to aggregate feature vectors into a pre-allocated centroid buffer on the GPU in a single pass.

**Impact**:

- Eliminates $O(N_{perts})$ Python overhead and repeated indexing.
- Provides a **30x speedup** over manual looping for data centering and baseline construction.

**Where it appears**:

- `_compute_centroids_gpu` helper in `run_v456_attention_ablation.py`.

### 15. Unified Device-Aware Meta-MLP Training

This class consolidates redundant, unoptimized training loops into a single, GPU-native helper that handles device placement, training state, and optimization.

**Core idea**: Standardize the regressive training pattern across all "Kitchen Sink" and "Meta-MLP" components.

**Impact**:

- Guarantees GPU residence for small MLP heads.
- Reduces code duplication and prevents `RuntimeError` due to device mismatch during evaluation.

**Where it appears**:

- `_fit_meta_mlp_on_device` helper in `run_v456_attention_ablation.py`.

---

## Case Studies

### Audit acceleration with fused centered projection and refit aggregation

What changed:

- Centered projection computes `(X - Mu[labels]) @ P` without allocating `X_centered`.
- Soft-bin residual aggregation computes correction, residual, and perturbation-by-bin sums in one kernel.
- Refit audits compute center-distance weights inside the aggregator, avoiding a separate prediction or weight-materialization pass.
- Treated projections and core tensors stay resident on GPU across bootstraps.

Primary references:

- `docs/GOD_FLOW_WHITE_PAPER.md`
- `docs/GPU_OPT.md`

### Triton KMeans assignment kernel

What changed:

- Fused distance and argmin, no distance matrix residency.
- Tight kernel that enables high-frequency audits at large cell counts.

Primary references:

- `docs/TRITON_KMEANS_KERNEL.md`
- `src/mantra/programs/v6/kmeans_gpu.py`

### GPU-resident KMeans training loop

What changed:

- Training loop is fully asynchronous. Normalization and relocation reside on the GPU.
- Tiled tensor core assignment uses `tl.dot` for the $X^2 + C^2 - 2XC$ expansion with hardware-aware tiling.
- Centers are updated in the same pass as assignment through atomic addition and block-local reductions where useful.

Impact:

- 4.8x speedup over Scikit-Learn Lloyd at N=500k.
- 8.0x speedup over standard Triton KMeans v1.

Primary references:

- `src/mantra/programs/v6/kmeans_singularity_gpu.py`

### v14 Speed Run for backed H5AD

What changed:

- Sort indices before read.
- One-shot read then permute back.
- Optionally ignore interrupts during the burst.

Primary references:

- `docs/v14_SPEED_RUN_WHITE_PAPER.md`
- `src/mantra/utils/fast_h5_loader.py`

### Sparse solver for discovery and training

What changed:

- Remove dense X times W intermediate.
- Compute sparse dot products inside the solver loop.

Primary reference:

- `docs/GOD_FLOW_WHITE_PAPER.md`
- `src/mantra/programs/v6/ghost_solver.py`

### TFAtlas binary CSR ingestion cache

What changed:

- Replaced repeated gzip ASCII TSV parsing with a one-time binary CSR cache.
- Baked the realized cell selection and inverse order into the cache at conversion time.
- Replaced dense Zarr chunks, per-band fancy indexing, dense-to-CSR conversion, and COO-to-CSR sorting with direct `csr_matrix((data, indices, indptr), shape=...)` construction.
- Added a single-pass S01-S04 converter so the shared 20 GB gzip source is streamed once and fanned out to shards `01`, `02`, `03`, and `04`.

Measured ladder for shard `05`, 5000 genes by 544K selected cells:

| Method | Wall time | Speedup |
| --- | ---: | ---: |
| Baseline gzip plus `np.fromstring` | 222.0 s | 1.0x |
| GPU vectorized parser plus async I/O | 74.0 s | 3.0x |
| Selection-aware dense Zarr plus COO scatter CSR | 20.6 s | 10.8x |
| Pre-sorted CSR binary Zarr | 0.95 s | 234x |

Claim check:

- `222.0 / 74.0 = 3.000x`
- `222.0 / 20.6 = 10.777x`
- `222.0 / 0.95 = 233.684x`, reported as 234x

Correctness checks:

- Nonzero count parity was exact at `69,476,431`.
- Sampled element parity had `max_diff = 0.0`.
- Gene names matched the reference path.

Bottlenecks eliminated:

- Full-shard decompression into `/dev/shm` was unsafe because decompressed ASCII could exceed 24 GB and fail with `ENOSPC`.
- CuPy parsing did not solve the main issue because Python bytes and non-pinned host memory yielded an observed 81 MB transfer in 1.75 s, about 46 MB/s.
- Dense binary storage removed gzip and tokenization but still paid for column scatter, float conversion, and dense-to-sparse construction.
- Direct pre-sorted CSR storage removed dense zeros from the runtime path and turned ingestion into compressed array loads plus CSR construction.

Primary references:

- `hoptf/scripts/convert_tf_atlas_to_zarr_csr.py`
- `hoptf/scripts/convert_tf_atlas_s01_s04_to_zarr_csr.py`
- `hoptf/src/hoptf/tf_atlas_streaming_surface_binary.py`
- `hoptf/data_contracts/tf_atlas/H5AD_SHARD_CONTRACT.md`
- `hoptf/data_contracts/tf_atlas/PREPROCESSING_CONTRACT.md`

---

## Primary Sources

- `docs/GOD_FLOW_WHITE_PAPER.md`
- `docs/GPU_OPT.md`
- `docs/TRITON_KMEANS_KERNEL.md`
- `docs/v14_SPEED_RUN_WHITE_PAPER.md`
- `src/mantra/programs/docs/GATING_OPTIMIZATION_WHITE_PAPER.md`
- `docs/PROJECT_MANIFEST.md`
- `hoptf/scripts/convert_tf_atlas_to_zarr_csr.py`
- `hoptf/scripts/convert_tf_atlas_s01_s04_to_zarr_csr.py`
- `hoptf/src/hoptf/tf_atlas_streaming_surface_binary.py`
- `hoptf/data_contracts/tf_atlas/H5AD_SHARD_CONTRACT.md`
- `hoptf/data_contracts/tf_atlas/PREPROCESSING_CONTRACT.md`
