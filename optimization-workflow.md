---
description: Evaluate code for potential optimizations using the MANTRA Optimization Best Practices.
---

// turbo-all

1.  **Analyze the Context**:
    - **Goal**: What is the bottleneck? (I/O latency, Compute, Memory, or Python Overhead?)
    - **Scope**: Is this a bottleneck kernel (inner loop) or a logistical script (data mover)?
    - **Dataflow Analysis**: Identify all dataflows in the script and pinpoint exactly how your changes will disrupt/alter them.

2.  **Audit against `OPTIMIZATION_BEST_PRACTICES.md`**:
    Systematically check the code against the [Core Pillars](/home/machina/MANTRA/docs/OPTIMIZATION_BEST_PRACTICES.md):
    - **Vectorization (Scalar Exhaustion)**:
      - [ ] deep-scan: Are there explicit Python `for` loops over cells or perturbations?
      - [ ] _Fix_: Replace with broad-spectrum NumPy/Torch ops or Sparse Group Matrices (`G @ X`).

    - **Matrix Strategy (CSR vs Dense)**:
      - [ ] check: Is the matrix $>80\%$ zeros? If so, is it `sp.csr_matrix`?
      - [ ] check: Are we doing column-slicing on CSR? (_Catastrophic_ $O(N)$).
      - [ ] _Fix_: Use `csc_matrix` for col-slicing, or densify only the active batch.

    - **I/O & Memory**:
      - [ ] check: Are we calling `.to_memory()` on full datasets?
      - [ ] check: Is `joblib` duplicating memory? (_Fix_: Use `prefer="threads"` for read-only).
      - [ ] _Fix_: Use `backed="r"` with streaming aggregation for large-scale stats.

    - **GPU Acceleration**:
      - [ ] check: Are we doing dense linear algebra (Ridge, NNLS) on CPU?
      - [ ] check: Are we transferring data in tiny chunks? (_Fix_: Batch transfer).

3.  **Propose an Optimization Plan**:
    - **Strategy**: Select the specific class (A=I/O, B=Caching, C=Vectorization, D=Parallelism, E=GPU).
    - **Impact Estimate**: Estimate speedup (e.g., "10x via Vectorization").
    - **Risk**: Identify potential numerical instability or OOM risks.

4.  **Implement & Verify (Safety First)**:
    - **Numerical Parity**:
      - Run the "Gold Standard" (unoptimized) vs "Candidate" (optimized).
      - Assert `np.allclose(actual, expected)`.
      - **Parity Scripts**: Design thorough parity scripts that ingest real data and enforce a plethora of hard invariants.
    - **Invariant-Maintenance Testing**:
      - [ ] **Design**: Create stringent tests that verify core logic remains invariant under optimization (e.g., bin assignment consistency, gradient flow, coordinate system stability).
      - [ ] **Real-Data Validation**: Execute tests against the optimized scripts using real-world datasets, not just toy examples.
      - **Reference Tests**:
        - `src/mantra/programs/tests/test_gated_dataset_pressure.py`
        - `src/mantra/programs/tests/test_pooled_cond_mem_offsets.py`
        - `src/mantra/programs/tests/test_soft_attn_chunked.py`
    - **Determinism**: Ensure threading/GPU doesn't introduce non-determinism.
    - **Speed Benchmarking**: Implement speed benchmarking to ensure the juice was worth the squeeze.

5.  **Mandatory Git Synchronization**:
    - **CRITICAL**: Once the optimization is verified and documented, you MUST perform a git sync to protect the improved code:

    ```bash
    git add .
    git commit -m "Optimize code: [Component] - [Strategy]"
    git push
    ```

6.  **Output**:
    - Present the **Before/After** metrics.
    - Cite the specific section of `OPTIMIZATION_BEST_PRACTICES.md` used.


## ABSOLUTE NON-DESTRUCTION POLICY
**CRITICAL**: You are PROHIBITED from deleting any file in this workspace. Refer to [AGENTS.md](file:///home/machina/MANTRA/AGENTS.md) for enforcement details. No `rm`, no `git rm`, no `git clean`.
---

**COMMUNICATION RULE**: Strictly avoid parenthetical commentary, especially narrative, process-oriented, or redundant restatements. Parentheticals are permitted only for acronym expansion or to provide essential mathematical context.
