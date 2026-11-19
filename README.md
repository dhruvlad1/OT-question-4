# Optimization Techniques Lab – Q4 (ML + Quantum + Parallel Computing)
 
**Name:** Dhruv Lad  
**Student ID:** 241070037  
**Notebook:** `Dhruv_Lad_241070037_OT_Lab_Q4.ipynb`
 
## Problem
A power network of 10 nodes with weighted links must be split into two islands of 5 nodes each, minimizing the total weight of links crossing between the islands. Link weights depend on predicted node failure probabilities.
 
## How to run (Google Colab)
1. Open [colab.research.google.com](https://colab.research.google.com) and choose **File → Upload notebook**.
2. Upload `Dhruv_Lad_241070037_OT_Lab_Q4.ipynb`.
3. Use a standard **CPU** runtime (no GPU needed).
4. Click **Runtime → Run all**. No manual changes are required.
The first code cell installs `dimod` and `dwave-neal` only if they are missing, so Colab needs internet access. NumPy, pandas and scikit-learn are preinstalled.
 
## Notebook structure
| Part | Marks | Content |
|------|-------|---------|
| A – Machine Learning | 1 | 600 synthetic samples (seed 37), `GradientBoostingClassifier`, 70/30 split, accuracy/precision/recall, failure probabilities for 10 nodes, connected weighted network |
| B – QUBO | 1 | Binary variables, symmetric 10×10 matrix Q, penalty λ, constant offset, energy check against the original objective |
| C – Quantum Computing | 1.5 | Simulated annealing (`dwave-neal` + `dimod`, CPU simulation), brute-force comparison over all 1024 configurations |
| D – Parallel Computing | 1 | `ProcessPoolExecutor` with 1, 2, 4 workers, timing table, speed-up and efficiency |
| E – Conclusion | 0.5 | Exactly 3 lines summarizing the computed results |
 
## Key design choices
- **Edge weights:** `w_ij = 1 + 10·(p_i + p_j)/2`, where `p` is the predicted failure probability. Links between failure-prone nodes cost more to cut; the constant 1 keeps weights positive.
- **QUBO convention:** `E(x) = xᵀQx + constant`, with x = 0 for Island A and x = 1 for Island B.
  - `Q_ii = d_i − 9λ`, where `d_i` is the sum of weights at node i
  - `Q_ij = Q_ji = −w_ij + λ`
  - `constant = 25λ`
- **Penalty:** `λ = (sum of all link weights) + 1`. Any invalid split then has higher energy than the best valid 5–5 split.
- **Label swap:** swapping Island A and Island B gives the same partition, so comparisons are made up to this swap.
- **Colab-safe multiprocessing:** the worker function is written to a temporary module `ot_worker.py` and imported, because functions defined inside notebook cells often cannot be pickled for worker processes.
- **Timing:** best of 3 runs per worker count, measured with `time.perf_counter()`, including process start-up.
  - Speed-up = T1 / Tp
  - Efficiency = Speed-up / p
## Built-in checks (assertions)
- Weight matrix and Q are 10×10 and symmetric.
- The network is connected.
- QUBO energy equals the original objective for all 1024 configurations.
- There are exactly C(10,5) = 252 valid 5–5 configurations.
- The reported annealing solution has exactly 5 nodes per island.
- Every worker count finds the same minimum energy as brute force.
## Notes and limitations
- Annealing is a **classical CPU simulation** of quantum annealing, not real quantum hardware. It is a heuristic and does not guarantee the exact optimum. The notebook reports whether it matched brute force in that run.
- With only 1024 cheap evaluations, process start-up overhead usually outweighs any gain, so parallel runs may be slower than 1 worker. All timings are measured at run time, not hardcoded.
- Results depend on the Colab machine, so timings will vary between runs.
- The notebook was tested end to end only in a local sandbox using stand-in versions of `dimod` and `neal`. It was not run on Colab or with the real packages. If a Part C cell errors on Colab, that is where to look first.
## Files
- `Dhruv_Lad_241070037_OT_Lab_Q4.ipynb` – the notebook
- `README.md` – this file
- `ot_worker.py` – created automatically when Part D runs (not part of the submission)
 