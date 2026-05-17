# Verification of Woods' Conjecture for Dimension 11

This repository contains the computational infrastructure used to verify Woods' Conjecture for dimension 11. The framework utilizes rigorous interval arithmetic and a global Branch-and-Bound optimizer to exhaustively prune the parameter space of unit-determinant well-rounded lattices, isolating the trivial cubic lattice singularity.

## Prerequisites

The verification script requires Python 3 and the `mpmath` library for exact, arbitrary-precision interval arithmetic.

To ensure strict reproducibility, please use the pinned version of `mpmath` provided in `requirements.txt`.

bash
pip install -r requirements.txt


## Running the Verification

The entire verification pipeline is self-contained within the `verify_m11.py` script. The script relies on Python's `multiprocessing` library to distribute the workload across available CPU cores.

To execute the full verification suite:

bash
python verify_m11.py


### Partial Execution

The search space is partitioned into 1,024 initial root configurations. To facilitate parallel execution across clusters or to re-run specific subtrees, the script supports processing specific ranges of these root boxes via command-line arguments.

To evaluate a specific slice (e.g., from index 0 up to, but not including, 50):

bash
python verify_m11.py 0 50


## Outputs and Telemetry

By default, the script creates an `outputs/` directory in the current working directory, containing:

*   **`execution.log`**: A continuous log of the solver's progress.
*   **`certificates/`**: A directory containing granular JSON telemetry certificates (`subtree_{box_idx}.json`) for each of the 1,024 root domains. These certificates detail the nodes explored, maximum depth, and specific pruning metrics for each sub-domain.
*   **`datasets/surviving_boxes.json`**: The dataset of the final bounded configurations that survived pruning and stalled strictly within the analytically verified continuous neighborhood of the trivial cubic limit.
*   **`datasets/execution_telemetry.json`**: An aggregate summary payload containing the overall search tree dynamics, pruning breakdowns, and resource efficiency metrics.

## Hardware Reference

The metrics reported in the manuscript were recorded using the following hardware topology:
*   Processor: Dual-Core Intel Core i5 (1.6 GHz)
*   System Memory: 8 GB RAM
*   Elapsed Wall-Clock Time: ~74.6 hours
