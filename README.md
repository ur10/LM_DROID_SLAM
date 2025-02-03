# LM-Droid SLAM

LM-Droid SLAM is a modified version of the state-of-the-art DROID-SLAM system that replaces the dense bundle adjustment (DBA) layers with a Levenberg–Marquardt (LM) solver. This integration aims to enhance accuracy and stability in camera pose and inverse depth estimation by leveraging the robustness of the LM algorithm, while keeping the overall system fully differentiable.

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Methodology](#methodology)
- [Experiments and Evaluation](#experiments-and-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations and Future Work](#limitations-and-future-work)
- [Citation](#citation)
- [Contact](#contact)



## Overview



![Screenshot from 2025-02-03 10-35-25](https://github.com/user-attachments/assets/4a74a07d-90eb-4183-a9e9-caafc2aadac0)

LM-Droid SLAM modifies DROID-SLAM by replacing the learned DBA layers—which use Schur complement and Gauss–Newton optimization—with an LM solver. The LM algorithm, known for its adaptive behavior and robustness even with poor initial estimates, refines camera poses and inverse depth estimates based on the same reprojection error framework as the original system.

## Motivation

- **Enhanced Robustness:** LM’s adaptive combination of gradient descent and Gauss–Newton makes it more stable in challenging optimization scenarios.
- **Efficiency Analysis:** Detailed timing analyses reveal that while the DBA layer is a performance bottleneck (~7.5% of total time), the graph aggregation and update operators account for the bulk of processing time. This motivates exploration of more efficient architectural alternatives.
- **Differentiable Integration:** Maintaining an end-to-end differentiable pipeline, the LM solver is integrated in a way that supports backpropagation, paving the way for potential end-to-end training improvements.

## Methodology

- **LM Solver Implementation:**  
  The project replaces the DBA layer with an LM solver that directly minimizes the reprojection error. The solver computes the necessary Jacobian matrices and dynamically adjusts the damping factor to ensure stable convergence.

- **Integration into DROID-SLAM:**  
  Only the update operator is modified to incorporate the LM-based optimization. The remaining network components—such as the ConvGRU block, feature extraction, and context update modules—remain unchanged to isolate the impact of the LM solver.

- **Optimization Approach:**  
  While the original DBA layer leverages the Schur complement method for efficient solution, the LM solver uses an iterative process that adapts between gradient descent and Gauss–Newton methods depending on local curvature.

## Experiments and Evaluation

- **Datasets:**  
  The system is evaluated on the TartanAir dataset, particularly on sequences such as MH001 and MHOO2, which present a range of challenging synthetic environments.

- **Performance Metrics:**  
  The primary metric is Absolute Trajectory Error (ATE). Experiments showed:
  - DROID-SLAM with DBA: ATE of approximately 0.05 m and 0.04 m.
  - LM-Solver based approach (without retraining the full network): Degraded ATE values of 0.27 m and 0.15 m.

- **Timing Analysis:**  
  Detailed cProfiler experiments on an abandoned factory scene reveal:
  - DBA layer takes about 7.3 seconds (7.5% of total runtime).
  - Graph aggregation and the update operator constitute approximately 30.7% and 35% of the total processing time respectively.
  
  These measurements highlight significant opportunities to optimize the context aggregation and update processes further.


