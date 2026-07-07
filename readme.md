# Reproduction Guide for [GSAR-Tree]

This document explains the data, code, and related materials provided to support the findings of our study, as well as the complexities involved in achieving full reproduction.

To ensure the **transparency, verifiability, and scientific rigor** of our research, we provide the following core materials:

## 1. Core Components Provided
We provide all the fundamental building blocks that demonstrate the validity of our approach:

*   **Relatively Complete Algorithm Implementation:** Includes the core training algorithms (e.g., in the `RLProcess/` directory), accompanied by detailed code comments (the core state design is partially hidden and explained in the form of comments).
*   **Data:** Provides scripts for generating synthetic data, downloading, and processing real data, with detailed comments (`data/`).
*   **Model:** Provides the implementation of the underlying R-Tree model, the key state design for fuzzy processing, and scripts for compilation (model/, rtree_cppout/ .bash).

## 2. How to Verify Our Results
We provide a complete, runnable example, including the RL algorithm part, C++ compilation files, and a well-trained model `train_TW1_NORMAL_rstar-tree_50_20_BestModel.pth`.

**Model Name Explanation:** 
*   `train_TW1`: Represents the use of a training dataset with 100,000 data points.
*   `rstar-tree`: Represents the use of R*-tree as the baseline.
*   `50`: Represents a node capacity of 50.
*   `20`: Represents training for 20 epochs.

**Execution method and file content overview:**
`gsar_tree.py` calls the tree class based on the compiled C file to interact with the RL agent, which uses AC and PPO.

```bash
python RLProcess/gsar_tree.py --welltrained True    # for testing the well-trained agent model
python RLProcess/gsar_tree.py --mode 1              # for training
python RLProcess/gsar_tree.py                       # for testing
```

*   **Trained RL agent model:** The example uses synthetic normal data.
*   **Base tree implementation:** Includes the basic class structure, operations for building a tree, retrieving state information of nodes' subtrees, etc.

Finally, a bar chart comparing the run results will be displayed. This demonstrates that our code is executable, the training logic is correct, and it produces the expected behavioral trends.

## 3. Statement on Full Experimental Reproducibility

We have trained a example model based on the source code that can be run test directly with one click to generate result charts. The running parameters are summarized in its name `train_TW1_NORMAL_rstar-tree_50_20_BestModel.pth` (this is one instance from our numerous combinations).

This minimal working example can verify that our method's training pipeline is correct and effective, and can reproduce performance trends consistent with those in the paper.

We fully recognize the importance of complete reproducibility and have prioritized refactoring the codebase and implementing strict version management to better serve the academic community in the future.
