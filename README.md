# Introduction

This repository contains codes that are used for generating numerical results in the following paper: 

W. U. Mondal, V. Aggarwal, and S. V. Ukkusuri, "Can Mean Field Control (MFC) Approximate Cooperative Multi Agent 
Reinforcement Learning (MARL) with Non-Uniform Interaction?", Conference on Uncertainty in Artificial Intelligence (UAI), 
Eindhoven, Netherlands, 2022.

```
@inproceedings{mondal2022can,
  title={Can Mean Field Control (MFC) Approximate Cooperative Multi Agent Reinforcement Learning (MARL) with Non-Uniform Interaction?},
  author={Mondal, Washim Uddin and Aggarwal, Vaneet and Ukkusuri, Satish},
  booktitle={The 38th Conference on Uncertainty in Artificial Intelligence},
  year={2022}
}
```
 
ArXiv: https://arxiv.org/abs/2203.00035


# Parameters

Various parameters used in the experiments can be found in [Scripts/Parameters.py](https://github.com/washim-uddin-mondal/UAI2022/blob/main/Scripts/Parameters.py) file.

# Software and Packages

```
python 3.8.12
pytorch 1.10.1
numpy 1.21.2
matplotlib 3.5.0
```
# Results

Generated results will be stored in Results folder (will be created on the fly).
Some pre-generated results are available for display in the Display folder. Specifically, 
[Fig. 1](https://github.com/washim-uddin-mondal/UAI2022/blob/main/Display/Fig1.png) depicts the percent error
as a function of N (the number of agents) for an affine reward function. On the contrary, 
[Fig. 2a](https://github.com/washim-uddin-mondal/UAI2022/blob/main/Display/Fig2a.png) and 
[Fig. 2b](https://github.com/washim-uddin-mondal/UAI2022/blob/main/Display/Fig2b.png) depict the 
percent error for non-linear rewards with non-linearity parameters σ=1.1, 1.2 respectively.

# Run Experiments

```
python3 Main.py
```

The progress of the experiment is logged in Results/progress.log

# Command Line Options

Various command line options are given below:

```
--train : if training is required from scratch, otherwise a pre-trained model will be used  
--sigma : non-linearity parameter of the reward  
--K : number of neighbours (local interactions)  
--minN : minimum value of N (must exceed K)  
--numN : number of N values  
--divN : difference between two consecutive N values  
--maxSeed: number of random seeds 
```