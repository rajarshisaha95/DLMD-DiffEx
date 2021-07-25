# DLMD-DiffEx
Decentralized optimization over noisy, rate-constrained networks

This public repository is a work in progress for everything related decentraized optimization overnoisy, rate-constrained networks.
Currently, it contains implementations to reproduce the results from the following works:

1. R. Saha, S. Rini, M. Rao and A. Goldsmith, "Decentralized Optimization Over Noisy, Rate-Constrained Networks: How We Agree By Talking About How We Disagree," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 5055-5059, doi: 10.1109/ICASSP39728.2021.9413527 (https://ieeexplore.ieee.org/document/9413527).

2. [Extended journal version]: R. Saha, S. Rini, M. Rao and A. Goldsmith, "Decentralized optimization over noisy, rate-constrained networks: Achieving consensus by communicating differences", arXiv preprint, 2021 (Currently under review) (https://arxiv.org/abs/2010.11292)

The directory labelled MATLAB codes contains decentralized implementations for LASSO and SVM on synthetically generated data. Some details are as follows:

1. DLMD_DiffEx_LASSO.m/DLMD_DiffEx_SVM.m contain the code for obtaining plots of suboptimality gap vs number of iterations.
2. DLMD_DiffEx_SVM_SuccessProbability.m contains the code to obtain success probability as dynamic range of the quantizers is varied.
3. LASSO_cvx_solver.m/SVM_cvx_solver.m contain codes to obtain the solution using interior point methods of CVX (with respect to which suboptimality gaps are plotted).
4. generate_LASSO_dataset.m/generate_SVM_dataset.m contain codes to generate the synthetic datasets.
