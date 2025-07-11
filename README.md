
## Overview
This repository contains the code for the experiments performed in the following paper:

Yuya Hikima, Hiroshi Sawada, and Akinori Fujino,
Guided Zeroth-Order Methods for Nonconvex Stochastic Problems with Decision-Dependent Distributions,
to appear in the Proceedings of the 42nd International Conference on Machine Learning (ICML 2025).

Contents of this repository:
- **README** This file.
- **NTT Software License Agreement** The user must comply with the rules described herein.
- **multproducts_pricing folder** It contains the Python script used in our experiments for the multiple-products pricing application.
- **strategic_classification folder** It contains the Python script used in our experiments for the strategic classification application.

## Requirements
The code was implemented in Python 3.12.2.

## Usage of the code for the multiple products pricing experiments.

> **Data Source:**  
> All CSV files were downloaded from:  
> [https://github.com/Yuya-Hikima/AAAI25-Zeroth-Order-Methods-for-Nonconvex-Stochastic-Problems-with-Decision-Dependent-Distributions/tree/main/data](https://github.com/Yuya-Hikima/AAAI25-Zeroth-Order-Methods-for-Nonconvex-Stochastic-Problems-with-Decision-Dependent-Distributions/tree/main/data)  
> These datasets were used in [Hikima & Takeda, 2024b].

1.  Run the experiment with the following command:

    ```bash
    python3 multproducts_pricing.py 20
    ```

    - Here, `20` specifies the number of simulations (can be changed as desired).

2. The results will be saved in a folder named according to  the corresponding data ID and the input argument.

3. To manually configure parameters, edit the sections labeled `#common settings` or `#settings for each method` in the Python script.

## Usage of the code for the strategic classification application experiments.
> **Data Source:**  
> The file `credit_processed.csv` was downloaded from:  
> [https://github.com/ustunb/actionable-recourse/tree/master/tests](https://github.com/ustunb/actionable-recourse/tree/master/tests)  
> The original dataset was provided by Yeh & Lien (2009), and the processed version was prepared by Ustun et al. (2019).

1. Run the experiment with the following command:

    ```bash
    python3 multproducts_pricing.py 10000 2 20
    ```

    - **First argument** (`10000`): Maximum number of samples (termination condition)  
    - **Second argument** (`2`): Parameter γ in the cost function of strategic agents (see Section 5.2 of the paper)  
    - **Third argument** (`20`): Number of simulation runs

2. The results will be saved in a folder named according to the corresponding data ID and the input arguments.

3. To manually configure parameters, edit the sections labeled `#common settings` or `#settings for each method` in the script.

## Licence
You must follow the terms of the "**NTT Software License Agreement**."
Be sure to read it.

## Author
Yuya Hikima wrote this text.
If you have any problems, please contact yuya.hikima at gmail.com.
