# Amortized Rejection Sampling in Universal Probabilistic Programming

Source code for the paper Naderiparizi, S. et al. (2022). Amortized Rejection Sampling in Universal Probabilistic Programming. In _The 25th International Conference on Artificial Intelligence and Statistics_.

---
We show that Naive approaches to amortized inference in probabilistic programs that explicitly include rejection sampling loops can produce estimators with infinite variance.
We develop Amortized Rejection Sampling (ARS), a new and efficient amortized importance sampling estimator for such programs.
We prove finite variance of our estimator and empirically demonstrate our method’s correctness and efficiency compared to existing alternatives on probabilistic programs containing rejection sampling loops and discuss how to implement our method in a generic probabilistic programming framework.

## This repositry

This repository contains the source code for the experiments in the paper. All the experiments are implemented in PyProb. Moreover, this repo has a fork of PyProb that includes our implementation of ARS as a submodule. For each experiment, we provide the model (located at `<experiment-name>/model.py`), the hyper-parameters (located under `configs/`), an inference script (`gen_weights.py`) and a plotting script (located under `plotting/`). Here we show how to run each experiment.

### Marsaglia experiment

- __Training__: In order to train the model, run the following command. It trains a network and stores it under `out/marsaglia/`.
    
    ```bash
    python train.py marsaglia --arg_file configs/marsaglia_training.json
    ```

- __Generating the weights__: We provide the script `gen_weights.py` that uses PyProb to generate program traces under different weighting schemes, extracts their weights and stores them on disk. The stored weights are later used for making the plots. Since the traces are generated in an i.i.d. fashion, we can run this script in parallel on multiple nodes.
    
    ```bash
    python gen_weights.py marsaglia --arg_file <config-file> --iw_mode <iw-mode> --num_traces <number-of-traces>
    ```

    where 
      - `<config-file>` is the path to the configuration file (either of `configs/marsaglia_a.json` or `configs/marsaglia_b.json` corresponding to the the different observations).
      - `<iw-mode>` is the weighting scheme corresponding to the different methods considered in the paper. It sohuld be either of `prior`, `ic`, `biased`, or `ars-<M>` where `<M>` is the number of samples used in the ARS algorithm (ours).
      - `<number-of-traces>` is the number of program traces (and their importance weights) to generate.
- __Plotting__: To plot the results like Fig. 2 in the paper, once the weights are generated, run the following:
    
    ```bash
    python plotting/plot_marsaglia.py
    ```

### Mini-SHERPA experiment

- __Training__: Similar to [the Marsaglia experiment](#marsaglia), in order to train the model, run the following command. It trains a network and stores it under `out/mini_sherpa/`.
    
    ```bash
    python train.py mini_sherpa --arg_file configs/mini_sherpa_training.json
    ```

- __Creating observations__: Before performing inference, we generate some outputs from the simulator to be used as the observation in the inference algorithm. Run the following command to generate the observations.

    ```bash
    python create_observation.py mini_sherpa --arg_file <config-file>
    ```

    where `<config-file>` is either of `configs/mini_sherpa_a.json`, `configs/mini_sherpa_b.json` or `configs/mini_sherpa_c.json`. These config file correspond to respectively the events of channel 1, 2, and 3 presented in the paper.
- __Generating the weights__: Use the same script as in [the Marsaglia experiment](#marsaglia) to generate program traces and store their weights.

    ```bash
    python gen_weights.py mini_sherpa --arg_file <config-file> --iw_mode <iw-mode> --num_traces <number-of-traces>
    ```

    where `<config-file>` is the path to the configuration file (either of `configs/marsaglia_a.json` or `configs/marsaglia_b.json` corresponding to the the different observations).
- __Plotting__: Once the weights are generated, run the following:
    
    ```bash
    python plotting/plot_mini_sherpa.py
    ```

### Beta experiment

- __Generating the weights__: Use the same script as in [the Marsaglia experiment](#marsaglia) to generate program traces and store their weights.

    ```bash
    python gen_traces.py beta --arg_file <config-file> --num_obs <n>
    ```

    where `<config-file>` is either of `configs/beta_a.json` or `configs/beta_b.json`, and `<n>` is the number "True" observations from the model. This is the parameter controlling how far the prior and posterior are from each other. To get the results in the paper, run this script for `1<=n<=30`.
- __Plotting__: Once the weights are generated, run the following:
    
    ```bash
    python plotting/plot_beta.py
    ```