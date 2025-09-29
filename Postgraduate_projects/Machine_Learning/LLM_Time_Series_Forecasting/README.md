# M2 Coursework: Repository Structure, Installation, and Usage Guide

- `report/`: Contains the final coursework report.
- `src/`: Contains source code files corresponding to each question/task in the coursework.
- `scripts/`: Contains a master script to run all experiments.

Each source file in `src/` corresponds to a particular question in the coursework:

- **`q2_1_visualise_data.py`**: Plots histograms to visualize prey and predator population counts.
- **`q2_2_example_sequence_evaluate_model.py`**: Tokenizes sequences, prints example sequences, and evaluates an untrained model using MSE and MAPE on test data.
- **`q3_1_1_train_base_model.py`**: Trains the base LoRA-wrapped Qwen model for 500 steps and evaluates it using MSE, MAPE, and cross-entropy.
- **`q3_1_2_untrained_base_model.py`**: Evaluates the untrained base model using MSE, MAPE, and cross-entropy on test data.
- **`q3_2_tuning_learnrate_rank.py`**: Performs hyperparameter tuning for learning rate and rank over 500 training steps.
- **`q3_3_tuning_context_length.py`**: Tunes context length using the best learning rate and rank over 500 steps.
- **`q3_4_final_model.py`**: Trains the final model using the best hyperparameters on all 1000 sequences for 3000 steps.
- **`q3_5_flopcalculations.py`**: Computes and prints the total number of FLOPs for each experiment.

---

## Installation

### Manually install (recommended)

1. Clone the repository:
```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/m2_coursework/jn492.git
```

2. CD into folder
```bash
cd jn492
```

2. Install dependencies: 
```bash
pip install -r requirements.txt
```

### Use docker (Docker does not support GUI-based plot windows, so you won’t be able to view plots interactively)

docker build -t m2_coursework .

## Run script

You can either 1. run script one source file at a time manually (recommended), 2. run it all at once manually, 3. run files one at a time using docker 4. run all with docker

### Option 1: Run source files individually manually (recommended)

```bash
python src/q2_2_example_sequence_evaluate_model.py
python src/q3_1_1_train_base_model.py
python src/q3_1_2_untrained_base_model.py
python src/q3_2_tuning_learnrate_rank.py
python src/q3_3_context_length.py
python src/3_4_final_model.py
python src/q3_5_flopcalculations.py
```

### Option 2: Run all source files all at once manually

```bash
python script/run_all.py
```

### Option 3: Run it individually with Docker

```bash
docker run -it m2_coursework /bin/bash

python src/q2_1_visualise_data.py
python src/q2_2_example_sequence_evaluate_model.py
python src/q3_1_1_train_base_model.py
python src/q3_1_2_untrained_base_model.py
python src/q3_2_tuning_learnrate_rank.py
python src/q3_3_context_length.py
python src/3_4_final_model.py
python src/q3_5_flopcalculations.py
```

### Option 4: Run all with Docker

```bash
docker run -it m2_coursework /bin/bash
python script/run_all.py
```

## Contributions
Please feel free to fork this repository and submit a pull request!

## License
This project is licensed under the MIT License, see license.txt.


