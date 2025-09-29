import subprocess

# These are all the scripts that I have written for the two questions in the coursework.
scripts = [
    "src/q2_1_visualise_data.py", #corresponds to histogram in figure 2
    "src/q2_2_example_sequence_evaluate_model.py", #corresponds to all results in section 2.3
    "src/q3_1_1_untrained_base_model.py", #corresponds to untrained model results in section 3.2.2
    "src/q3_1_2_train_base_model.py", #corresponds to trained model results in section 3.2.2
    "src/q3_2_tuning_learnrate_rank.py", #corresponds to all results in section 3.3.1 and 3.3.2
    "src/q3_3_tuning_context_length.py", #corresponds to results in section 3.3.3
    "src/q3_4_final_model.py", #corresponds to results in section 3.4
    "src/q3_5_flopcalculations.py", #corresponds to flop table in section 3.5
]

print("Running all experiments...")

for script in scripts:
    print(f"Now running: {script}")
    try:
        subprocess.run(["python", script], check=True) #throw an error if it doesnt run
    except subprocess.CalledProcessError as error: #store the error in this variable
        print(f"Error in script {script}: {error}")
        break

print("Finished running all scripts.")
