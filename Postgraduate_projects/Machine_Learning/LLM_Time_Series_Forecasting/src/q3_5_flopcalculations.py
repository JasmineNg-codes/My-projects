from flops import flops 
import numpy as np

## training default qwen model for 512 context lengths, 500 steps (updates)
print("The number of flops used for the default qwen model is:")
flops_1 = flops(S = 512, D = 896, training=True, training_steps= 500, validation=True, validation_steps= 50)

## hyperparameter tuning lora qwen model:

#tuning learn rate 
print("The number of flops used for three learning rate tuning runs for LoRA is:")
flops_2 = 3 * flops(S = 512, D = 896, training=True, training_steps= 500, validation = True, validation_steps= 50)

#tuning lora rank
print("The number of flops used for three lora rank tuning runs for LoRA is:")
flops_3 = 3 * flops(S = 512, D = 896, training=True, training_steps= 500, validation=True, validation_steps= 50)

#tuning context length
print("The number of flops used for 128, 512 and 768 context length tuning runs for LoRA is respectively:")
flops_4 = flops(S = 128, D = 896, training=True, training_steps= 500, validation=True, validation_steps= 50)
flops_5 = flops(S = 512, D = 896, training=True, training_steps= 500, validation=True, validation_steps= 50)
flops_6 = flops(S = 768 , D = 896, training=True, training_steps= 500, validation=True, validation_steps= 50)

print("Another context length loRA tunign was ran for 300 sequences rather than 100 sequences")
flops_7 = flops_4 + flops_5 + flops_6 #additional for testing 300 sequences
print(np.format_float_scientific(flops_7, precision=2))

#Final model
print("The number of flops used for The final model is:")
final_flops = flops(512,896, training=True, training_steps=3000, validation=True, validation_steps= 60)

print(f"The total number of flops used in this report is:")
total_flops = flops_1 + flops_2 + flops_3 + flops_4 + flops_5 + flops_6 + flops_7 + final_flops
print(np.format_float_scientific(total_flops, precision=2))
