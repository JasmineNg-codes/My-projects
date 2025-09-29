import numpy as np

def flops(S, D, H=14, z=4864, v=151936, training = False, training_steps = None, validation = False, validation_steps = None):

    """
    Compute total FLOPs for a full forward pass through a 24-layer transformer. 
    Could also be thought of as inference mode, unless specified training, training steps, lora configuration, and lora rank.

    Args:
        S (int): Sequence length
        D (int): Hidden dimension
        H (int): Number of attention heads (default: 14)
        z (int): SwiGLU intermediate dimension (default: 4864)
        v (int): Vocabulary size used in final linear layer (default: 151936)

    Returns:
        total FLOPs for the experiment in scientific notation
        
    """
    
    # --- Validate arguments ---
    if training and training_steps is None:
        raise ValueError("You must provide 'training_steps' when training=True.")
    
        # --- Validate arguments ---
    if validation and validation_steps is None:
        raise ValueError("You must provide 'validation_steps' when validation=True.")
    
    # Initialize FLOPs for optional modes
    train_flops = 0
    valid_flops = 0
        
    # Positional Encoding added 
    add_positional_encoding = S * D

    # ----- FLOPs for 1 transformer block -----
    # RMSNorm
    rms_flops = 3 * ((S * D) + (S * (D - 1)) + D)

    # QKV projection
    qkv_proj = 3 * (S * D * (2 * D - 1)) + 3 * (S * D)
    
    # Multi-headed Attention
    mha_qk = H * (S * S * ((2 * D // H) - 1))
    mha_qk_norm = H * (S * S)
    mha_softmax = H * (23 * S * S)
    mha_v_softmax = H * (S * S * ((2 * D // H) - 1))
    mha_output = S * D * (2 * D - 1)
    mha_flops = mha_qk + mha_qk_norm + mha_softmax + mha_v_softmax + mha_output

    # Residual Connection after attention
    res1 = S * D

    # RMSNorm again
    rms2_flops = rms_flops

    # SwiGLU
    swiglu_proj = 2 * (S * D * (2 * z - 1))
    swiglu_swish = 13 * S * z + S * z
    swiglu_mult = S * z
    swiglu_output = S * z * (2 * D - 1)
    swiglu_total = swiglu_proj + swiglu_swish + swiglu_mult + swiglu_output

    # Residual Connection after FFN
    res2 = S * D

    # FLOPs for one transformer block
    transformer_block_flops = (rms_flops + qkv_proj + mha_flops + res1 +
                               rms2_flops + swiglu_total + res2)

    # Total for 24 transformer blocks
    total_transformer_flops = 24 * transformer_block_flops

    # Final RMSNorm and Linear layer (only once)
    final_rms = rms_flops
    linear = S * v * (2 * D - 1)

    ##### INFERNECE MODE #####
    one_pass_flops = total_transformer_flops + final_rms + linear + add_positional_encoding
        
    #### TRAINING MODE ####
    if training: 
        train_flops = 3 * one_pass_flops * training_steps
        
    #### VALIDATION MODE ####
    if validation: 
        valid_flops = 3 * one_pass_flops * validation_steps
        
    total_flops =  train_flops + valid_flops
            
    print(np.format_float_scientific(total_flops, precision=2))
    return total_flops
