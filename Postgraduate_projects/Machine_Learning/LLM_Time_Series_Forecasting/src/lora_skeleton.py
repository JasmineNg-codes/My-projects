import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from accelerate import Accelerator


# LoRA implementation
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None):
        
        #This chunk initialises the class with a linear layer, keeping the weights and biases of the original linear layer
        super().__init__()
        assert isinstance(original_linear, nn.Linear)
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        in_dim = original_linear.in_features #input dimensions
        out_dim = original_linear.out_features #output dimensions
        
        self.r = r #rank, meaning how much capacity LoRA has to change the weights (heigher R, more trainable parameters)
        self.alpha = alpha if alpha else r #scaling factor, to prevent extreme modifications.

        device = original_linear.weight.device
        
        self.A = nn.Parameter(torch.empty(r, in_dim, device=device)) #low rank adaptation matrixes
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device)) #matches the device, so if the original layer stores tensors in CPU or GPU, it will follow that.
        #Essentially, after the linear transformation of the layer, I take the weight matrix and update that with my small matrix, giving the modified weight matrix.
        
        # Initialise A with He initialization
        #Ensures stable training by preventing vanishing/exploding gradients
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")

    def forward(self, x):
        base_out = self.original_linear(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + lora_out * (self.alpha / self.r)
    
        # Modified tokenization with chunking
    def process_sequences(texts, tokenizer, max_length=512, stride=256):
        all_input_ids = []
        for text in texts:
            # Apply Qwen's tokenization scheme to the text:
            encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            seq_ids = encoding.input_ids[0]

            # Create sliding windows to further divide the data into chunks:
            for i in range(0, len(seq_ids), stride):
                chunk = seq_ids[i : i + max_length]
                if len(chunk) < max_length:
                    chunk = torch.cat(
                        [
                            chunk,
                            torch.full((max_length - len(chunk),), tokenizer.pad_token_id),
                        ]
                    )
                all_input_ids.append(chunk)
        return torch.stack(all_input_ids)
