import torch
from model.model import encoder_decoder
from model.trainer import vocab_size

device='cuda' if torch.cuda.is_available() else 'cpu'

model=encoder_decoder(em_size=512,
                            hidden_size=512,
                            expand_scale=4,
                            vocab_size=vocab_size,
                            num_expert=3,
                            max_length=1000,
                            num_head=8,
                            head_dim=512,
                            num_layers=4,
                            dropout_size=0.2     
                      ).to(device)