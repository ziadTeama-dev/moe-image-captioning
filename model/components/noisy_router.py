import torch
import torch.nn as nn



class Noisy_router(nn.Module):
    """
    input_size:the dimension that each token have ex.. (512,1024,etc)
    num_experts:the number of expert will used to get the propability dist for each expert
    this noisy_router class will output the propability distrubution for each expert is likily to have most knowledge about that specific domain
    like..(syntax,pronoun,extructure,...,etc)

    """
    def __init__(self,
                 input_size
                 ,num_experts
                 ,noise=True):
        super().__init__()

        self.input_size=input_size
        self.num_experts=num_experts
        self.noise=noise

        if noise:

            self.noise_layer = nn.Linear(input_size, num_experts)
         
            

        self.router=nn.Sequential(
           nn.Linear(input_size,512),
           nn.GELU(),
           nn.LayerNorm(512),
           nn.Linear(512,self.num_experts)
                            )
                                            
        self.softmax=nn.Softmax(dim=-1)
                                 
    
    def forward(self,x):
        
        logits=self.router(x)
        ## set rand_noise if in training mode
        if self.training:
           
         if self.noise:
           
           rand_noise=torch.randn_like(logits)*torch.sigmoid(self.noise_layer(x))

        else:
           
           rand_noise=torch.zeros_like(logits)

        return self.softmax(rand_noise+logits)


    








