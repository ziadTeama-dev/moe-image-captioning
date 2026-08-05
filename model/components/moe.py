import torch
import torch.nn as nn

from model.components.utils.diversity_loss import diversity_loss

class MOE(nn.Module):
    def __init__(self,input_size,hidden_size,num_experts=1,expand_scale=3 ):
        super().__init__()
        """
        input_size: the input going in through the network like (batch,sequance,em_size) the input_size here is the em_size
        num_expert: the number of expert that will process one token at once (one expert for one token )
        this will return the input_size as the output size 
        """
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.expand=expand_scale
        self.num_experts=num_experts

        self.moe=nn.ModuleList([
            nn.Sequential(nn.Linear(self.input_size
                                    ,hidden_size*self.expand)
                                    ,nn.GELU()
                                    ,nn.Linear(hidden_size*self.expand,
                                               self.input_size))
        for _ in range(self.num_experts)])

        self.shared_expert= nn.Sequential(nn.Linear(self.input_size
                                    ,hidden_size*self.expand)
                                    ,nn.GELU()
                                    ,nn.Linear(hidden_size*self.expand,
                                               self.input_size))

    def forward(self,x,router_logits):
        top_1=torch.argmax(router_logits,dim=-1)
        output=torch.zeros_like(x)
        div_loss=diversity_loss(router_logits)

        
        shared_experiance=self.shared_expert(x)

        for exprt_idx,expert in enumerate(self.moe):
            mask=top_1==exprt_idx
            
            # print(f'exceed num:{exceed}')

            if mask.any():
                selected=x[mask]
                output[mask]=expert(selected)+shared_experiance[mask]
        

    
        return output , div_loss



