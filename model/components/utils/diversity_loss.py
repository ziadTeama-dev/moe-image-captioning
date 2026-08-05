import torch
import torch.nn.functional as F
def diversity_loss(expert_logits):
    _,_,num_expert=expert_logits.shape
    if num_expert ==1:
        return torch.tensor([0],requires_grad=False,device=expert_logits.device)
    # print(num_expert)
    
    expert_=torch.argmax(expert_logits,dim=-1) # expert logits from softmax

    
    importance=expert_logits.sum(dim=(0,1))
    load=F.one_hot(expert_,num_expert).sum(dim=(0,1)).float() #insteed of for loop

    return ( cv(importance)**2 + cv(load)**2 )



def cv(x):
    return  x.std(unbiased=False) / (x.mean() + 1e-6)  
    