import torch.nn as nn
from model.components.mha import MHA
from model.components.noisy_router import Noisy_router
from model.components.moe import MOE

class Moe_Decoder(nn.Module):
    def __init__(self,input_size=512,hidden_size=512,expand_scale=3,num_expert=1,dropout_size=.2,num_heads=4,head_dim=312):
        """
        input_size: the input going in through the network like (batch,sequance,em_size) the input_size here is the em_size
        num_expert: the number of expert that will process one token at once (one expert for one token )
        drop_out:   the probabilty (0-1) to stop that nueron (make it 0). good for small data or small net.
        this will return output size the same as input_size
        """
        super().__init__()

        self.hidden_size=hidden_size
        self.input_size=input_size
        self.expand_scale=expand_scale
        self.num_expert=num_expert
        self.dropout_size=dropout_size
        self.num_heads=num_heads
        self.head_dim=head_dim
        
        
        self.pre_norm=nn.LayerNorm(self.input_size)

        self.mha=MHA(self.input_size
                     ,self.head_dim
                     ,self.num_heads
                     ,self.input_size
                     ,True)
        
        self.cross_norm = nn.LayerNorm(input_size)
        self.cross_mha = MHA(self.input_size, 
                             self.head_dim, 
                             self.num_heads, 
                             self.input_size, 
                             mask=False)
        
        self.dropout=nn.Dropout(self.dropout_size) 

        #out= input+ out-> from multiheadattenetion 

        self.pre_norm2=nn.LayerNorm(self.input_size)
        self.noisy_router=Noisy_router(input_size,self.num_expert,True)

        self.moe=MOE(input_size,
                     hidden_size,
                     self.num_expert,
                     self.expand_scale) 
        

        # out = out + out-> from (moe)
    def forward(self, x, encoder_features=None, padding_mask=None, enc_padding_mask=None):

        # X ------------------------
        out_norm=self.pre_norm(x)#--------------------------------------------------
        attention_output=self.mha(out_norm,out_norm,out_norm,padding_mask) #----------------------

        res1= attention_output + x # attention_output + X <--------


        if encoder_features is not None:

            res1_norm = self.cross_norm(res1)
            cross_attn = self.cross_mha(res1_norm, encoder_features, encoder_features, enc_padding_mask)
            cross_attn = self.dropout(cross_attn)
            res2 = res1 + cross_attn

        else:
            
            res2 = res1  

        # res1 ----------------------------
        out_norm2=self.pre_norm2(res2) #---------
        rout_logits=self.noisy_router(out_norm2)#--------------
        out,div_loss=self.moe(out_norm2,rout_logits)# ---------
        out=self.dropout(out) #out + res1<----------------------

        res2=res2+out 

        # if self.training ==False: #just for searching purposes
            # print(f'gated_scores:{attention_gated}')
            # just to know what the model focusses on more rather than each element
            # print(f'AVG_gated_scores:{torch.mean(attention_gated,dim=-1)}')

        return res2,div_loss
    



