import torch.nn as nn
import torch
from model.components.moe_decoder import Moe_Decoder

class Moe_Transformer(nn.Module):
    def __init__(self,input_size,hidden_size,vocab_size,expand_scale,padding_idx,max_length,num_heads,head_dim,num_layers,num_expert,dropout):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.num_layers=num_layers
        self.num_expert=num_expert
        self.max_length=max_length
        self.dropout=dropout
        self.expand_scale=expand_scale
        self.num_heads=num_heads
        self.head_dim=head_dim
        self.padding_idx=padding_idx
        
        self.em_layer = nn.Embedding(self.vocab_size,self.input_size,padding_idx=self.padding_idx)
        self.pos_em   = nn.Embedding(max_length,self.input_size)

        self.decoder_layers  = nn.ModuleList(

            [Moe_Decoder(self.input_size,
                         self.hidden_size,
                         self.expand_scale,
                         self.num_expert,
                         self.dropout,
                         self.num_heads,
                         self.head_dim)

                         for _ in range(self.num_layers)]

            )
        
        self.layer_norm=nn.LayerNorm(self.input_size)

        self.fc=nn.Linear(self.input_size,vocab_size)

    

    def forward(self,input,encoder_feature,padding_mask=None,enc_padding_mask=None , add_noise:bool =True):
        ems=self.em_layer(input)
        B , seq , _ = ems.shape
        postions=torch.arange(0,seq,device=input.device).expand(B,seq)
        em_postion=self.pos_em(postions)

        if self.training and add_noise:
          is_rand=torch.randint(0,2,(1,))

          if is_rand ==0:
              rand_gu=torch.randn_like(ems)
          else:
              rand_gu=torch.zeros_like(ems)

        else:
              rand_gu=torch.zeros_like(ems)



 


        # adding ems + em_postion
        out=ems+em_postion+ (rand_gu.to(input.device)*(torch.rand(B,seq,1).to(input.device)*.5))

        # the out will go throught all the layer of the decoders sequantially 
        div_loss=0
        for decoder in self.decoder_layers:
            out,loss=decoder(out,encoder_feature,padding_mask,enc_padding_mask)
            div_loss+=loss

        out=self.layer_norm(out)
        
        logits=self.fc(out) # the classification_layer

        return logits , div_loss / self.num_layers

    



# some test


# x=torch.randint(0,100,(5,12))
# model=Moe_Transformer(512,712,1000,1000,3,3,0.1)

# out,div_loss=model(x)

# print(out.shape)
# print(out)
# print(div_loss)

        

