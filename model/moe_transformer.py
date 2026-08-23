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

        # embiddings layers 
        self.em_layer = nn.Embedding(self.vocab_size,self.input_size,padding_idx=self.padding_idx)
        self.pos_em   = nn.Embedding(max_length,self.input_size)


        # this layer diside wheather to early exit or not

        self.exit_layers =nn.ModuleList(
            
         nn.Sequential(
             nn.RMSNorm(self.input_size),
            nn.Linear(self.input_size,1),

        )
        for _ in range(self.num_layers)
        )

        # for each Layer it's own classification layer to learn own representation
        self.early_classification_layer =nn.ModuleList(
            
         nn.Sequential(
            nn.RMSNorm(self.input_size),
            nn.Linear(self.input_size,vocab_size)

        )
        for _ in range(self.num_layers)
        )

        # decoders Layers
            

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

        self.exit_cr_entropy = nn.CrossEntropyLoss(ignore_index=padding_idx)

        self.bce_loss = nn.BCEWithLogitsLoss() 
        
        self.layer_norm=nn.RMSNorm(self.input_size)

        self.fc=nn.Linear(self.input_size,vocab_size)

    

    def forward(self,input,target,encoder_feature,padding_mask=None,enc_padding_mask=None , add_noise:bool =True , confidance = 1):
        
        ems=self.em_layer(input)
        B , seq , _ = ems.shape
        postions=torch.arange(0,seq,device=input.device).expand(B,seq)
        em_postion=self.pos_em(postions)

        # applying noise on training to make the model able to generalize
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
        confidance_loss = 0
        exit_cr_entropy = 0
        exit_probability= 0
        for i , (decoder,exit_layer,early_class_layer) in enumerate(zip(self.decoder_layers , self.exit_layers , self.early_classification_layer)):
            out,loss=decoder(out,encoder_feature,padding_mask,enc_padding_mask)
            div_loss+=loss
            
            confidance_logits = exit_layer(out)

            exit_probability = confidance_logits

            
            # during training try to make the early layer to be able to output high quality output 
            if self.training:
            # break if this the last layer
                if i == self.num_layers - 1 :
                    break
                
                
                logits=early_class_layer(out) # the classification_layer

                weight = (self.num_layers -1 - i) / (self.num_layers -1)

                #  (num_layers - current_layer) / self.num_layers  will help the model focus on the quality not just being fast
                exit_cr_entropy += weight * self.exit_cr_entropy(

                                         logits.reshape(-1,self.vocab_size),
                                         target.reshape(-1)
                                     )

                with torch.no_grad():
                    mask = target != self.padding_idx
                    pred = logits.argmax(-1)
                    correct = ((pred == target) & mask).float()
                    # print(correct[mask])

                confidance_loss  +=  weight * self.bce_loss( 

                    exit_probability.squeeze(-1)[mask] , 

                    correct[mask])
                

            
            if i == self.num_layers - 1 :
                break;
            
            # for inferance if exit probability > confidance it will classify as fast
            exit_probability = torch.sigmoid(exit_probability)
            if exit_probability[0,-1,0].item() >= confidance and not self.training :

                
                logits=early_class_layer(out)

                return logits , i , div_loss / self.num_layers , 0 , 0 , exit_probability[0,-1,0].item()



        out=self.layer_norm(out)
        
        logits=self.fc(out) # the classification_layer

        # this mean the last layer
        i = -1
        return logits , i , div_loss / self.num_layers , confidance_loss , exit_cr_entropy , exit_probability[0,-1,0].item()
        

