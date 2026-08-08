import torch
import torchvision
import torch.nn as nn

from model.moe_transformer import Moe_Transformer



class Encoder(nn.Module):
    def __init__(self,em_size):
        super().__init__()
        self.em_size=em_size

        self.resnet50=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet50.avgpool=nn.Identity()
        self.resnet50.fc=nn.Identity()
        
        
        for name , param in self.resnet50.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad=True

            else:
                param.requires_grad=False


        self.projection_layer=nn.Sequential(
            nn.Linear(2048,self.em_size),
            nn.LeakyReLU(.2),
            nn.LayerNorm(self.em_size),
            nn.Dropout(0.01)
           )
        
        
    def forward(self,x):
        
        out=self.resnet50(x)
        B,_=out.shape

        out=out.reshape(B,2048,7,7)
        projection_out=out.flatten(2).transpose(1,2)
        

        return self.projection_layer(projection_out)

        

class encoder_decoder(nn.Module):
    def __init__(self,em_size,
                 hidden_size,
                 vocab_size,
                 num_expert,
                 expand_scale,
                 max_length=1000,
                 padding_idx=0,
                 num_head=4,
                 head_dim=312,
                 num_layers=2,
                 dropout_size=.1 ):
        super().__init__()

        self.em_size=em_size
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.num_expert=num_expert
        self.max_length=max_length
        self.num_layers=num_layers
        self.dropout_size=dropout_size
        self.expand_scale=expand_scale
        self.num_head=num_head
        self.head_dim=head_dim
        self.padding_idx=padding_idx



        self.encoder=Encoder(self.em_size)

        self.em_layer=nn.Embedding(self.vocab_size,self.em_size,padding_idx=self.padding_idx)

        self.pos_em_image   = nn.Embedding(49,self.em_size)


        self.decoder=Moe_Transformer(self.em_size
                                     ,self.hidden_size,
                                     self.vocab_size,
                                     self.expand_scale,
                                     self.padding_idx,
                                     self.max_length,
                                     
                                     self.num_head,
                                     self.head_dim,
                                     self.num_layers,
                                     self.num_expert,
                                     self.dropout_size)
        
        self.decoder.em_layer=nn.Identity()
        
        

    
    def forward(self,images,captions,target_caption,padding_mask=None,enc_padding_mask=None ,add_noise = True , confidance = 1):
        image_pos=torch.arange(0,49,device=images.device).expand(images.size(0),-1)
        image_posem=self.pos_em_image(image_pos)

        image_em=self.encoder(images)
        image_em = image_em + image_posem

        captions_em=self.em_layer(captions)


        logits, exit_layer, div_loss, computation_loss, exit_cr_entropy , exit_probability=self.decoder(captions_em,target_caption,image_em,padding_mask,enc_padding_mask,add_noise , confidance)

        return logits, exit_layer, div_loss, computation_loss, exit_cr_entropy , exit_probability
    


       