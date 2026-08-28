import torch
from model.trainer import training , train_loader
from model.model_config import model , device


# Hyper-params initialize


optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)

epochs = 10

lr_decay=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-6)


# The training

training(model=model
         ,optimizer=optimizer
         ,train_loader=train_loader,
         epochs=epochs,
         accumlation_steps=1, # i didn't used it :)
         warm_steps=0,
         lr=1e-4, # learning rate value
         alpha=1e-5, # determine how fast should the router learn to route (MOE)
         weight_decay=1e-3,
         l1_value=0, # L1 norm help the model pick the most important feature
         device=device,
         logs_dir='Logs/test2_with gussian_noise and noise_strength = .35 and early layer classification loss wiht beta of .1 & confiance_loss with beta2 of 1e-3',
         save_epoch=1, # determine what when to make a checkpoint
         lr_scheduler=lr_decay, #decreese the learning rate **Mostly** each epoch 
         add_noise=True, #add noise to the both output attention of the image and token embidding 
         noise_strength=.35

         )