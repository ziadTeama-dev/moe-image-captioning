import torch
from model.trainer import training , train_loader , vocab_size
from model.model import encoder_decoder


device='cuda' if torch.cuda.is_available() else 'cpu'


model=encoder_decoder(
                            em_size=712,
                            hidden_size=712,
                            expand_scale=2,
                            vocab_size=vocab_size,
                            num_expert=3,
                            max_length=1000,
                            num_head=8,
                            head_dim=712,
                            num_layers=4,
                            dropout_size=0.2
                      ).to(device)

optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)

epochs = 10

lr_decay=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-6)



training(model=model
         ,optimizer=optimizer
         ,train_loader=train_loader,
         epochs=epochs,
         accumlation_steps=1,
         warm_steps=0,
         lr=1e-4,
         alpha=1e-5,
         weight_decay=1e-3,
         l1_value=0,
         device=device,
         logs_dir='Logs/test1_with gussian_noise and early layer exiting wiht beta of .5',
         save_epoch=1,
         lr_scheduler=lr_decay,
         # checkpoints_path="checkpoints/checkpoint_9.pth",
         )