import os
import torch
import torchvision
import torch.nn as nn
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from model.evaluation import evaluate_bleu

from model.dataset import *
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from model.components.utils.transform import train_transform
import random
import numpy as np


# make the same split each time
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Vocab initialization
vocab = Vocabulary(
    thres=3
)

vocab.build_vocab(
    list(train_df['caption'])
)

print(
    f"Vocabulary size: {vocab.vocap_size}"
)


# Create dataset

train_dataset = Flicker_data(
    dataframe=train_df,
    image_path='Data/Images/',
    vocab=vocab,
    transform=train_transform
)

vocab_size = vocab.vocap_size

test_dataset = Flicker_data(
    dataframe=test_df,
    image_path='Data/Images/',
    vocab=vocab,
    transform=torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
)


# Create DATALOADERS

batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)





# FINAL INFORMATION


print("\n========== DATASET SUMMARY ==========")

print(
    f"Train unique images : {len(train_image_set)}"
)

print(
    f"Test unique images  : {len(test_image_set)}"
)

print(
    f"Train samples       : {len(train_dataset)}"
)

print(
    f"Test samples        : {len(test_dataset)}"
)

print(
    f"Vocabulary size     : {vocab.vocap_size}"
)

print(
    f"Image overlap       : {len(overlap)}"
)


print("\n================= Trainer ===============")

# training function
def training(model, 
             optimizer=None, 
             train_loader=None,
             epochs:int=100,
             accumlation_steps=2,
             warm_steps=5,
             lr=None,
             alpha=0.001,
             weight_decay=None,
             l1_value:int=0, 
             device='cpu',
             logs_dir:str=None,
             save_epoch:int=None,
             lr_scheduler=None,
             checkpoints_path:str=None):
    """
    model: the caption model that you will train on images,captions
    train_loader: the data-set that will be used to train the model
    epoch: number of epoch used to train the model
    logs_dir:the path where to put logs file of (loss,avg) for monitoring using tensorboard
    save_epoch:determine when to take a checkpoint of the model
    optimizer: the optimizer used to train the model (adam,adamw,sdg,...etc)
    checkpoints_path: the path you specify to save a checkpoint of model
    lr_scheduler : used to decay the learning rate 
    device: 'cuda','cpu'
    """
    

    device = device
    writer = SummaryWriter(logs_dir)

    model = model.to(device)
    cr = nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.05)

    total_batches = len(train_loader)
    start_epoch = 0
    total_steps = 0




    print(f'Total batch_size = batch_size * accumlation_steps = ( {batch_size} * {accumlation_steps} ) ={batch_size*accumlation_steps}')

    if os.path.exists(f"{checkpoints_path}"):
        checkpoint = torch.load(f"{checkpoints_path}", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1  
        total_steps = checkpoint.get('steps', 0)
        print(f"✅ Resuming training from epoch {start_epoch}")
        
    if weight_decay != None:   
     for param_group in optimizer.param_groups: ## just changing the weight_decay
      param_group["weight_decay"]=weight_decay
     



    try:
        print("Training..")

        for e in tqdm(range(start_epoch, epochs)):
            avg_loss = 0
            avg_div_loss=0

            avg_computation_loss = 0
            avg_exit_cr_entropy=0

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            for step, (image, caption , padding_mask) in enumerate(train_loader):
                image = image.to(device)
                caption = caption.to(device)
                in_caption = caption[:,:-1]

                is_rand=torch.randint(0,2,(1,))

                if is_rand ==0:

                    B,_,_,_=image.shape
                    image = image + (torch.randn_like(image).to(device) *torch.rand(B,1,1,1).to(device)*.5)  

                padding_mask = padding_mask[: , :-1]
                caption = caption[: , 1:]

                out, _, div_loss, computation_loss, exit_cr_entropy , _ = model( image , in_caption , caption , padding_mask )

                out = out.reshape(-1, vocab_size).to(device)

                caption = caption.reshape(-1).to(device)

                # it will make the training alittile bit longer but it's good for now :)
                if l1_value !=0:
                 l1_norm=sum(p.abs().sum() for p in model.parameters())
                
                else:
                   l1_norm=0
                   
                #  quality metrices
                raw_loss = cr(out, caption)
                raw_loss = raw_loss + (alpha * div_loss) + (l1_value * l1_norm) + 0.5 * exit_cr_entropy + 1e-5 * computation_loss
                avg_loss += raw_loss.item() / total_batches
                avg_exit_cr_entropy  += exit_cr_entropy / total_batches

                # performance metrices

                avg_div_loss += div_loss.item() / total_batches
                avg_computation_loss += computation_loss / total_batches
                
                loss = raw_loss
                loss.backward()

                # if (step+1) % accumlation_steps == 0:


                total_steps += 1

                if step % 20 == 0:
                    tota_grad=0
                    for param in model.decoder.parameters():
                        if param.grad != None:
                            param_norm=param.grad.norm(2)
                            tota_grad+=param_norm.item()**2

                    for param_group in optimizer.param_groups: 
                       learning_rate = param_group["lr"]
                    
                    tota_grad = tota_grad ** 0.5

                    if div_loss !=0:
                    
                      print(f'[{e}|{epochs}][{step}]: loss={raw_loss.item():.4f}-------->div_loss={div_loss:.4f}')
                    
                    else:
                      
                      print(f'[{e}|{epochs}][{step}]: loss={raw_loss.item():.4f}')
                       
                       

                    writer.add_scalar('LOSS PER BATCH',raw_loss,total_steps)
                    writer.add_scalar('Computation LOSS PER BATCH',computation_loss,total_steps)
                    writer.add_scalar('Exit Centropy LOSS PER BATCH', exit_cr_entropy , total_steps)
                    writer.add_scalar('DIV LOSS PER BATCH',div_loss,total_steps)
                    writer.add_scalar('Gradiant norm',tota_grad,total_steps)
                    
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()

                torch.cuda.empty_cache()

            # if (step + 1) % accumlation_steps != 0 and accumlation_steps !=1:
               
            #     optimizer.step()
            #     optimizer.zero_grad()
            #     model.zero_grad()



            # --- Checkpoint ---
            if not os.path.exists("checkpoints"):
              os.mkdir('checkpoints')
            
            # alpha*=0.99
              
            if save_epoch and (e % save_epoch == 0):
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'steps': total_steps,
                    'lr_scheduler_state_dict':lr_scheduler.state_dict()
                }, f"checkpoints/checkpoint_{e}.pth")
                
            if lr_scheduler != None and e >= warm_steps:
                lr_scheduler.step()

            learning_rate = optimizer.param_groups[0]["lr"]

            writer.add_scalar('Learning rate',learning_rate,e)
            writer.add_scalar("AVG Loss", avg_loss, e)
            writer.add_scalar('AVG DIV LOSS',avg_div_loss,e)
            writer.add_scalar('AVG Computation LOSS PER BATCH',avg_computation_loss,e)
            writer.add_scalar('AVG Exit Centropy LOSS PER BATCH', avg_exit_cr_entropy , e)

            if avg_div_loss!=0:
               
              print(f'---- AVG LOSS Epoch {e} ----> {avg_loss:.4f}------->{avg_div_loss:.4f}')
            
            else:
              print(f'---- AVG LOSS Epoch {e} ----> {avg_loss:.4f}')


            # evalution on small subset of test data
            bleu1, bleu4 = evaluate_bleu(model, test_dataset, vocab, 'cuda', max_examples=10)

            writer.add_scalar("BLEU-1", bleu1, e)
            writer.add_scalar('BLEU-4', bleu4, e)

            print("Test BLEU-1:", bleu1)
            print("Test BLEU-4:", bleu4)

        writer.close()

    except KeyboardInterrupt:
        print("Training stopped by user.")
        writer.close()
