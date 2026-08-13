import os
import torch
# from model.evaluation import evaluate_bleu
from model.generation import image_caption
from model.trainer import  vocab , vocab_size
from model.model import encoder_decoder
# from model.trainer import test_dataset

device = "cuda"  if torch.cuda.is_available() else "cpu"

model=encoder_decoder(em_size=512,
                            hidden_size=512,
                            expand_scale=4,
                            vocab_size=vocab_size,
                            num_expert=3,
                            max_length=1000,
                            num_head=8,
                            head_dim=64,
                            num_layers=4,
                            dropout_size=0.1
                      ).to(device)


# loading the checkpoint
if os.path.exists(f"checkpoints/checkpoint_0.pth"):
    checkpoint = torch.load(f"checkpoints/checkpoint_0.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

# print("\n the model loaded \n\n")
# print("*Evaluation*: \n")

# bl1,bl4=evaluate_bleu(model, test_dataset, flick, device="cuda", max_examples=None)

# print(f'bl1:{bl1}')
# print(f'bl4:{bl4}')



print(image_caption(model=model,
                    full_image_path='test_images/two-dogs-playing-with-flying-disc-park.jpg',
                    vocab=vocab.W2i,
                    max_length=40))

