import torch
import torchvision
import imageio.v3 as iio
import matplotlib.pyplot as plt

from PIL import Image

def image_caption(model, full_image_path, vocab, max_length=30, device='cuda', show_image=True):
    model.eval()

    # ----- Load & preprocess image -----
    image = iio.imread(full_image_path)
    image = Image.fromarray(image).convert('RGB')
    for_shw = image

    transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    image = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    

    # ----- Encode image -----
    image_em = model.encoder(image)  # (1, seq_len, em_size)

    image_token_pos=torch.arange(0,49,device=image.device).expand(image.size(0),-1)
    image_pos_empeddings=model.pos_em_image(image_token_pos)

    image_em = image_em + image_pos_empeddings

    # ----- Start token -----
    start_id = vocab["<SOS>"]
    end_id   = vocab["<EOS>"]

    generated = [start_id]

    # initial context: start token
    context = model.em_layer(torch.tensor([[start_id]]).to(device))  # (1, 1, em_size)



    # ----- Autoregressive decoding -----
    for _ in range(max_length):
        # decoder expects captions + encoder_features
        logits, _ = model.decoder(context, image_em)  # (1, seq_len, vocab)
        
        next_step = logits[:, -1, :]                # last token logits
        next_token = next_step.softmax(-1).argmax(-1).item()

        if next_token == end_id:
            break

        generated.append(next_token)

        # embed the new token
        new_tok_em = model.em_layer(torch.tensor([[next_token]]).to(device))
        # append to context
        context = torch.cat([context, new_tok_em], dim=1)

    # ----- Convert indices to words -----
    I2w = {i: w for w, i in vocab.items()}
    gen_text = " ".join([I2w[i] for i in generated[1:]])  # skip <SOS>

    if show_image:
        plt.imshow(for_shw)
        plt.title(gen_text)
        plt.axis('off')
        plt.show()

    return gen_text
