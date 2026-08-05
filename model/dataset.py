import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms

from PIL import Image
import imageio.v3 as iio
import pandas as pd
import spacy
import os

# loading spacy tokinizer
spacy_eng=spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self,thres:int):
        
        self.thres=thres
        self.vocap_size=None
        self.W2i={'<PAD>':0,'<SOS>':1,'<EOS>':2,'<UNK>':3}
        self.I2w={0:'<PAD>',1:'<SOS>',2:'<EOS>',3:'<UNK>'}
    
    def __len__(self):
        return len(self.W2i)
    
    def tokinization(self,text:str):
        return [w.text.lower() for w in spacy_eng.tokenizer(text)] # this will take the raw text and transform it to list of words
    
    def build_vocab(self,sent_list:list):
        freq_list={}
        idx=4
        for sent in sent_list:
            for w in self.tokinization(sent):
                if w not in freq_list:
                    freq_list[w]=1
                else:
                    freq_list[w]+=1
                
                if freq_list[w]==self.thres:
                    self.W2i[w]=idx
                    self.I2w[idx]=w
                    idx+=1
                    
        self.vocap_size=len(self.W2i)
        
    def encoded(self,text:str):

        tokinized_text=self.tokinization(text)

        return [self.W2i[w] if w in self.W2i else self.W2i['<UNK>'] 
                for w in tokinized_text]



class Flicker_data(Dataset):
    def __init__(self,captions_path:str,image_path:str,transform:transforms=None , thresshold = 3):
        super().__init__()
        self.x_y=pd.read_csv(captions_path)
        self.image_path=image_path
        self.transform=transform
        self.images=self.x_y['image']
        self.captions=self.x_y['caption']

        self.vocab=Vocabulary(thres=thresshold)
        self.vocab.build_vocab(list(self.captions))
    
    def __len__(self):
        return len(self.x_y)
    
    def __getitem__(self, index):
        y=[1] # the list included <sos> token

        image_file_name=self.images.iloc[index]
        full_image_path=os.path.join(self.image_path,image_file_name)

        x=iio.imread(full_image_path)
        x=Image.fromarray(x).convert('RGB')

        caption_text=self.captions.iloc[index]

        caption_text=clean_caption(caption_text)
        
        y+=self.vocab.encoded(caption_text)
        
        y.append(2) # adding the eos_token to the end
        y=torch.tensor(y)


        if self.transform != None:
            x=self.transform(x)
        
        return x,y


import re
import string

def clean_caption(caption: str) -> str:
    """
    Clean a caption string for image captioning.
    Steps:
    - Lowercase
    - Remove punctuation
    - Remove numbers
    - Remove extra spaces
    - Remove special/strange characters
    """
    # 1️⃣ lowercase
    caption = caption.lower()

    # 2️⃣ remove punctuation
    caption = caption.translate(str.maketrans('', '', string.punctuation))

    # 3️⃣ remove digits
    caption = re.sub(r'\d+', '', caption)

    # 4️⃣ remove any non-alphabetic characters (keep spaces)
    caption = re.sub(r'[^a-z ]', ' ', caption)

    # 5️⃣ remove extra spaces
    caption = re.sub(r'\s+', ' ', caption).strip()

    return caption


def collate_fn(batch):
    images=[item[0] for item in batch]
    captions=[item[1] for item in batch]

    images=torch.stack(images,dim=0)

    captions=pad_sequence(captions,
                          batch_first=True,
                          padding_value=0,
                          padding_side='right')
    
    padding_mask= captions == 0
    # padding_mask=None

    return images, captions, padding_mask

