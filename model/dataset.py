
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms

from PIL import Image
import imageio.v3 as iio
import pandas as pd
import spacy
import os
import re
import string

from sklearn.model_selection import train_test_split



# SPACY TOKENIZER


spacy_eng = spacy.load("en_core_web_sm")



# cleaning function


def clean_caption(caption: str) -> str:

    caption = caption.lower()

    # Remove punctuation
    caption = caption.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove digits
    caption = re.sub(r'\d+', '', caption)

    # Keep alphabetic characters and spaces
    caption = re.sub(r'[^a-z ]', ' ', caption)

    # Remove extra spaces
    caption = re.sub(r'\s+', ' ', caption).strip()

    return caption


# VOCABULARY

class Vocabulary:

    def __init__(self, thres: int):

        self.thres = thres

        self.vocap_size = None

        self.W2i = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3
        }

        self.I2w = {
            0: '<PAD>',
            1: '<SOS>',
            2: '<EOS>',
            3: '<UNK>'
        }

    def __len__(self):
        return len(self.W2i)

    def tokinization(self, text: str):

        return [
            w.text.lower()
            for w in spacy_eng.tokenizer(text)
        ]

    def build_vocab(self, sent_list: list):

        freq_list = {}

        idx = 4

        for sent in sent_list:

            for w in self.tokinization(sent):

                if w not in freq_list:
                    freq_list[w] = 1
                else:
                    freq_list[w] += 1

                if freq_list[w] == self.thres:

                    self.W2i[w] = idx
                    self.I2w[idx] = w

                    idx += 1

        self.vocap_size = len(self.W2i)

    def encoded(self, text: str):

        tokenized_text = self.tokinization(text)

        return [
            self.W2i[w]
            if w in self.W2i
            else self.W2i['<UNK>']
            for w in tokenized_text
        ]


# DATASET

class Flicker_data(Dataset):

    def __init__(
        self,
        dataframe,
        image_path: str,
        vocab,
        transform=None
    ):

        super().__init__()

        self.x_y = dataframe.reset_index(drop=True)

        self.image_path = image_path

        self.transform = transform

        self.images = self.x_y['image']

        self.captions = self.x_y['caption']

        self.vocab = vocab

    def __len__(self):

        return len(self.x_y)

    def __getitem__(self, index):

        # <SOS>
        y = [1]

        # Image
        image_file_name = self.images.iloc[index]

        full_image_path = os.path.join(
            self.image_path,
            image_file_name
        )

        x = iio.imread(full_image_path)

        x = Image.fromarray(x).convert('RGB')

        # Caption
        caption_text = self.captions.iloc[index]

        caption_text = clean_caption(caption_text)

        # Encode
        y += self.vocab.encoded(caption_text)

        # <EOS>
        y.append(2)

        y = torch.tensor(
            y,
            dtype=torch.long
        )

        # Transform
        if self.transform is not None:
            x = self.transform(x)

        return x, y



# LOAD FULL CSV


df = pd.read_csv(
    'Data/captions.csv'
)

# UNIQUE IMAGES


unique_images = df['image'].unique()

print(
    f"Total unique images: {len(unique_images)}"
)



# IMAGE-LEVEL TRAIN / TEST SPLIT


train_images, test_images = train_test_split(
    unique_images,
    test_size=0.125,
    random_state=42
)

train_images = set(train_images)
test_images = set(test_images)



# CREATE DATAFRAMES


train_df = df[
    df['image'].isin(train_images)
].reset_index(drop=True)

test_df = df[
    df['image'].isin(test_images)
].reset_index(drop=True)



# SANITY CHECK


train_image_set = set(train_df['image'])
test_image_set = set(test_df['image'])

overlap = train_image_set.intersection(
    test_image_set
)

print(
    f"Train images: {len(train_image_set)}"
)

print(
    f"Test images: {len(test_image_set)}"
)

print(
    f"Train samples: {len(train_df)}"
)

print(
    f"Test samples: {len(test_df)}"
)

print(
    f"Image overlap: {len(overlap)}"
)


# BUILD VOCABULARY USING TRAIN CAPTIONS ONLY

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


