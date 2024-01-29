from PIL import Image
import pandas as pd

import random
import numpy as np

import torch
from torch.utils.data import Dataset,DataLoader

train_df_path = './data/train.csv'
test_df_path = './data/test.csv'

train_img_path = './data/train/'
test_img_path = './data/test/'

batch_size=8

train_df = pd.read_csv(train_df_path)
test_df = pd.read_csv(test_df_path)

train_df = train_df.head(100)
device = 'cuda:0'

from transformers import AutoProcessor, Blip2ForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map=device, load_in_8bit=True,torch_dtype=torch.float16)

from torch.utils.data import Dataset, DataLoader

class ImageCaptioningDataset(Dataset):
    def __init__(self, df, img_path, processor,is_train=False):
        self.df = df
        self.img_path = img_path
        self.processor = processor
        self.is_train = is_train
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self.row = self.df.iloc[idx]
        
        self.image = Image.open(self.img_path + self.row['img_name'] + ".jpg").convert('RGB')
        if self.is_train == True:
            self.text = self.row['comments']
    
        encoding = self.processor(images=self.image, padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = self.text
        
        return encoding

def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch

from transformers import AutoProcessor, Blip2ForConditionalGeneration

import transformers
transformers.__version__

from peft import LoraConfig, get_peft_model

# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

train_dataset = ImageCaptioningDataset(train_df,train_img_path, processor,is_train = True)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=collate_fn)

test_dataset = ImageCaptioningDataset(test_df,test_img_path, processor,is_train = True)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=16, collate_fn=collate_fn)

from tqdm import tqdm

import torch

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.train()

for epoch in range(5):
    print("Epoch:", epoch)
    for idx, batch in enumerate(tqdm(train_dataloader)):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)

        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids)
        
        loss = outputs.loss

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    print("Loss:", loss.item())
    
    torch.save(model.state_dict(),f'./results/blip2_epoch{epoch}.pth')