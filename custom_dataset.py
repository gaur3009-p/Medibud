import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
from transformers import BertTokenizer

class BiomedicalDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, max_text_length=128, tokenizer_name='bert-base-uncased'):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        image_path = f"{self.image_dir}/{self.data.iloc[idx]['image_dataset']}"
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load text
        text = self.data.iloc[idx]['text']
        encoded_text = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_text_length
        )

        # Load label
        label = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.long)

        return image, encoded_text, label
