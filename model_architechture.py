import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel, BertTokenizer

class VisionLanguageModel(nn.Module):
    def __init__(self, vision_model_name='resnet50', language_model_name='bert-base-uncased', num_classes=10):
        super(VisionLanguageModel, self).__init__()
        
        # Vision Model (ResNet as an example)
        self.vision_model = models.__dict__[vision_model_name](pretrained=True)
        self.vision_model.fc = nn.Identity()  # Removing the final classification layer

        # Language Model (BERT as an example)
        self.language_model = BertModel.from_pretrained(language_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(language_model_name)
        self.language_fc = nn.Linear(self.language_model.config.hidden_size, 512)

        # Combined FC layer
        self.fc_combined = nn.Linear(512 + self.vision_model.fc.in_features, num_classes)

    def forward(self, image, text):
        # Vision Path
        vision_features = self.vision_model(image)
        
        # Language Path
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        language_outputs = self.language_model(**inputs)
        language_features = self.language_fc(language_outputs.pooler_output)
        
        # Combine Features
        combined_features = torch.cat((vision_features, language_features), dim=1)
        output = self.fc_combined(combined_features)
        
        return output
