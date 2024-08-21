from torchvision import transforms

# Image transformations: Resize, normalize, augment if needed
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize dataset
dataset = BiomedicalDataset(
    csv_file='*********.csv',
    image_dir='**********',
    transform=image_transforms,
    max_text_length=128,  # You can adjust this based on your specific task
    tokenizer_name='bert-base-uncased'
)

# DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
