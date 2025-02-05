import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Custom Dataset Class
class MTGCardDataset(Dataset):
    def __init__(self, captions_file, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        with open(captions_file, 'r') as f:
            self.data = [line.strip().split('\t') for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, caption = self.data[idx]
        image = Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
        image = self.transform(image)
        return image, caption

# Load dataset
dataset = MTGCardDataset('dataset/captions.txt', 'dataset', preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch.nn.functional as F
import torch.optim as optim

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Training Loop
for epoch in range(5):  # Adjust epochs as needed
    total_loss = 0
    for images, captions in dataloader:
        images = images.to(device)
        texts = clip.tokenize(captions).to(device)

        optimizer.zero_grad()

        # Encode images and text
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Contrastive loss (InfoNCE)
        logits_per_image = image_features @ text_features.T
        logits_per_text = logits_per_image.T
        labels = torch.arange(images.size(0), device=device)

        loss = (F.cross_entropy(logits_per_image, labels) + 
                F.cross_entropy(logits_per_text, labels)) / 2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

torch.save(model.state_dict(), "clip_mtg_finetuned.pt")
