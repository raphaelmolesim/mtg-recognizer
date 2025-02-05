import time
print("Loading model...")
now = time.time()

import clip
import torch
torch.set_num_threads(8)  # Adjust based on your CPU cores
import sys
from PIL import Image

# Load the fine-tuned CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the base model architecture
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the fine-tuned weights
model.load_state_dict(torch.load("clip_mtg_finetuned.pt", map_location=device))
model.eval()

# Load card names from captions.txt
with open("dataset/captions.txt", "r", encoding="utf-8") as f:
    card_names = [line.strip().split('\t')[1] for line in f]

ellapsed_time = time.time() - now

print(f"Model loaded in {ellapsed_time:.2f} seconds")

def search_card(card_name):
    now = time.time()
    # Load and preprocess the card image
    image = preprocess(Image.open(card_name)).unsqueeze(0).to(device)

    # Encode the text descriptions
    text = clip.tokenize(card_names).to(device)

    # Get CLIP features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    similarity = (image_features @ text_features.T).squeeze(0)

    elapsed_time = time.time() - now

    # if param --top
    if "--top" in sys.argv:
        top_indices = similarity.topk(10).indices.tolist()
        print("Top 10 Closest Cards:")
        for idx in top_indices:
            print(f"{card_names[idx]} (Confidence: {similarity[idx].item():.2f})") 
    else:
        # Print the best march
        max_idx = torch.argmax(similarity)
        card_name_match = card_names[max_idx]
        print(f"Best Match: {card_name_match} (Confidence: {similarity[max_idx].item():.2f} in {elapsed_time:.2f} seconds)", end = " ")
    return card_name_match

def batch_search(card_files):
    now = time.time()
    images = [preprocess(Image.open(file)).unsqueeze(0) for file in card_files]
    images = torch.cat(images).to(device)

    text = clip.tokenize(card_names).to(device)

    # Encode card names once
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize(card_names).to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    similarity = image_features @ text_features.T

    top_matches = []
    for sim in similarity:
        max_idx = torch.argmax(sim)
        candidate = card_names[max_idx]
        confidence = sim[max_idx].item()
        top_matches.append(candidate)
        end = "✅\n"  if confidence >= 0.35 else "❌\n"
        print(f"Best Match: {candidate} (Confidence: {confidence:.2f}) ", end=end)

    print(f"Batch search completed in {time.time() - now:.2f} seconds")
    return top_matches

# list files in the directory screenshot
import os
import glob

screenshot_files = glob.glob("test/Screenshot-*.png")

# import cropper.py
from cropper import ScreenshotCropper

# for each screenshot file, crop and save the cards
for file in screenshot_files:
    # read the answers
    answers = []
    file_name = file.split("\\")[-1].split(".")[0]
    with open(f"test/{file_name}-answers.txt", "r") as f:
        answers = f.readlines()
    # crop the screenshot
    cropper = ScreenshotCropper(file, 'tmp/screenshot')
    cropper.crop_and_save()
    # search for each card in the screenshot
    cards = glob.glob("tmp/screenshot/*.png")
    matches = batch_search(cards)
    test_result = [matches[i].strip() == answer.strip() for i, answer in enumerate(answers)]    
    success = test_result.count(True)
    print(f"🧪 Test result for {file}: {success} out of {len(test_result)} correct")
