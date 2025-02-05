import requests
import os
from tqdm import tqdm

# Set code for Bloomburrow
SET_CODE = "blb"  # Replace with the correct set code if needed
OUTPUT_DIR = "dataset/images"
CAPTIONS_FILE = "dataset/captions.txt"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fetch all cards from Bloomburrow
print("Fetching cards from Scryfall API...")
url = f"https://api.scryfall.com/cards/search?q=e%3A{SET_CODE}"

all_cards = []
while url:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")

    data = response.json()
    all_cards.extend(data['data'])
    url = data.get('next_page')  # Get the next page if it exists

# Save images and captions
print(f"Downloading {len(all_cards)} cards...")
with open(CAPTIONS_FILE, 'w', encoding='utf-8') as f:
    for card in tqdm(all_cards):
        # Get card name and image URL
        card_name = card['name'].replace("/", "-").replace(":", "-")

        # Handle double-faced cards
        if 'image_uris' in card:
            image_url = card['image_uris'].get('art_crop')
        elif 'card_faces' in card and 'image_uris' in card['card_faces'][0]:
            image_url = card['card_faces'][0]['image_uris'].get('art_crop')
        else:
            continue  # Skip if no image URL

        # Download image
        image_filename = f"{card_name}.jpg"
        image_path = os.path.join(OUTPUT_DIR, image_filename)

        if os.path.exists(image_path):
            # delete the existing file
            os.remove(image_path)
        
        img_data = requests.get(image_url).content
        with open(image_path, 'wb') as img_file:
            img_file.write(img_data)

        # Write to captions.txt
        f.write(f"images/{image_filename}\t{card['name']}\n")

print("Download complete! Dataset is ready.")
