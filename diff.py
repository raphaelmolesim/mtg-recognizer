import cv2
import numpy as np

def resize_to_smaller(img1, img2):
    # Get dimensions of both images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Determine the smaller dimensions
    min_height = min(h1, h2)
    min_width = min(w1, w2)

    # Resize both images to the smaller dimensions
    img1_resized = cv2.resize(img1, (min_width, min_height))
    img2_resized = cv2.resize(img2, (min_width, min_height))

    return img1_resized, img2_resized

def calculate_pixel_difference(img1, img2):
    # Calculate absolute difference
    diff = cv2.absdiff(img1, img2)
    return diff

def main(image_path1, image_path2):
    # Load images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    if img1 is None or img2 is None:
        print("Error: One of the image paths is incorrect or the image could not be loaded.")
        return

    # Resize images
    img1_resized, img2_resized = resize_to_smaller(img1, img2)

    # Calculate pixel difference
    diff = calculate_pixel_difference(img1_resized, img2_resized)
    
    # print an array of the differences
    print(diff)

if __name__ == "__main__":
    # Replace with your image paths
    main('tmp\screenshot\card_0_3.png', r'dataset\\images\\Nightwhorl Hermit.jpg')
