import glob
from PIL import Image
import os

# Path to the images
folder = "./data/train/pics/*/*.jpg"
count_penguin = 0
count_turtle = 0
image_paths = glob.glob(folder)

# Ensure the train directory exists
os.makedirs("./data/val", exist_ok=True)

for image_path in image_paths:
    image = Image.open(image_path)
    image = image.resize((224, 224))

    # Define the base path for saving images
    base_save_path = "./data/val/"
    
    # Debugging output to see the current image path
    print(f"Current image path: {image_path}")  

    # Check if the image path contains the keyword 'penguin' or 'turtle'
    if "penguin" in image_path:
        save_path = f"{base_save_path}val_penguin{count_penguin}.jpeg"
        image.save(save_path, "JPEG")
        print(f"Saved penguin image to: {save_path}")  # Debugging output
        count_penguin += 1
    elif "turtle" in image_path:
        save_path = f"{base_save_path}val_turtle{count_turtle}.jpeg"
        image.save(save_path, "JPEG")
        print(f"Saved turtle image to: {save_path}")  # Debugging output
        count_turtle += 1
