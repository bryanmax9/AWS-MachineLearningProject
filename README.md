<img src="https://i.imgur.com/ChMev9C.jpeg" alt="MLH-banner" width="100%" height="400px">

# AWS-MachineLearningProject

# Step 1: Setting Jupiter Lab in AWS and downloading the dataset of images I created in Kaggle

<h2>⭐ Once you are in your Jupyter lab, use the following codes to enter on the cells</h2> 
On the first 3 cells, we will use "!" to tell Jupiter that we are executing a command line command. Otherwise, if it doesn't have "!" than it is python code.
This command will install Kaggle:

```bash
!pip install -q kaggle
```
and 

```bash
!mkdir ~/.kaggle
```
and 

```bash
!touch ~/.kaggle/kaggle.json
```

Now, you will need to create a Kaggle account on the official website and get your API Key.

Once you get your API key, you will need to put it in the next cell:

```bash
api_token ={"username":"your_user","key":"your_key"}
```

Then, We will permit Kaggle to download datasets from the internet.
Run this code:
```bash
import json
import os

# Use a directory where you have write permissions
directory = os.path.expanduser("~/.kaggle/")
os.makedirs(directory, exist_ok=True)
file_path = os.path.join(directory, "kaggle.json")

with open(file_path, 'w') as file:
    json.dump(api_token, file)

# Make sure to adjust permissions of the file
!chmod 600 {file_path}

```

Now, is the time to download the dataset. 
In my case, I am using the dataset I created and published in Kaggle "https://www.kaggle.com/datasets/bryanmax9/penguin-and-turtles-dataset".
In this case, to download it we need to extract from the URL this part "bryanmax9/penguin-and-turtles-dataset".
The command will be:

format: !kaggle datasets download -d <dataset-name> --force

```bash
!kaggle datasets download -d bryanmax9/penguin-and-turtles-dataset --force
```

Now, lets extract the Zip file using the following code. The following code will extract all to the current directory:

```bash
import zipfile

with zipfile.ZipFile('./penguin-and-turtles-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
```

All commands will look like this in your Jupiter lab, execute from top to bottom one by one:
<img src="https://i.imgur.com/0M1jCay.png" alt="MLH-banner" width="100%" height="400px">


> **Note:** If you are using another image dataset, use this code to check the image.
> 
> - You will need to check one of the images, in this case, I am checking the first image of my dataset
> 
> ```bash
> from PIL import Image
> 
> image = Image.open("data/train/train_penguin0.jpeg")
> print(image.format)
> print(image.size)
> print(image.mode)
> ```
> 
> > The output will be:
> >
> > ![MLH-banner](https://i.imgur.com/TH5Dpzg.png)
> 
> - I had to modify the format, size, and mode of the images on my dataset in order to be able to be trained. Therefore, the images have to be JPEG, 224x224, and RGB in order to use it to train our AI model.
> 
> This is an example code snippet that I used in order to accomplish that (The script is also in the repo folder):
> 
> ```bash
> import glob
> from PIL import Image
> import os
> 
> # Path to the images
> folder = "./data/train/pics/*/*.jpg"
> count_penguin = 0
> count_turtle = 0
> image_paths = glob.glob(folder)
> 
> # Ensure the train directory exists
> os.makedirs("./data/val", exist_ok=True)
> 
> for image_path in image_paths:
>     image = Image.open(image_path)
>     image = image.resize((224, 224))
> 
>     # Define the base path for saving images
>     base_save_path = "./data/val/"
>     
>     # Debugging output to see the current image path
>     print(f"Current image path: {image_path}")  
> 
>     # Check if the image path contains the keyword 'penguin' or 'turtle'
>     if "penguin" in image_path:
>         save_path = f"{base_save_path}val_penguin{count_penguin}.jpeg"
>         image.save(save_path, "JPEG")
>         print(f"Saved penguin image to: {save_path}")  # Debugging output
>         count_penguin += 1
>     elif "turtle" in image_path:
>         save_path = f"{base_save_path}val_turtle{count_turtle}.jpeg"
>         image.save(save_path, "JPEG")
>         print(f"Saved turtle image to: {save_path}")  # Debugging output
>         count_turtle += 1
> ```
