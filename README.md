<img src="https://i.imgur.com/ChMev9C.jpeg" alt="MLH-banner" width="100%" height="400px">

# AWS-MachineLearningProject

# Step 1: Setting Jupiter Lab in AWS and downloading the dataset of images I created in Kaggle

<h2>⭐ Step 1: Once you are in your Jupyter lab, use the following codes to enter on the cells</h2> 
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



<h2>⭐ Step 2: Visualization - Python modules and Pandas module</h2>

# (Visualization 1)

First, we are going to create a table to check the number of images in each folder of test, train, and val. 
In this case, we will be making a table where the columns are "📁Folder Type", "🫎Animal Type", and "🗄️File Path".
In this case, we will use "glob" module to get the full path of each image from all the folders "train", "test", and "val". Using the full path we will use that to save each column information in an array.
In this case, one array for "📁Folder Type" one for "🫎Animal Type" and one for "🗄️File Path".
After saving all this information in their respective array, we will use pandas modules to order these array items in a table.

This is the code I am using:

```bash
import glob
import pandas as pd

folder ="./data/*/*.jpeg"

# store the information in a list
folder_types =[]
animal_types = []
file_paths =[]

all_files=glob.glob(folder)

for filename in all_files:
    if "test" in filename:
        if "turtle" in filename:
            folder_types.append("test")
            animal_types.append("turtle")
            file_paths.append(filename)
        elif "penguin" in filename:
            folder_types.append("test")
            animal_types.append("penguin")
            file_paths.append(filename)
    elif "train" in filename:
        if "turtle" in filename:
            folder_types.append("train")
            animal_types.append("turtle")
            file_paths.append(filename)
        elif "penguin" in filename:
            folder_types.append("train")
            animal_types.append("penguin")
            file_paths.append(filename)
    elif "val" in filename:
        if "turtle" in filename:
            folder_types.append("val")
            animal_types.append("turtle")
            file_paths.append(filename)
        elif "penguin" in filename:
            folder_types.append("val")
            animal_types.append("penguin")
            file_paths.append(filename)
all_data = pd.DataFrame({"📁Folder Type":folder_types,"🫎Animal Type":animal_types,"🗄️File Path":file_paths})
print(all_data)
```

Once it is Run, you will get this output:

<img src="https://i.imgur.com/LSlBJwC.png" alt="MLH-banner" width="550px" height="250px">

# (Visualization 2)

Continuing from the First Visualization, we will be going to do a graph visualization.

We will be using "seaborn" module to do this visualization. In this case, seaborn will work with pandas in order to do this using the current columns we already specified here "{"📁Folder Type":folder_types,"🫎Animal Type":animal_types,"🗄️File Path":file_paths}", so we will be using only the names.

Before continuing, let's download the module using:

```bash
!pip install seaborn
```

Now that we have installed the module, now I can continue talking about this module 😅
Ok, so as I was saying, we are going to use the names of the columns we already specified with pandas.

We will use the following code (I made the code to not use emojis to avoid errors🫠😭):

```bash
import seaborn as sns
# Renaming columns to avoid using emojis
all_data = all_data.rename(columns={"🫎Animal Type": "Animal Type", "📁Folder Type": "Folder Type"})

bar_graph = sns.catplot(x="Animal Type", hue="Animal Type", col="Folder Type", kind="count", palette="ch:.55", data=all_data, legend=False)
```
Description of Code:
- So, we will first import the "seaborn" in order to use it for the creation of the bar graph. In this case, we will use "catplot" where we need to specify x,hue,col,kind,palette, data,and,legend. (catplot might be capable for more parameters but for these bar charts we will only use the one I mentioned). In this case, for the x-coordinate we will specify the column "🫎Animal Type". For "hue" parameter, this means the color encoding (hue) will be based on the same variable as the x-axis, which is "🫎Animal Type". The col specifies the column of each bar graph depending on the "📁Folder Type", the folders "test", "train","val". For "Kind", in this case, we want the count of the number of images of each animal. The "pallete" is just for the type of style of the bar chart, in this case I used "ch:.55" but you can use a different one. For data, as I said before, we are using the variable that stored the table created by pandas. And "Legend" is false because it would be redundant because when you use hue, seaborn automatically adds a legend.

Once it is Run, you will get this output:

<img src="https://i.imgur.com/s7zJqtB.png" alt="MLH-banner" width="550px" height="250px">

Right now as you can see, it is very difficult to actually determine the number of images of each graph.

Therefore, we will be adding labels with the actual quantities. We will update the code like this:

```bash
import seaborn as sns
# Renaming columns to avoid using emojis
all_data = all_data.rename(columns={"🫎Animal Type": "Animal Type", "📁Folder Type": "Folder Type"})

bar_graph = sns.catplot(x="Animal Type", hue="Animal Type", col="Folder Type", kind="count", palette="ch:.55", data=all_data, legend=False)

# ensuring that the loop iterates only over the existing columns of the FacetGrid.
folder_types = all_data["Folder Type"].unique()

# Ensure that the loop iterates over the correct number of columns
for column in range(len(folder_types)):

    subplot = bar_graph.facet_axis(0, column)
    for patch in subplot.patches:
        height = patch.get_height()
        subplot.text(patch.get_x() + patch.get_width() / 2., 
                     height + 0.5, 
                     '{:1.2f}'.format(height), 
                     ha="center")
```

We are iterating through each possible "Folder Type" and adding the label of each bar of each column.

Once it is Run, you will get this output:

<img src="https://i.imgur.com/jyQdnKt.png" alt="MLH-banner" width="550px" height="250px">



<h2>⭐ Step 3: Training Model - Using S3 buckets and Jupyter for training with help of lst file</h2>

Now that we have the right format, there's just one last step before training our model – and yes, it's an important one! 🫠

In AWS Jupyter, we need to create a .lst file. This is a simple yet crucial file format that helps our model find and learn from each image stored in the S3 bucket. Think of it as a map, guiding the model to the right data.

Here's how our .lst file should look (we will do this with pandas module 🐼):

| Identifier ID | Class Category | S3 Image Path                  |
|---------------|----------------|--------------------------------|
| 1             | 0              | `<path_to_turtle_image>`       |
| 5             | 3              | `<path_to_cat_image>`          |
| 9             | 0              | `<path_to_another_turtle_image>` |
| 2             | 1              | `<path_to_penguin_image>`      |
| 6             | 2              | `<path_to_crocodile_image>`    |

Each line in this file represents an image, where:
 • Identifier ID is a unique number for each image.
 • Class Category is the label or category of the image.
 • S3 Image Path is the full path to the image in the S3 bucket.

So, below the code cell we were working previously and we will use this code in order to create the lst file using the dataset I provided from my Kaggle:

```bash
import glob
import pandas as pd
import os

test_folder = "./data/test/*.jpeg"
test_lst = pd.DataFrame(columns=["labels","s3_path"], dtype=object)
test_images = glob.glob(test_folder)
counter =0
class_arg=""

for i in test_images:
    if "turtle" in i:
        class_arg=1
    else:
        class_arg=0
    test_lst.loc[counter]=[class_arg,os.path.basename(i)]
    counter+=1
print(test_lst)
```

The code is essentially using the same techniques we were using to create tables using pandas. In this case, the DataFrame will already create automatically the column for the "Identifier Id" by giving an id in increment order of 1,2,3, etc.

Then, we made the two remaining columns "labels" that represent the "Class Category" column and we created s3_path" representing "S3 Image Path".

We are adding the values using this line "test_lst.loc[counter]=[class_arg,os.path.basename(i)]".

Once it is Run, you will get this output:

<img src="https://i.imgur.com/pE9y1hr.png" alt="MLH-banner" width="550px" height="250px">

We will do the same for train images.

```bash
import glob
import pandas as pd
import os

train_folder = "./data/train/*.jpeg"
train_lst = pd.DataFrame(columns=["labels","s3_path"], dtype=object)
train_images = glob.glob(train_folder)
counter =0
class_arg=""

for i in train_images:
    if "turtle" in i:
        class_arg=1
    else:
        class_arg=0
    train_lst.loc[counter]=[class_arg,os.path.basename(i)]
    counter+=1
print(train_lst)
```

Finally, we have to convert these tables into a CSV. In this case, lst files are similar to CSV files but are tab-separated. Therefore, when saving the lst file we are going to specify tab separated and these files will be saved in the root directory of the Jupyter Lab:

```bash
def save_as_lst_file(df,prefix):
    return df[["labels","s3_path"]].to_csv(
        f"{prefix}.lst",sep="\t",index=True,header=False
    )
save_as_lst_file(train_lst.copy(),"train")
save_as_lst_file(test_lst.copy(),"test")
```



