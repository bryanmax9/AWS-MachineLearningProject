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

I messed up an image folder where instead of "test" the name starts with "train" inside the test folder. We will use this code to fix that:

```bash
import os
import glob

def rename_images(directory):
    # Path to the directory
    path = directory

    # Pattern to match files starting with 'train_turtles'
    pattern = "train_turtles*"

    # Iterate over all files matching the pattern
    for file in glob.glob(os.path.join(path, pattern)):
        # Construct the new file name by replacing 'train' with 'test'
        new_file_name = file.replace('train_turtles', 'test_turtles')

        # Renaming the file
        os.rename(file, new_file_name)
        print(f"Renamed '{file}' to '{new_file_name}'")

# Specify the directory where the images are located
directory = "/data/test/"
rename_images(directory)
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



<h2>⭐ Step 3: Creating LST file for Training Model - creating lst Data Frames for training</h2>

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

<h2>⭐ Step 4: Creating S3 Bucket and syncing to Jupyter - setting S3 for training</h2>

In the AWS Dashboard go to "Amazon S3" and create a new bucket. Name the S3 Bucket as you want, mine will be named "penguin-turtle-ai" and then click "Create bucket". 

<img src="https://i.imgur.com/unXG3X0.png" alt="MLH-banner" width="550px" height="250px">


Once the bucket is created, click on it.

![s3-bucket-2](https://github.com/bryanmax9/AWS-MachineLearningProject/assets/69496341/ad3357dc-db5f-42d7-bb53-2ea9c2d0d9c3)

Then click on the "properties" tab. There you will need what is circled in red, so don't close that tab.

![s3-bucket-3](https://github.com/bryanmax9/AWS-MachineLearningProject/assets/69496341/19ae832d-3fcf-4b3f-b517-a9ab4a3ef366)

Now in a new cell of Jupyter, we are going to store in variables the "bucket_name", "bucket_region", and "bucket_Arn" like this:

```bash
bucket_name="penguin-turtle-ai"
bucket_region="us-east-1"
bucket_Arn="arn:aws:s3:::penguin-turtle-ai"
```

Now, we are going to retrieve the environment variable  of the default S3 bucket name and replace it with the name of our s3 bucket name "penguin-turtle-ai" 

we will run this code in the next Jupyter cell:

```bash
import os

os.environ["DEFAULT_S3_BUCKET"]=bucket_name
```

After that, we will upload our train and test folder with the images to our S3 bucket.

In the next two cells run both of these commands (These commands use the paths of the folders we retrieved from my Kaggle so this might be different for you):

```bash
!aws s3 sync ./data/train s3://${DEFAULT_S3_BUCKET}/train/
```

and 

```bash
!aws s3 sync ./data/test s3://${DEFAULT_S3_BUCKET}/test/
```

Now that the folders with the images are already uploaded to the S3 Bucket. We can now upload the lst files we created by using this code:

```bash
import boto3

# Create a session
session = boto3.Session()

# Get the S3 resource
s3_resource = session.resource('s3')

# Upload the files
s3_resource.Bucket(bucket_name).Object("train.lst").upload_file("./train.lst")
s3_resource.Bucket(bucket_name).Object("test.lst").upload_file("./test.lst")
```

👻Now we are almost There

From the open browser window  I told you not to close 👀

Inside your bucket, click the "Objects" tab and reload the page and you will see the uploaded folders and files just like this:

![s3-bucket-4](https://github.com/bryanmax9/AWS-MachineLearningProject/assets/69496341/d64fcd3b-e1d7-4a4b-89af-768c843d2ba8)

<h2>⭐ Step 5: Training model with Sagemaker - setting a SageMaker session and training</h2>

First, we will create a "SageMaker" session to train our model. In this case, the Python code below will create a sagemaker session using our specifications. In this case, we are first retrieving the URI of a Docker image in ECR for the specified machine learning framework we specified as the "image-classification" framework. We specified the region since Docker image URIs are region-specific.

Overall, we are  setting up a SageMaker session and preparing to use an image classification algorithm from SageMaker's set of built-in algorithms. Also, we need to specify the S3 location for storing the output of the model.

```bash
import sagemaker
from sagemaker import image_uris
import boto3
from sagemaker import get_execution_role
sess= sagemaker.Session()

algorithm_image=image_uris.retrieve(
    region=boto3.Session().region_name,
    framework="image-classification"
)

s3_output_location = f"s3://{bucket_name}/models/image_model"
```

Now we will allow SageMaker to have access to our AWS

The "get_execution_role()" retrieves the AWS Identity and Access Management (IAM) role that was assigned to the SageMaker notebook instance or the SageMaker training job. This role is used to grant necessary permissions to SageMaker for accessing AWS resources like S3 buckets, executing training jobs, deploying models, and more. The role should have the necessary policies attached to it to allow these actions.

Therefore we will use this code line to store the role:

```bash
role =  get_execution_role()
```

Now, we will set the estimator to be ready to execute a training job in SageMaker

We are going to use the "algorithm_image" variable we created to specify that we want an image classifier. For the role, we are giving the role variable we made that will permit to SageMaker to access AWS resources. For the instance count, we specified 1 EC2 instance since we are using not much data for the training. The instance type is "ml.p2.xlarge" which is a specific type of instance optimized for machine learning purposes. Volume size 50 with a max run for the maximum run time until it fails. The input mode is "File", this mode means that SageMaker copies the training dataset from S3 to the local file system before starting the training job. The "output_path"=s3_output_location will specify the S3 location for saving the training result after the training job. The "sagemaker_session=sess" specifies the SageMaker session object, which manages interactions with the Amazon SageMaker APIs and any other AWS services needed. The session is used to execute the training job.

So the following code would be:

```bash
img_classifier_model= sagemaker.estimator.Estimator(
    algorithm_image,
    role=role,
    instance_count=1,
    instance_type="ml.p2.xlarge",
    volume_size=50,
    max_run=432000,
    input_mode="File",
    output_path=s3_output_location,
    sagemaker_session=sess
)
```

Now, we will need to finalize the image-classifier configurations. All of these configurations are needed for the trained model to be as accurate as possible. 

So the next 2 separate code cells would be, run one by one and wait for the first one to finish:

```bash
import glob

count_files=0

for filepath in glob.glob("./data/train/*.jpeg"):
    count_files+=1
print(count_files)
```

after that one finishes running, run:

```bash

img_classifier_model.set_hyperparameters(
    image_shape="3,224,224",
    num_classes=2,
    use_pretrained_model=1,
    num_training_samples=count_files,
    epochs=15,
    early_stopping=True,
    early_stopping_min_epochs=8,
    early_stopping_patience=5,
    early_stopping_tolerance=0.0,
    lr_scheduler_factor=0.1,
    lr_scheduler_step="8,10,12",
    augmentation_type="crop_color_transformation"
)
```

- We are essentially first getting the number of images in the train folder and setting the final settings for the image classifier model. We specify in "image shape" 3 since we use RGB of 3 dimensions and the 224,224 to depict the image dimensions. we are using only turtles and penguins then we set "num_classes" to 2. "use_pretrained_model=1" uses a pre-trained model, which is a common practice to leverage pre-learned features. We specify the "num_training_samples" to be the number of images in the train folder. "epochs=15" means that the entire dataset will be passed through the neural network 15 times, this might be more if you have a larger enterprise dataset. The "early_stopping=True" will enable early stopping, a method used to prevent overfitting by stopping training when a monitored metric has stopped improving, which is a default. "early_stopping_min_epochs=8" is the minimum number of epochs to run before early stopping can be initiated, this will also be different depending on the epoch number. The "early_stopping_patience=5" is the number of epochs with no improvement after which training will be stopped, this also might vary depending on your number of epochs. The "early_stopping_tolerance=0.0" is the tolerance for early stopping, so we set it to 0.0 to stop immediately when the condition is met.
The "lr_scheduler_factor=0.1" is the factor by which the learning rate will be reduced. The last one "lr_scheduler_step="8,10,12" is the epoch numbers at which learning rate reduction will happen, this might also differ and change depending on the amount of epoch. Finally, the line "augmentation_type="crop_color_transformation"" will make that during each epoch of the training (or pass through the data), the model will see slightly different versions of each image, due to the random cropping and color transformations. This can prevent the model from overfitting to the exact details of the training images and improve its ability to generalize to new images.

Finally

we will set the HyperParameter Range:

```bash
from sagemaker.tuner import CategoricalParameter, ContinuousParameter, HyperparameterTuner

hyperparameter_ranges={
    "learning_rate":ContinuousParameter(0.01,0.1),
    "mini_batch_size":CategoricalParameter([8,16,32]),
    "optimizer":CategoricalParameter(["sgd","adam"])
}
```

- The hyperparameter_ranges variable we will store a dictionary, where each key-value pair represents a specific hyperparameter and its possible range of values. For the "learning_rate" hyperparameter, we are setting a ContinuousParameter class used to define a continuous numeric range from 0.01 to 0.1. This means that the learning rate can take any real-numbered value within this specified range, allowing for fine-grained adjustments to how the model learns from the data. Moreover, the "mini_batch_size" hyperparameter will use the CategoricalParameter class that will only take discrete values from the set [8, 16, 32]. This approach is used for hyperparameters that have specific, predefined options. Ultimately, the choice of batch sizes often comes down to empirical testing. In this case, the set [8, 16, 32] provides a range of sizes that are commonly used in practice. You would use hyperparameter tuning to test these different values and observe which one results in the best performance for your specific application. Similarly, the "optimizer" hyperparameter is also defined as a categorical parameter, with the options being either 'sgd' (stochastic gradient descent) or 'adam' (an optimizer that combines features of other optimization algorithms). The goal is to achieve the best performance based on a specified objective, such as minimizing loss or maximizing accuracy.


<h1>Before starting our Training job</h1>

We have to first go to the AWS Dashboard and search for "Service Quotas":

![service quotas](https://github.com/bryanmax9/AWS-MachineLearningProject/assets/69496341/5c53a497-e20b-48ee-9a7b-32d3a1b9de8b)


In "Manage Quotas" search for "Amazon SageMaker" and click on "View quotas":

![service quotas 2](https://github.com/bryanmax9/AWS-MachineLearningProject/assets/69496341/a125d0c0-2c51-4234-b562-55a4a7e23e5c)

In "Service quotas" search for "ml.p2.xlarge for training job usage" and click on it:

![service quotas 3](https://github.com/bryanmax9/AWS-MachineLearningProject/assets/69496341/26baae63-c9e8-4788-968d-0f79048ad2c1)

Now click on "Request quota increase":

![service quotas 4](https://github.com/bryanmax9/AWS-MachineLearningProject/assets/69496341/1bb01597-1ba8-405b-8888-25e42e6fa2b8)

Specify 1 quota since we are just doing 1 for this project, then click "request":

![service quotas 5](https://github.com/bryanmax9/AWS-MachineLearningProject/assets/69496341/c1fd0939-296d-47e7-9d3f-999890d0d7d2)


Now, Return to the Jupyter Lab and we will use this final code to initialize the Training Job:

```bash
from sagemaker.session import TrainingInput
import time

objective_metric_name = "validation:accuracy"
objective_type = "Maximize"
max_jobs=5
max_parallel_jobs=1

tuner=HyperparameterTuner(estimator=img_classifier_model,
                         objective_metric_name=objective_metric_name,
                         hyperparameter_ranges=hyperparameter_ranges,
                         objective_type=objective_type,
                         max_jobs=max_jobs,
                         max_parallel_jobs=max_parallel_jobs
                         )

model_inputs={
    "train":sagemaker.inputs.TrainingInput(s3_data=f"s3://{bucket_name}/train/", content_type="image/jpeg"),
    "validation":sagemaker.inputs.TrainingInput(s3_data=f"s3://{bucket_name}/test/", content_type="image/jpeg"),
    "train_lst":sagemaker.inputs.TrainingInput(s3_data=f"s3://{bucket_name}/train.lst", content_type="image/jpeg"),
    "validation_lst":sagemaker.inputs.TrainingInput(s3_data=f"s3://{bucket_name}/test.lst", content_type="image/jpeg"),
}

job_name_prefix="classifier"
timestamp=time.strftime("-%Y-%m-%d-%H-%M-%S", time.gmtime())
job_name=job_name_prefix+timestamp

tuner.fit(inputs=model_inputs,job_name=job_name,logs=True)

```

- This code snippet sets up and starts a hyperparameter tuning job using Amazon SageMaker. The HyperparameterTuner object is configured to optimize the img_classifier_model, which is a predefined image classification model. The goal of the tuning job is to maximize the model's accuracy on a validation dataset (validation:accuracy).

Key configurations for the tuning job include:

    max_jobs: The total number of different hyperparameter combinations that the tuner will evaluate is set to 5.
    max_parallel_jobs: Specifies that only one hyperparameter tuning job should be run at a time.

The model_inputs dictionary defines the data sources for the tuning job, with training and validation datasets as well as their corresponding list files (.lst) specified by their paths in an S3 bucket. These inputs are expected to be JPEG image files.

Finally, the tuning job is given a unique name by appending a timestamp to the job_name_prefix, which in this case, is "classifier". The tuner.fit method then starts the tuning job with the specified inputs and job name, with logs enabled for monitoring.

<h1>Monitor Training</h1>

Now, let's go to the AWS dashboard and search for "Cloud watch":

![cloud watch](https://github.com/bryanmax9/AWS-MachineLearningProject/assets/69496341/031246fe-afc0-4aed-9196-41f2d784f7e5)

Then on the left side go to "Logs" and click on "Log groups". In there click on "/aws/sagemaker/TrainingJobs" that is circled in red:

![cloud watch 2](https://github.com/bryanmax9/AWS-MachineLearningProject/assets/69496341/3255647c-deac-4881-a842-a0433805c3e3)




