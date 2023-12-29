import os

def rename_images(folder_path):
    # Get all files in the folder
    files = os.listdir(folder_path)

    # Filter out only image files if needed (e.g., jpg, png)
    # Uncomment the following line if you want to filter by file extension
    # files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]

    # Rename each file
    for i, filename in enumerate(files):
        # Create the new file name
        new_name = f"penguin_{str(i).zfill(2)}"

        # Get the file extension (e.g., .jpg, .png)
        file_extension = os.path.splitext(filename)[1]

        # Create the full path for the old and new names
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name + file_extension)

        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed {old_file} to {new_file}")

# Replace 'turtle-pics' with the correct path if it's different
rename_images('penguin-pics')
