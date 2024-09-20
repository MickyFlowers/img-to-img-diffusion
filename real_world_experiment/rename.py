import os

directory = "../data/real_world_data_4/img"  # Replace with the path to your directory

# Get a list of all the jpg files in the directory
jpg_files = [file for file in os.listdir(directory) if file.endswith(".jpg")]
# Sort the jpg files
extrta_number = lambda file: int(file.split(".")[0].split("-")[1])
jpg_files.sort(key=extrta_number)

# Rename the jpg files
for i, file in enumerate(jpg_files):
    new_name = f"{i:05d}.jpg"
    os.rename(os.path.join(directory, file), os.path.join(directory, new_name))
