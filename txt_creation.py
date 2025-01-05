import os

# Define the directory where your images are
directory = r'/home/lamda/Desktop/AIP/L-CRP/datasets/coco2017/coco/val'

# Filter for images
image_files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png', '.txt'))]  # add other image types if needed
image_paths = [os.path.join(directory, image_file) for image_file in image_files]

# Write to train2017.txt
with open(os.path.join(directory, 'val2017.txt'), 'w') as file:
    for image_path in image_paths:
        file.write(image_path + '\n')