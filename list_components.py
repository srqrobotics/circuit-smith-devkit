import os


def get_matching_filenames(folder_path):
    files = os.listdir(folder_path)  # List all files in the folder
    base_names = set()  # To store unique base names

    # We will use a set to ensure each base name is unique
    for file in files:
        # Split the file name and its extension
        name, ext = os.path.splitext(file)

        # Check if both .json and .png exist for the same name
        if ext == '.json' or ext == '.png':
            if f"{name}.json" in files and f"{name}.png" in files:
                base_names.add(name)

    return list(base_names)


# Example usage
folder_path = 'ref'  # Replace with your folder path
result = get_matching_filenames(folder_path)
print(result)
