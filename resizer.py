import json
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def show_image_with_grid(png_path, grid_divisions):
    image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image: {png_path}")

    height, width = image.shape[:2]
    grid_size_x = width // grid_divisions
    grid_size_y = height // grid_divisions

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_xticks(np.arange(0, width, grid_size_x))
    ax.set_yticks(np.arange(0, height, grid_size_y))
    ax.grid(color='r', linestyle='-', linewidth=0.5)
    plt.show()


def resize_image_and_adjust_coordinates(png_path, json_path, scale_ratio, output_png_path="output.png",
                                        output_json_path="updated_coordinates.json"):
    # Load image
    image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image: {png_path}")

    original_height, original_width = image.shape[:2]
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)

    # Resize image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Save resized image
    cv2.imwrite(output_png_path, resized_image)

    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Adjust coordinates
    if "digital-pins" in data and "reloc" in data["digital-pins"]:
        for pin in data["digital-pins"]["reloc"]:
            pin["points"][0] = int(pin["points"][0] * scale_ratio)
            pin["points"][1] = int(pin["points"][1] * scale_ratio)

    # Save updated JSON
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Resized image saved to: {output_png_path}")
    print(f"Updated JSON saved to: {output_json_path}")


if __name__ == '__main__':
    device = "ultrasonic-SR04"
    image_path = f"ref/{device}.png"
    json_path = f"ref/{device}.json"

    # grid_divisions = int(input("Enter number of grid divisions: "))
    grid_divisions = 10
    show_image_with_grid(image_path, grid_divisions)

    scale_ratio = float(input("Enter scale ratio: "))
    resize_image_and_adjust_coordinates(
        image_path,
        json_path,
        scale_ratio,
        f"dev-boards/{device}.png",
        f"dev-boards/{device}.json"
    )
