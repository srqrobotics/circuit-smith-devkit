import os
import json
import cv2

def generate_sensor_bible(output_folder, output_file="sensorBible.json"):
    components = []

    # Get all PNG files
    for filename in os.listdir(output_folder):
        if not filename.endswith(".png"):
            continue

        base_name = filename[:-4]  # Strip .png
        image_path = os.path.join(output_folder, filename)
        json_path = os.path.join(output_folder, f"{base_name}.json")

        if not os.path.exists(json_path):
            print(f"⚠ Skipping {base_name}: Missing JSON file.")
            continue

        # Load image to get width and height
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠ Skipping {base_name}: Image could not be loaded.")
            continue

        height, width = image.shape[:2]

        component_entry = {
            "id": base_name,
            "name": base_name,
            "x": 0,
            "y": 0,
            "rotation": 0,
            "image": {
                "src": f"./packages/Modules/{base_name}.png",
                "width": width,
                "height": height
            },
            "pin-map": {
                "src": f"./packages/Modules/{base_name}.json"
            }
        }

        components.append(component_entry)

    # Final JSON structure
    sensor_bible = {
        "components": components
    }

    # Save to file
    with open(output_file, "w") as f:
        json.dump(sensor_bible, f, indent=4)

    print(f"✅ Generated {output_file} with {len(components)} components.")


# Run this
if __name__ == "__main__":
    generate_sensor_bible("dev-boards")
