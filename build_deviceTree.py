import os
import json

BASE_COMPONENTS_DIR = "components"
PACKAGES_DIR = "./packages"

def build_component_tree(current_path, relative_path=""):
    entries = []
    for item in sorted(os.listdir(current_path)):
        full_path = os.path.join(current_path, item)
        rel_path = os.path.join(relative_path, item)
        if os.path.isdir(full_path):
            children = build_component_tree(full_path, rel_path)
            if children:  # Only include directories with .json children
                entries.append({
                    "name": item,
                    "path": f"{os.path.join(PACKAGES_DIR, rel_path).replace(os.sep, '/').lower()}",
                    "type": "directory",
                    "children": children
                })
        elif item.endswith(".json"):
            base_name = os.path.splitext(item)[0]
            png_file = base_name + ".png"
            if png_file in os.listdir(current_path):
                # Build the path used for file
                folder_parts = os.path.normpath(relative_path).split(os.sep)
                if folder_parts[0].lower() == "microcontrollers" and len(folder_parts) >= 2:
                    package_path = f"{PACKAGES_DIR}/Microcontrollers/{folder_parts[1]}/{item}"
                else:
                    package_path = f"{PACKAGES_DIR}/Modules/{item}"
                entries.append({
                    "name": item,
                    "path": package_path,
                    "type": "file"
                })
    return entries


if __name__ == "__main__":
    result = build_component_tree(BASE_COMPONENTS_DIR)
    print(json.dumps(result, indent=2))
