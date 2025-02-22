import cv2
import json
import numpy as np
from typing import List, Dict
import os
import pytesseract

class PinMapper:
    def __init__(self):
        self.image = None
        self.display_image = None
        self.pin_locations: List[Dict] = []  # For pins only
        self.label_locations: List[Dict] = []  # For OCR text labels
        self.linked_pins: List[Dict] = []  # For final linked pin-label pairs
        self.window_name = "Pin Mapper"
        self.zoom_factor = 1.0
        self.drag_start = None
        self.offset = (0, 0)
        self.current_pin = 0
        self.current_label = 0
        # Rectangle drawing state
        self.drawing_rect = False
        self.rect_start = None
        self.rect_end = None
        
        # Add undo history
        self.history: List[Dict] = []  # Store actions for undo
        self.crop_rect = None  # Store crop rectangle coordinates
        self.cropped_image = None  # Store cropped image for background removal
        self.is_cropped = False

    def load_image(self, img_path: str) -> bool:
        """Load the image from the specified path."""
        try:
            self.image = cv2.imread(img_path)
            if self.image is None:
                raise ValueError("Image could not be loaded")
            self.display_image = self.image.copy()
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False

    def get_original_coordinates(self, x: int, y: int) -> tuple:
        """Convert screen coordinates to original image coordinates."""
        orig_x = int((x - self.offset[0]) / self.zoom_factor)
        orig_y = int((y - self.offset[1]) / self.zoom_factor)
        return orig_x, orig_y

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zooming, panning and pin selection."""
        orig_x, orig_y = self.get_original_coordinates(x, y)
        
        # Show alignment guides whenever Alt is pressed
        if flags & cv2.EVENT_FLAG_ALTKEY:
            self.rect_end = (orig_x, orig_y)
            self.draw_current_state(flags)
        
        if event == cv2.EVENT_MOUSEWHEEL:
            # Calculate cursor position relative to image content
            old_zoom = self.zoom_factor
            
            # Update zoom factor
            if flags > 0:  # Scroll up to zoom in
                self.zoom_factor *= 1.1
            else:  # Scroll down to zoom out
                self.zoom_factor /= 1.1
            
            # Calculate offset adjustment to keep cursor position fixed
            height, width = self.image.shape[:2]
            cursor_x = (x - self.offset[0]) / old_zoom
            cursor_y = (y - self.offset[1]) / old_zoom
            
            new_x = cursor_x * self.zoom_factor
            new_y = cursor_y * self.zoom_factor
            
            self.offset = (
                self.offset[0] - (new_x - (x - self.offset[0])),
                self.offset[1] - (new_y - (y - self.offset[1]))
            )
            
            self.update_display()
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:  # Shift + Left click for OCR
                self.drawing_rect = True
                self.rect_start = (orig_x, orig_y)
            elif flags & cv2.EVENT_FLAG_CTRLKEY:  # Ctrl + Left click for linking
                self.drawing_rect = True
                self.rect_start = (orig_x, orig_y)
            elif flags & cv2.EVENT_FLAG_ALTKEY:  # Alt + Left click for cropping
                self.drawing_rect = True
                self.rect_start = (orig_x, orig_y)
            else:  # Normal left click for pin placement
                if self.is_cropped:
                    # Convert coordinates relative to cropped image
                    x1, y1 = self.crop_rect[0:2]  # Get crop region start coordinates
                    pin_x = orig_x
                    pin_y = orig_y
                    
                    # Ensure coordinates are within cropped bounds
                    if (x1 <= orig_x <= self.crop_rect[2] and 
                        y1 <= orig_y <= self.crop_rect[3]):
                        pin_data = {
                            "pin_number": self.current_pin,
                            "x": pin_x,
                            "y": pin_y
                        }
                        self.pin_locations.append(pin_data)
                        self.add_to_history("pin", pin_data)
                        print(f"Added pin {self.current_pin} at ({pin_x}, {pin_y})")
                        self.current_pin += 1
                        self.draw_current_state()
                    else:
                        print("Pin must be placed within the cropped region")
                else:
                    print("Please crop the image first (Alt + Left click and drag)")
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_rect and self.rect_start:
                self.rect_end = (orig_x, orig_y)
                self.draw_current_state(flags)
            elif flags & cv2.EVENT_FLAG_RBUTTON:  # Right button drag for panning
                if self.drag_start is not None:
                    dx = x - self.drag_start[0]
                    dy = y - self.drag_start[1]
                    self.offset = (self.offset[0] + dx, self.offset[1] + dy)
                    self.drag_start = (x, y)
                    self.update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing_rect and self.rect_start and self.rect_end:
                if flags & cv2.EVENT_FLAG_SHIFTKEY:  # OCR
                    self.process_ocr_rectangle()
                elif flags & cv2.EVENT_FLAG_CTRLKEY:  # Linking
                    self.process_linking_rectangle()
                elif flags & cv2.EVENT_FLAG_ALTKEY:  # Cropping
                    self.process_crop_rectangle()
                
                self.drawing_rect = False
                self.rect_start = None
                self.rect_end = None
                self.draw_current_state()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drag_start = (x, y)
        
        elif event == cv2.EVENT_RBUTTONUP:
            self.drag_start = None

    def process_ocr_rectangle(self):
        """Process OCR for the selected rectangle area."""
        if not self.rect_start or not self.rect_end:
            return

        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        
        # Ensure correct order of coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Extract the region of interest
        roi = self.image[y1:y2, x1:x2]
        if roi.size == 0:
            return

        try:
            # Preprocess image for better OCR
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Resize image (2x) to improve OCR accuracy
            scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Apply threshold to get black text on white background
            _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR Configuration
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_/'
            
            # Perform OCR
            text = pytesseract.image_to_string(binary, config=custom_config).strip()
            
            # Debug output
            print(f"OCR Debug - Raw text detected: '{text}'")
            
            if not text:
                text = "_PIN"
            else:
                # Clean up the text - remove any unwanted characters
                text = ''.join(c for c in text if c.isalnum() or c in ['_', '/'])
            
            label_data = {
                "label_id": self.current_label,
                "text": text,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
            self.label_locations.append(label_data)
            self.add_to_history("label", label_data)
            print(f"Added label {self.current_label}: '{text}'")
            self.current_label += 1
            
        except Exception as e:
            print(f"OCR Error: {e}")
            # Save debug image on error
            try:
                cv2.imwrite("ocr_debug.png", roi)
                print("Saved problematic image to ocr_debug.png")
            except:
                pass

    def process_linking_rectangle(self):
        """Link a pin with a label using the bounding rectangle."""
        if not self.rect_start or not self.rect_end:
            return

        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        
        # Ensure correct order of coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Find pins and labels within the rectangle
        contained_pins = []
        contained_labels = []

        for pin in self.pin_locations:
            if x1 <= pin["x"] <= x2 and y1 <= pin["y"] <= y2:
                contained_pins.append(pin)

        for label in self.label_locations:
            if (x1 <= label["x1"] <= x2 and y1 <= label["y1"] <= y2 and 
                x1 <= label["x2"] <= x2 and y1 <= label["y2"] <= y2):
                contained_labels.append(label)

        # Link if we found exactly one pin and one label
        if len(contained_pins) == 1 and len(contained_labels) == 1:
            pin = contained_pins[0]
            label = contained_labels[0]
            
            linked_data = {
                "pin_number": pin["pin_number"],
                "x": pin["x"],
                "y": pin["y"],
                "label": label["text"],
                "label_bounds": {
                    "x1": label["x1"],
                    "y1": label["y1"],
                    "x2": label["x2"],
                    "y2": label["y2"]
                }
            }
            self.linked_pins.append(linked_data)
            self.add_to_history("link", linked_data)  # Add to undo history
            print(f"Linked pin {pin['pin_number']} with label '{label['text']}'")
        else:
            print("Error: Linking rectangle must contain exactly one pin and one label")

    def process_crop_rectangle(self):
        """Process the crop rectangle and wait for pin mapping before background removal."""
        if not self.rect_start or not self.rect_end:
            return

        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        
        # Ensure correct order of coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Store crop rectangle
        self.crop_rect = (x1, y1, x2, y2)
        
        # Crop the image
        self.cropped_image = self.image[y1:y2, x1:x2].copy()
        self.is_cropped = True
        
        print("\nCrop area selected!")
        print("Now map all pins and labels, then press 'b' to start background removal")
        self.draw_current_state()

    def start_background_removal(self):
        """Start the background removal process."""
        if not self.is_cropped:
            print("Please crop the image first")
            return
            
        # Launch background removal tool
        from DevBoardMaker import DevBoardMaker
        maker = DevBoardMaker()
        
        # Save temporary image for processing
        temp_path = "temp_crop.png"
        cv2.imwrite(temp_path, self.cropped_image)
        
        # Run background removal
        output_path = f"dev-boards/{DEV_BOARD}.png"
        maker.run(temp_path, output_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Load the processed image back
        self.cropped_image = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
        self.draw_current_state()

    def draw_current_state(self, flags=0):
        """Draw the current state including crop box, pins, labels, and rectangles."""
        self.display_image = self.image.copy()
        
        # Draw alignment guides when Alt is pressed
        if flags & cv2.EVENT_FLAG_ALTKEY and self.rect_end:
            height, width = self.image.shape[:2]
            # Draw vertical guide
            cv2.line(self.display_image, 
                    (self.rect_end[0], 0), 
                    (self.rect_end[0], height), 
                    (0, 0, 255), 1)  # Red line
            # Draw horizontal guide
            cv2.line(self.display_image, 
                    (0, self.rect_end[1]), 
                    (width, self.rect_end[1]), 
                    (0, 0, 255), 1)  # Red line

        # Draw crop rectangle if exists
        if self.is_cropped:
            x1, y1, x2, y2 = self.crop_rect
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw unlinked pins
        for pin in self.pin_locations:
            x = pin["x"]
            y = pin["y"]
            cv2.circle(self.display_image, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(self.display_image, str(pin["pin_number"]), 
                       (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)

        # Draw OCR label rectangles
        for label in self.label_locations:
            # Create semi-transparent overlay for the label background
            overlay = self.display_image.copy()
            cv2.rectangle(overlay, 
                         (label["x1"], label["y1"]),
                         (label["x2"], label["y2"]),
                         (255, 0, 0), -1)  # Blue filled rectangle
            
            # Apply the overlay with transparency
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, self.display_image, 1 - alpha, 0, self.display_image)
            
            # Draw the border
            cv2.rectangle(self.display_image, 
                         (label["x1"], label["y1"]),
                         (label["x2"], label["y2"]),
                         (255, 0, 0), 1)  # Blue border
            
            # Add the OCR text inside the box
            text = label["text"]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            text_x = label["x1"] + (label["x2"] - label["x1"] - text_width) // 2
            text_y = label["y1"] + (label["y2"] - label["y1"] + text_height) // 2
            cv2.putText(self.display_image, text,
                       (text_x, text_y),
                       font, font_scale, (0, 0, 0), thickness)

        # Draw linked pins and their labels
        for linked in self.linked_pins:
            # Draw pin
            pin_x, pin_y = linked["x"], linked["y"]
            cv2.circle(self.display_image, (pin_x, pin_y), 3, (0, 0, 255), -1)
            
            # Draw label box with semi-transparent background
            bounds = linked["label_bounds"]
            overlay = self.display_image.copy()
            cv2.rectangle(overlay,
                         (bounds["x1"], bounds["y1"]),
                         (bounds["x2"], bounds["y2"]),
                         (0, 0, 255), -1)  # Red filled rectangle
            
            # Apply the overlay with transparency
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, self.display_image, 1 - alpha, 0, self.display_image)
            
            # Draw the border
            cv2.rectangle(self.display_image,
                         (bounds["x1"], bounds["y1"]),
                         (bounds["x2"], bounds["y2"]),
                         (0, 0, 255), 1)  # Red border
            
            # Add the label text inside the box
            text = linked["label"]
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = bounds["x1"] + (bounds["x2"] - bounds["x1"] - text_width) // 2
            text_y = bounds["y1"] + (bounds["y2"] - bounds["y1"] + text_height) // 2
            cv2.putText(self.display_image, text,
                       (text_x, text_y),
                       font, font_scale, (0, 0, 0), thickness)
            
            # Calculate center points
            label_center_x = (bounds["x1"] + bounds["x2"]) // 2
            label_center_y = (bounds["y1"] + bounds["y2"]) // 2
            
            # Determine whether to use vertical or horizontal line
            dx = abs(pin_x - label_center_x)
            dy = abs(pin_y - label_center_y)
            
            if dx < dy:  # Use vertical line if height difference is greater
                # Draw vertical line
                cv2.line(self.display_image,
                        (pin_x, pin_y),
                        (pin_x, label_center_y),
                        (0, 0, 255), 1)
                # Draw horizontal line if needed
                if pin_x != label_center_x:
                    cv2.line(self.display_image,
                            (pin_x, label_center_y),
                            (label_center_x, label_center_y),
                            (0, 0, 255), 1)
            else:  # Use horizontal line
                # Draw horizontal line
                cv2.line(self.display_image,
                        (pin_x, pin_y),
                        (label_center_x, pin_y),
                        (0, 0, 255), 1)
                # Draw vertical line if needed
                if pin_y != label_center_y:
                    cv2.line(self.display_image,
                            (label_center_x, pin_y),
                            (label_center_x, label_center_y),
                            (0, 0, 255), 1)

        # Draw current rectangle if being drawn
        if self.drawing_rect and self.rect_start and self.rect_end:
            cv2.rectangle(self.display_image, 
                         self.rect_start, 
                         self.rect_end, 
                         (0, 0, 255), 1)  # Red with thickness 1 for active rectangle
        
        self.update_display()

    def draw_pins(self):
        """Draw all pins on the display image."""
        self.draw_current_state()

    def update_display(self):
        """Update the display with current zoom and pan settings."""
        height, width = self.image.shape[:2]
        new_width = int(width * self.zoom_factor)
        new_height = int(height * self.zoom_factor)
        
        # Resize image according to zoom factor
        resized = cv2.resize(self.display_image, (new_width, new_height))
        
        # Create a black canvas of the window size
        display = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Ensure the offset keeps the image within bounds
        self.offset = (
            min(max(self.offset[0], width - new_width), 0),
            min(max(self.offset[1], height - new_height), 0)
        )
        
        # Calculate visible region
        y1 = max(0, int(self.offset[1]))
        y2 = min(height, int(self.offset[1] + new_height))
        x1 = max(0, int(self.offset[0]))
        x2 = min(width, int(self.offset[0] + new_width))
        
        display_y1 = max(0, int(-self.offset[1]))
        display_y2 = display_y1 + (y2 - y1)
        display_x1 = max(0, int(-self.offset[0]))
        display_x2 = display_x1 + (x2 - x1)
        
        try:
            display[y1:y2, x1:x2] = resized[display_y1:display_y2, display_x1:display_x2]
        except ValueError:
            # Reset zoom and offset if we encounter an error
            self.zoom_factor = 1.0
            self.offset = (0, 0)
            display = self.display_image.copy()
        
        cv2.imshow(self.window_name, display)

    def save_pin_locations(self, output_file: str):
        """Save pin locations in the required format."""
        try:
            if not self.is_cropped:
                raise ValueError("Image must be cropped before saving")
                
            if not self.linked_pins:
                raise ValueError("No pins to save")

            # Create the output structure
            output_data = {
                "name": DEV_BOARD.replace("-", " "),
                "type": TYPE[0],
                "digital-pins": {
                    "id": [],
                    "reloc": []
                },
                "specs": {
                    "processor": PROCESSOR[0],
                    "clockSpeed": CLOCK_SPEED[0],
                    "voltage": VOLTAGE
                }
            }

            # Add all linked pins to both id and reloc lists
            for pin in self.linked_pins:
                pin_id = pin["label"]
                
                # Convert coordinates relative to cropped image
                x1, y1 = self.crop_rect[0:2]
                rel_x = pin["x"] - x1
                rel_y = pin["y"] - y1
                
                # Add to id list
                output_data["digital-pins"]["id"].append(pin_id)
                
                # Add to reloc list with coordinates relative to cropped image
                reloc_entry = {
                    "id": pin_id,
                    "points": [rel_x, rel_y]
                }
                output_data["digital-pins"]["reloc"].append(reloc_entry)

            print(f"Saving {len(self.linked_pins)} linked pins to {output_file}")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save the JSON file
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Pin locations saved to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error saving pin locations: {e}")
            return False

    def add_to_history(self, action_type: str, data: Dict):
        """Add an action to the undo history."""
        self.history.append({
            "type": action_type,  # "pin", "label", or "link"
            "data": data
        })
        
    def undo_last_action(self):
        """Undo the last action."""
        if not self.history:
            print("Nothing to undo")
            return
            
        last_action = self.history.pop()
        action_type = last_action["type"]
        
        if action_type == "pin":
            if self.pin_locations:
                removed_pin = self.pin_locations.pop()
                self.current_pin -= 1
                print(f"Undid pin {removed_pin['pin_number']}")
        
        elif action_type == "label":
            if self.label_locations:
                removed_label = self.label_locations.pop()
                self.current_label -= 1
                print(f"Undid label {removed_label['label_id']}")
        
        elif action_type == "link":
            if self.linked_pins:
                removed_link = self.linked_pins.pop()
                print(f"Undid link between pin {removed_link['pin_number']} and label '{removed_link['label']}'")
        
        self.draw_current_state()

    def run(self, image_path: str, output_file: str):
        """Run the pin mapping process."""
        if not self.load_image(image_path):
            return

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Controls:")
        print("- Mouse wheel: Zoom in/out")
        print("- Right click and drag: Pan image")
        print("- Alt + Left click and drag: Crop image")
        print("- Left click: Add pin (after cropping)")
        print("- Shift + Left click and drag: Draw rectangle for OCR")
        print("- Ctrl + Left click and drag: Draw linking rectangle")
        print("- F: Save pins and proceed to background removal")
        print("- Ctrl + Z: Undo last action")
        print("- Press 'ESC': Exit")

        # Initial display of the image
        self.update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('f') or key == ord('F'):  # F key (case insensitive)
                if self.linked_pins:
                    try:
                        self.save_pin_locations(output_file)
                        print("Pins saved! Opening background removal tool...")
                        cv2.destroyAllWindows()
                        self.start_background_removal()
                        break
                    except Exception as e:
                        print(f"Error during save: {e}")
                else:
                    print("No pins to save! Please add and link pins first.")
            elif key == 26 or (key == ord('z') and (cv2.waitKey(1) & 0xFF == 0)):  # Ctrl + Z
                self.undo_last_action()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    DEV_BOARD = "Arduino-UNO"
    TYPE = ("microcontroller",)  # Fix tuple formatting
    PROCESSOR = ("ATmega328P",)
    CLOCK_SPEED = ("16 MHz",)
    VOLTAGE = "5V"  # Not a tuple

    mapper = PinMapper()
    
    # Create directory if it doesn't exist
    os.makedirs("dev-boards", exist_ok=True)
    
    component_path = f"ref/{DEV_BOARD}.png"
    json_file = f"dev-boards/{DEV_BOARD}.json"

    mapper.run(component_path, json_file)