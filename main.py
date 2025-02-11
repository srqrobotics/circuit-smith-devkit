import cv2
import json
import numpy as np
from typing import List, Dict
import os

class PinMapper:
    def __init__(self):
        self.image = None
        self.display_image = None
        self.pin_locations: List[Dict] = []
        self.window_name = "Pin Mapper"
        self.zoom_factor = 1.0
        self.drag_start = None
        self.offset = (0, 0)
        self.current_pin = 0

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

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zooming, panning and pin selection."""
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
        
        elif event == cv2.EVENT_LBUTTONDOWN:  # Left click to add pin
            # Convert screen coordinates to original image coordinates
            orig_x = int((x - self.offset[0]) / self.zoom_factor)
            orig_y = int((y - self.offset[1]) / self.zoom_factor)
            
            # Ensure coordinates are within image bounds
            height, width = self.image.shape[:2]
            if 0 <= orig_x < width and 0 <= orig_y < height:
                # Store the pin location in original image coordinates
                pin_data = {
                    "pin_number": self.current_pin,
                    "x": orig_x,
                    "y": orig_y
                }
                self.pin_locations.append(pin_data)
                print(f"Added pin {self.current_pin} at ({orig_x}, {orig_y})")  # Debug print
                
                # Draw a circle at the selected point
                self.draw_pins()
                self.current_pin += 1
        
        elif event == cv2.EVENT_RBUTTONDOWN:  # Right click to start panning
            self.drag_start = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON:  # Right button drag for panning
            if self.drag_start is not None:
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                self.offset = (self.offset[0] + dx, self.offset[1] + dy)
                self.drag_start = (x, y)
                self.update_display()

        elif event == cv2.EVENT_RBUTTONUP:  # End panning
            self.drag_start = None

    def draw_pins(self):
        """Draw all pins on the display image."""
        # Start with a fresh copy of the image
        self.display_image = self.image.copy()
        
        # Draw all pins
        for pin in self.pin_locations:
            x = pin["x"]
            y = pin["y"]
            cv2.circle(self.display_image, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(self.display_image, str(pin["pin_number"]), 
                       (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)
        
        self.update_display()

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
        """Save pin locations to a JSON file."""
        try:
            # Debug print to check if we have any pins
            print(f"Saving {len(self.pin_locations)} pins to {output_file}")
            
            with open(output_file, 'w') as f:
                json.dump({"pins": self.pin_locations}, f, indent=4)
            print(f"Pin locations saved to {output_file}")
            
            # Debug print to show what was saved
            print("Saved pins:", self.pin_locations)
        except Exception as e:
            print(f"Error saving pin locations: {e}")

    def run(self, image_path: str, output_file: str):
        """Run the pin mapping process."""
        if not self.load_image(image_path):
            return

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Controls:")
        print("- Mouse wheel: Zoom in/out")
        print("- Right click and drag: Pan image")
        print("- Left click: Add pin")
        print("- Press 's' to save pins")
        print("- Press 'ESC' to exit")

        # Initial display of the image
        self.update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                # Save pins before exiting
                if self.pin_locations:  # Only save if we have pins
                    self.save_pin_locations(output_file)
                break
            elif key == ord('s'):  # Save
                self.save_pin_locations(output_file)

        cv2.destroyAllWindows()

if __name__ == '__main__':
    mapper = PinMapper()
    
    # Create directory if it doesn't exist
    os.makedirs("dev-boards", exist_ok=True)
    
    component_path = "dev-boards/Arduino-UNO.png"
    json_file = "dev-boards/Arduino-UNO.json"

    mapper.run(component_path, json_file)