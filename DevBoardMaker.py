import cv2
import numpy as np
import os

class DevBoardMaker:
    def __init__(self):
        self.image = None
        self.mask = None
        self.window_name = "Background Removal Tool"
        self.drawing = False
        self.mode = "remove"  # Start directly in remove mode since image is already cropped
        self.rect_start = None
        self.rect_end = None
        self.crop_points = []
        self.history = []  # List of (image, mask, checker_pattern) tuples
        # Add checkerboard pattern
        self.checker_size = 10  # Size of checker pattern squares
        self.checker_pattern = None
        
    def create_checker_pattern(self, height, width):
        """Create a checkerboard pattern for background visualization."""
        pattern = np.zeros((height, width), dtype=np.uint8)
        for i in range(0, height, self.checker_size):
            for j in range(0, width, self.checker_size):
                if (i + j) // self.checker_size % 2 == 0:
                    pattern[i:min(i+self.checker_size, height), 
                           j:min(j+self.checker_size, width)] = 255
        return pattern

    def load_image(self, image_path: str) -> bool:
        """Load the image from the specified path."""
        try:
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise ValueError("Image could not be loaded")
            self.mask = np.ones(self.image.shape[:2], dtype=np.uint8) * 255
            # Create checker pattern
            self.checker_pattern = self.create_checker_pattern(self.image.shape[0], self.image.shape[1])
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False

    def save_state(self):
        """Save current state to history."""
        self.history.append((
            self.image.copy(),
            self.mask.copy(),
            self.checker_pattern.copy()  # Also save checker pattern
        ))
        
    def undo_last_action(self):
        """Restore the last state from history."""
        if self.history:
            self.image, self.mask, self.checker_pattern = self.history.pop()
            self.update_display()
            print("Undid last action")
        else:
            print("Nothing to undo")

    def remove_similar_colors(self, x, y, tolerance=30):
        """Remove areas with similar color to the selected point."""
        # Get the color at the selected point
        color = self.image[y, x].astype(np.int32)
        
        # Create a mask for similar colors
        color_mask = np.all(np.abs(self.image.astype(np.int32) - color) <= tolerance, axis=2)
        
        # Update the mask
        self.mask[color_mask] = 0
        self.update_display()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for cropping and background removal."""
        if self.mode == "crop":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.rect_start = (x, y)
                self.crop_points = []
            
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                img_copy = self.image.copy()
                cv2.rectangle(img_copy, self.rect_start, (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, img_copy)
            
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.rect_end = (x, y)
                self.crop_image()
                
        elif self.mode == "remove":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.save_state()
                if flags & cv2.EVENT_FLAG_SHIFTKEY:  # Shift + Click for color-based removal
                    self.remove_similar_colors(x, y)
            
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                if not (flags & cv2.EVENT_FLAG_SHIFTKEY):  # Normal removal
                    cv2.circle(self.mask, (x, y), 10, 0, -1)
                    self.update_display()
            
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False

    def crop_image(self):
        """Crop the image to the selected rectangle."""
        if self.rect_start and self.rect_end:
            # Save current state before cropping
            self.save_state()
            
            x1, y1 = self.rect_start
            x2, y2 = self.rect_end
            
            # Ensure correct order of coordinates
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Crop both image and mask
            self.image = self.image[y1:y2, x1:x2]
            self.mask = self.mask[y1:y2, x1:x2]
            
            # Update checker pattern for new dimensions
            self.checker_pattern = self.create_checker_pattern(self.image.shape[0], self.image.shape[1])
            
            # Reset points
            self.rect_start = None
            self.rect_end = None
            
            self.update_display()

    def update_display(self):
        """Update the display with the current image and mask."""
        # Create a result image
        result = self.image.copy()
        
        # Create a white background
        white_bg = np.ones_like(result) * 255
        
        # Create a checker pattern background (gray and white)
        checker_bg = white_bg.copy()
        checker_bg[self.checker_pattern == 0] = [192, 192, 192]  # Light gray
        
        # Blend the image with backgrounds based on mask
        result = np.where(self.mask[:, :, np.newaxis] == 255, result, checker_bg)
        
        cv2.imshow(self.window_name, result)

    def save_image(self, output_path: str):
        """Save the processed image with transparency."""
        try:
            # Create RGBA image
            rgba = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)
            # Set alpha channel based on mask (0 where mask is 0, 255 elsewhere)
            rgba[:, :, 3] = self.mask
            
            cv2.imwrite(output_path, rgba)
            print(f"Image saved to {output_path}")
        except Exception as e:
            print(f"Error saving image: {e}")

    def run(self, input_path: str, output_path: str):
        """Run the background removal tool."""
        if not self.load_image(input_path):
            return

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\nBackground Removal Controls:")
        print("- Left click and drag: Remove background")
        print("- Shift + Left click: Remove similar colors")
        print("- Press 'Ctrl + Z': Undo last action")
        print("- Press 's': Save and return to pin mapper")
        print("- Press 'ESC': Cancel and return")

        self.update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('s'):  # Save
                self.save_image(output_path)
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    DEV_BOARD = "symbols.jpg"

    maker = DevBoardMaker()
    
    # Create output directory if it doesn't exist
    os.makedirs("dev-boards", exist_ok=True)
    
    input_image = f"ref/{DEV_BOARD}"  # Replace with your input image path
    output_image = f"dev-boards/{DEV_BOARD}.png"
    
    maker.run(input_image, output_image)
