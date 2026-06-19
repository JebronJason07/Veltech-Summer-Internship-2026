# exercise_5_complete_app.py
import cv2
import numpy as np
import os
from datetime import datetime

class CompleteImageAnalysisApp:
    """Complete application with GUI-like interface"""
 
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from: {image_path}")
 
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) 
        self.original = self.image.copy()
        self.steps = []
        self.current_step = 0
 
    def add_step(self, name, image):
        """Add a step to the pipeline"""
        self.steps.append((name, image))
        return self
 
    def process(self):
        """Run the complete processing pipeline"""
        print("Processing Image...")
 
        # Step 1: Original
        self.add_step("Original", self.image.copy())
 
        # Step 2: Grayscale
        self.add_step("Grayscale", cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR))
 
        # Step 3: Blur
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        self.add_step("Blurred", cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR))
 
        # Step 4: Histogram Equalization
        equalized = cv2.equalizeHist(blurred)
        self.add_step("Equalized", cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR))
 
        # Step 5: Threshold
        _, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.add_step("Binary", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
 
        # Step 6: Edges
        edges = cv2.Canny(equalized, 50, 150)
        self.add_step("Edges", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
 
        # Step 7: Contours
        # Cross-version safe contour extraction
        contours_info = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        
        contour_vis = self.image.copy()
        cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
        self.add_step("Contours", contour_vis)
 
        # Step 8: Analysis
        analysis_vis = self.image.copy()
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 50:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(analysis_vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(analysis_vis, f"#{i+1}", (x+5, y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        self.add_step("Analysis", analysis_vis)
 
        return self
 
    def show_grid(self): 
        """Display all steps in a grid layout"""
        # Determine grid size
        n = len(self.steps)
        cols = 4
        rows = (n + cols - 1) // cols
 
        # Define accurate box sizes and margin spaces
        img_w, img_h = 200, 150
        pad = 10
        
        # Create grid background canvas dynamically based on rows/columns
        grid_h = rows * img_h + (rows + 1) * pad
        grid_w = cols * img_w + (cols + 1) * pad
        grid = np.zeros((grid_h, grid_w, 3), dtype="uint8")
 
        for i, (name, img) in enumerate(self.steps):
            row = i // cols
            col = i % cols
            
            # Absolute coordinates mapping
            x = pad + col * (img_w + pad)
            y = pad + row * (img_h + pad)
 
            # Resize and blend step image matrix into target slice
            img_resized = cv2.resize(img, (img_w, img_h))
            grid[y:y+img_h, x:x+img_w] = img_resized
 
            # Place visual title tags
            cv2.putText(grid, name, (x+5, y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
 
        return grid
 
    def generate_summary(self):
        """Generate summary report metrics string"""
        report = f"""=========================================================================
IMAGE ANALYSIS SUMMARY
=========================================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Image Dimensions: {self.image.shape[1]} x {self.image.shape[0]}
=========================================================================
PROCESSING STEPS
=========================================================================
"""
        for i, (name, _) in enumerate(self.steps):
            report += f" Step {i+1}: {name}\n"
 
        report += f"""=========================================================================
COMPLETED
=========================================================================
Total steps: {len(self.steps)}
Image processed successfully!
"""
        return report

if __name__ == "__main__":
    # Create an empty test image if "test_image.png" doesn't exist yet
    if not os.path.exists("test_image.png"):
        dummy_img = np.zeros((400, 400, 3), dtype="uint8")
        # Draw some arbitrary shapes to trace contours against
        cv2.rectangle(dummy_img, (60, 60), (180, 180), (0, 0, 255), -1)
        cv2.circle(dummy_img, (280, 280), 60, (255, 0, 0), -1)
        cv2.imwrite("test_image.png", dummy_img)

    # Initialize and execute application pipeline
    app = CompleteImageAnalysisApp("test_image.png")
    app.process()

    # Show grid
    grid_matrix = app.show_grid()
    cv2.imshow("Complete Analysis Pipeline", grid_matrix)
    cv2.imwrite("complete_pipeline.png", grid_matrix)

    # Print summary execution logs
    print(app.generate_summary())
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()