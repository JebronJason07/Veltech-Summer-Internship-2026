# exercise_1_custom_analysis.py
import cv2
import numpy as np
from day14_mini_project import ObjectAnalysisPipeline

class CustomObjectAnalysis(ObjectAnalysisPipeline):
    """Extended pipeline with custom analysis""" 
 
    def analyze_contours(self):
        """Override to add custom analysis"""
        print("\n[Custom] Analyzing contours with additional metrics...")
        
        # Run base analysis first to populate self.results['contours']
        super().analyze_contours()
 
        # Add custom metrics
        for i, contour in enumerate(self.contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Circularity
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0
 
            # Extent (bbox area ratio)
            x, y, w, h = cv2.boundingRect(contour)
            bbox_area = w * h
            extent = area / bbox_area if bbox_area > 0 else 0
 
            # Add to results dictionary created by base class
            self.results['contours'][i]['circularity'] = circularity
            self.results['contours'][i]['extent'] = extent
 
            print(
                f" Contour {i+1}: circularity={circularity:.3f}, "
                f"extent={extent:.3f}"
            )
 
        return self

if __name__ == "__main__":
    # Run custom analysis
    pipeline = CustomObjectAnalysis("test_image.png", "custom_output")
    pipeline.run()