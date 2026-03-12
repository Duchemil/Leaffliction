#!/usr/bin/env python3
"""
Leaf Image Transformation using PlantCV
Usage: python Transformation.py <image_path>
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Try to import plantcv, fall back to opencv if not available
try:
    from plantcv import plantcv as pcv
    PLANTCV_AVAILABLE = True
except ImportError:
    PLANTCV_AVAILABLE = False

import cv2


def process_leaf_plantcv(image_path):
    """Process leaf image using PlantCV library."""
    
    # Configure PlantCV
    pcv.params.debug = None
    pcv.params.line_thickness = 3

    # 1. Read the original image
    img, path, filename = pcv.readimage(filename=image_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Leaf Image Transformations (PlantCV)", fontsize=16, fontweight='bold')

    # Figure IV.1: Original
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Figure IV.1: Original")
    axes[0, 0].axis('on')

    # Figure IV.2: Gaussian Blur
    gaussian = pcv.gaussian_blur(img=img, ksize=(11, 11), sigma_x=0, sigma_y=None)
    axes[0, 1].imshow(gaussian, cmap='gray' if len(gaussian.shape) == 2 else None)
    if len(gaussian.shape) == 3:
        axes[0, 1].imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY), cmap='gray')
    axes[0, 1].set_title("Figure IV.2: Gaussian blur")
    axes[0, 1].axis('on')

    # Convert to grayscale for thresholding
    gray_img = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    # Figure IV.3: Mask
    binary_thresh = pcv.threshold.binary(gray_img=gray_img, threshold=50, object_type='light')
    mask = pcv.fill(bin_img=binary_thresh, size=200)
    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title("Figure IV.3: Mask")
    axes[1, 0].axis('on')

    # Figure IV.4: ROI objects (region of interest with green fill)
    masked_img = pcv.apply_mask(img=img, mask=mask, mask_color='white')
    
    # Create ROI overlay
    roi_display = img.copy()
    green_overlay = np.zeros_like(img)
    green_overlay[mask > 0] = [0, 255, 0]  # Green in BGR
    roi_display = cv2.addWeighted(roi_display, 0.5, green_overlay, 0.5, 0)
    
    # Add blue border
    h, w = roi_display.shape[:2]
    cv2.rectangle(roi_display, (5, 5), (w-5, h-5), (255, 0, 0), 4)
    
    axes[0, 2].imshow(cv2.cvtColor(roi_display, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Figure IV.4: Roi objects")
    axes[0, 2].axis('on')

    # Figure IV.5: Analyze object (shape analysis with outlines)
    # Find contours for shape analysis
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    analyze_img = img.copy()
    
    # Draw outer contour (magenta/pink)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(analyze_img, [largest], -1, (255, 0, 255), 3)  # Magenta
        
        # Fit ellipse if possible
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            cv2.ellipse(analyze_img, ellipse, (255, 128, 0), 2)  # Orange ellipse
        
        # Draw bounding box
        x, y, bw, bh = cv2.boundingRect(largest)
        cv2.rectangle(analyze_img, (x, y), (x+bw, y+bh), (255, 0, 0), 2)  # Blue
        
        # Smaller contours (disease spots) in blue
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < cv2.contourArea(largest) * 0.1:
                cv2.drawContours(analyze_img, [cnt], -1, (0, 0, 255), 2)
    
    axes[1, 1].imshow(cv2.cvtColor(analyze_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Figure IV.5: Analyze object")
    axes[1, 1].axis('on')

    # Figure IV.6: Pseudolandmarks
    landmark_img = img.copy()
    
    if contours and len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        
        # Sample evenly spaced points along the contour (blue dots - outer leaf)
        n_points = 20
        indices = np.linspace(0, len(largest)-1, n_points, dtype=int)
        for idx in indices:
            pt = largest[idx][0]
            cv2.circle(landmark_img, tuple(pt), 5, (255, 0, 0), -1)  # Blue
        
        # Disease/lesion spots (orange/red dots)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < cv2.contourArea(largest) * 0.05:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(landmark_img, (cx, cy), 6, (0, 128, 255), -1)  # Orange
        
        # Midrib line points (green dots along center)
        M_leaf = cv2.moments(largest)
        if M_leaf['m00'] != 0:
            cx_leaf = int(M_leaf['m10'] / M_leaf['m00'])
            cy_leaf = int(M_leaf['m01'] / M_leaf['m00'])
            x, y, bw, bh = cv2.boundingRect(largest)
            for i in range(5):
                py = y + int(bh * i / 4)
                cv2.circle(landmark_img, (cx_leaf, py), 5, (0, 255, 0), -1)  # Green
    
    axes[1, 2].imshow(cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Figure IV.6: Pseudolandmarks")
    axes[1, 2].axis('on')

    plt.tight_layout()
    
    output_path = "leaf_transformations_output.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Output saved to: {output_path}")
    plt.close()
    
    return output_path


def process_leaf_opencv(image_path):
    """Process leaf image using OpenCV (fallback when PlantCV not available)."""
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Leaf Image Transformations", fontsize=16, fontweight='bold')

    # Figure IV.1: Original
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Figure IV.1: Original")
    axes[0, 0].axis('on')

    # Figure IV.2: Gaussian Blur → grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (11, 11), 0)
    axes[0, 1].imshow(gaussian, cmap='gray')
    axes[0, 1].set_title("Figure IV.2: Gaussian blur")
    axes[0, 1].axis('on')

    # Figure IV.3: Mask via HSV threshold
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    blurred_s = cv2.GaussianBlur(s_channel, (5, 5), 0)
    _, binary = cv2.threshold(blurred_s, 50, 255, cv2.THRESH_BINARY)
    
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title("Figure IV.3: Mask")
    axes[1, 0].axis('on')

    # Figure IV.4: ROI objects (green overlay + blue border)
    roi_display = img.copy()
    green_overlay = np.zeros_like(img)
    green_overlay[mask > 0] = [0, 255, 0]
    roi_display = cv2.addWeighted(roi_display, 0.5, green_overlay, 0.5, 0)
    h, w = roi_display.shape[:2]
    cv2.rectangle(roi_display, (5, 5), (w-5, h-5), (255, 0, 0), 4)
    
    axes[0, 2].imshow(cv2.cvtColor(roi_display, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Figure IV.4: Roi objects")
    axes[0, 2].axis('on')

    # Figure IV.5: Analyze object (shape analysis)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    analyze_img = img.copy()
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(analyze_img, [largest], -1, (255, 0, 255), 3)  # Magenta outer
        
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            cv2.ellipse(analyze_img, ellipse, (255, 128, 0), 2)
        
        x, y, bw, bh = cv2.boundingRect(largest)
        cv2.rectangle(analyze_img, (x, y), (x+bw, y+bh), (255, 0, 0), 2)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < cv2.contourArea(largest) * 0.1:
                cv2.drawContours(analyze_img, [cnt], -1, (0, 0, 255), 2)
    
    axes[1, 1].imshow(cv2.cvtColor(analyze_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Figure IV.5: Analyze object")
    axes[1, 1].axis('on')

    # Figure IV.6: Pseudolandmarks
    landmark_img = img.copy()
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest)
        
        # Blue dots: evenly spaced along outer contour
        n_points = 20
        indices = np.linspace(0, len(largest)-1, n_points, dtype=int)
        for idx in indices:
            pt = largest[idx][0]
            cv2.circle(landmark_img, tuple(pt), 5, (255, 0, 0), -1)
        
        # Orange dots: disease/lesion spots
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < largest_area * 0.05:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(landmark_img, (cx, cy), 6, (0, 128, 255), -1)
        
        # Green dots: midrib landmarks
        M_leaf = cv2.moments(largest)
        if M_leaf['m00'] != 0:
            cx_leaf = int(M_leaf['m10'] / M_leaf['m00'])
            x, y, bw, bh = cv2.boundingRect(largest)
            for i in range(5):
                py = y + int(bh * i / 4)
                cv2.circle(landmark_img, (cx_leaf, py), 5, (0, 255, 0), -1)
    
    axes[1, 2].imshow(cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Figure IV.6: Pseudolandmarks")
    axes[1, 2].axis('on')

    plt.tight_layout()
    
    output_path = "leaf_transformations_output.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Output saved to: {output_path}")
    plt.close()
    
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python Transformation.py <image_path>")
        print("Example: python Transformation.py ./Apple/apple_healthy/image\\ \\(1\\).JPG")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    
    print(f"Processing image: {image_path}")
    
    if PLANTCV_AVAILABLE:
        print("Using PlantCV library...")
        output = process_leaf_plantcv(image_path)
    else:
        print("PlantCV not found, using OpenCV fallback...")
        output = process_leaf_opencv(image_path)
    
    print(f"Done! Result saved to: {output}")


if __name__ == "__main__":
    main()