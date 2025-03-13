import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb
from skimage.metrics import adapted_rand_error  


def load_dataset(image_path, gt_path=None):
    """
    Loads a satellite image and an optional ground truth segmentation.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found at", image_path)
        return None, None
    # Convert BGR to RGB for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ground_truth = None
    if gt_path is not None:
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        ground_truth = gt
    return image_rgb, ground_truth

def linear_filter(image_gray):
    """Applies a Gaussian filter (linear) to the grayscale image."""
    return cv2.GaussianBlur(image_gray, (5, 5), sigmaX=1.5)

def nonlinear_filter(image_gray):
    """Applies a Median filter (non-linear) to the grayscale image."""
    return cv2.medianBlur(image_gray, 5)

def segmentation_kmeans(image_gray, K=2):
    """Segments image using K-Means clustering."""
    Z = image_gray.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented = res.reshape(image_gray.shape)
    # Binarize segmentation based on the mean of centers
    thresh_val = np.mean(center)
    _, binary = cv2.threshold(segmented, thresh_val, 255, cv2.THRESH_BINARY)
    return segmented, binary

def segmentation_meanshift(image_rgb):
    """Segments image using Mean Shift filtering."""
    # Mean shift filtering works on color images
    shifted = cv2.pyrMeanShiftFiltering(image_rgb, sp=20, sr=40)
    gray_shifted = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_shifted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return shifted, binary

def segmentation_graph_based(image_rgb):
    """Segments image using Graph-Based (Felzenszwalb) segmentation."""
    image_float = image_rgb.astype(np.float32) / 255.0
    segments = felzenszwalb(image_float, scale=100, sigma=0.5, min_size=50)
    return segments

def region_growing(image_gray, seed_point, threshold=15):
    """
    Applies a simple region-growing algorithm starting from a seed point.
    The floodFill function is used with a tolerance threshold.
    """
    h, w = image_gray.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    floodfilled = image_gray.copy()
    cv2.floodFill(floodfilled, mask, seed_point, 255, loDiff=threshold, upDiff=threshold)
    return floodfilled

def remove_small_components(binary_image, min_area=500):
    """
    Removes small connected components from a binary image.
    Components with area smaller than min_area are removed.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    output = np.zeros(binary_image.shape, dtype=np.uint8)
    for i in range(1, num_labels):  # Skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output[labels == i] = 255
    return output

def compute_iou(pred, gt):
    """Computes the Intersection over Union (IoU) metric."""
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compute_dice(pred, gt):
    """Computes the Dice Coefficient."""
    intersection = np.sum(pred * gt)
    dice = (2. * intersection) / (np.sum(pred) + np.sum(gt))
    return dice

def compute_pixel_accuracy(pred, gt):
    """Computes pixel accuracy."""
    return np.sum(pred == gt) / pred.size

def adaptive_segmentation(image_gray):
    """Adaptive segmentation using adaptive Gaussian thresholding."""
    adaptive = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    return adaptive

# =============================================================================
# Main Pipeline Execution & Visualization
# =============================================================================
def main():
    # ----------------------------
    # Section 1: Problem Selection & Dataset Preparation
    # ----------------------------
    image_path = "22678945_15.png"  # Update with your satellite image file
    gt_path = "22678945_152.png"          # Optionally, update with your ground truth segmentation image
    image_rgb, ground_truth = load_dataset(image_path, gt_path)
    if image_rgb is None:
        return
    
    # Convert the image to grayscale for processing
    image_gray = cv2.cvtColor(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    
    # Display original image and ground truth (if available)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Satellite Image")
    plt.axis("off")
    if ground_truth is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(ground_truth, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")
    plt.show()

    # ----------------------------
    # Section 2: Noise Reduction
    # ----------------------------
    gauss_filtered = linear_filter(image_gray)
    median_filtered = nonlinear_filter(image_gray)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image_gray, cmap="gray")
    plt.title("Original Grayscale")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(gauss_filtered, cmap="gray")
    plt.title("Gaussian Filtered")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(median_filtered, cmap="gray")
    plt.title("Median Filtered")
    plt.axis("off")
    plt.show()
    
    # ----------------------------
    # Section 3: Segmentation and Object Extraction
    # ----------------------------
    # K-Means Segmentation on Gaussian filtered image
    segmented_km, binary_km = segmentation_kmeans(gauss_filtered, K=2)
    
    # Mean Shift Segmentation on color image
    shifted, binary_ms = segmentation_meanshift(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    
    # Graph-Based Segmentation
    segments_gb = segmentation_graph_based(image_rgb)
    
    # Innovation: Adaptive segmentation using adaptive thresholding
    adaptive_seg = adaptive_segmentation(gauss_filtered)
    
    # Display segmentation results
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(binary_km, cmap="gray")
    plt.title("K-Means Segmentation")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.imshow(binary_ms, cmap="gray")
    plt.title("Mean Shift Segmentation")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(segments_gb, cmap="nipy_spectral")
    plt.title("Graph-Based Segmentation")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.imshow(adaptive_seg, cmap="gray")
    plt.title("Adaptive Segmentation")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    # For further processing, we choose Mean Shift segmentation (binary_ms) as an example.
    selected_segmentation = binary_ms
    
    # ----------------------------
    # Section 4: Region-Based Processing
    # ----------------------------
    # Example of region growing using a manually selected seed point
    seed_point = (50, 50)  # Adjust based on image content
    region_grown = region_growing(image_gray, seed_point, threshold=10)
    
    # Refine segmentation using connected component analysis
    refined_segmentation = remove_small_components(selected_segmentation, min_area=500)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(region_grown, cmap="gray")
    plt.title("Region Growing Result")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(refined_segmentation, cmap="gray")
    plt.title("Refined Segmentation (Connected Components)")
    plt.axis("off")
    plt.show()
    
    # ----------------------------
    # Section 5: Final Evaluation & Report
    # ----------------------------
    if ground_truth is not None:
        # Convert to binary masks for evaluation
        pred = (refined_segmentation > 0).astype(np.uint8)
        gt_binary = (ground_truth > 0).astype(np.uint8)
        
        iou = compute_iou(pred, gt_binary)
        dice = compute_dice(pred, gt_binary)
        accuracy = compute_pixel_accuracy(pred, gt_binary)
        
        print("Evaluation Metrics:")
        print("IoU: {:.4f}".format(iou))
        print("Dice Coefficient: {:.4f}".format(dice))
        print("Pixel Accuracy: {:.4f}".format(accuracy))
    else:
        print("No ground truth provided for quantitative evaluation.")
    
    
if __name__ == "__main__":
    main()
