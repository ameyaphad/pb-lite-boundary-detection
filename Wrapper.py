#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate
import skimage.transform as tf
from sklearn.cluster import KMeans
from scipy.signal import convolve2d


def gaussian_kernel(size, mean_x, mean_y, var_x, var_y, order = 0):

	t = np.linspace(-(size - 1)/2, (size - 1)/2, size)

	x = np.exp(-(t - mean_x) ** 2 / (2 * var_x))
	y = np.exp(-(t - mean_y) ** 2 / (2 * var_y))

	if order == 1:
		x = -x * (t / var_x)
		# y = -y * (t / var_y)

	elif order == 2:
		x = x * ((t - mean_x) ** 2 - var_x)/(var_x ** 2)
		# y = y * ((t - mean_y) ** 2 - var_y) / (var_y ** 2)


	# make a 2-D kernel out of it
	kernel = x[:, np.newaxis] * y[np.newaxis, :]

	return kernel/np.max(kernel)


def Gaussian_bank(scales, orientations, size, plot=False, var_scale=0.25):
	variances = np.array([1, 2, 4, 8])

	var_vers = var_scale * np.linspace(size / 4, 3 * size / 4, scales)
	degrees_vec = np.linspace(0, 360 * (1 - 1 / orientations), orientations)
	gaussian_bank = np.array([[gaussian_kernel(size, 0, 0, variances[i], variances[i])] for i in range(4)])

	if plot:
		save_plot(DoG_filter_bank, 'DoG')

	return gaussian_bank



def LOG(size, var_x, var_y):
	filter = gaussian_kernel(size, 0, 0, var_x, var_y)
	return cv2.filter2D(filter, -1, Laplacian())


def sobel_filter():
	return np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])

def Laplacian():
	return np.array([[0, 1, 0],[1, -4, 1], [0, 1, 0]])

def rotateImage(image, angle, clock_wise = True):
	rows, cols = image.shape[0], image.shape[1]
	matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 2 * (int(clock_wise) - 0.5))
	# fixing the rotation matrix to be n=in the (0 ,0)
	matrix[0, 2] += (matrix[0, 0] + matrix[0, 1] - 1) / 2
	matrix[1, 2] += (matrix[1, 0] + matrix[1, 1] - 1) / 2
	result = cv2.warpAffine(image, matrix, (cols, rows))
	return result

def save_plot(filter_bank, name):

	rows , cols = filter_bank.shape[0:2]

	plt.figure(figsize=(16, 2))
	sub = 1
	for row in range(rows):
		for col in range(cols):
			plt.subplot(rows, cols, sub)
			plt.imshow(filter_bank[row][col], cmap='gray')
			plt.axis('off')
			sub += 1

	plt.savefig(name)

def gen_DoG_filter_bank(scales, orientations, size, plot=False, var_scale=0.25):

	var_vers = var_scale * np.linspace(size / 4, 3 * size / 4, scales)
	degrees_vec = np.linspace(0, 360 * (1 - 1/orientations), orientations)
	DoG_filter_bank = np.array([[DoG(size, var, degree) for degree in degrees_vec] for var in var_vers])

	if plot:
		save_plot(DoG_filter_bank, 'DoG')

	return DoG_filter_bank

def DoG(size, var, rotation_degree):
	kernel = gaussian_kernel(size, 0, 0, var, var)
	sobel = sobel_filter()
	filter = cv2.filter2D(kernel, -1, sobel)

	return rotateImage(filter, rotation_degree, clock_wise=False)


def gaussian_anisotropic_kernel(size, sigma_x, sigma_y, theta):
    """
    Generate an anisotropic Gaussian kernel with orientation.
    """
    assert size % 2 == 1, "Kernel size must be odd!"
    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)

    # Rotate coordinates by theta
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Anisotropic Gaussian
    gaussian = np.exp(-(x_theta**2 / (2 * sigma_x**2) + y_theta**2 / (2 * sigma_y**2)))
    gaussian /= np.sum(gaussian)
    return gaussian


def laplacian_of_gaussian_kernel(size, sigma):
    """
    Generate a Laplacian of Gaussian (LoG) kernel.
    """
    assert size % 2 == 1, "Kernel size must be odd!"
    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian /= np.sum(gaussian)
    log = (x**2 + y**2 - 2 * sigma**2) / (sigma**4) * gaussian
    return log


def convolve(kernel1, kernel2):
    """
    Perform 2D convolution of two kernels manually.
    """
    size1 = kernel1.shape[0]
    size2 = kernel2.shape[0]
    result_size = size1 + size2 - 1
    result = np.zeros((result_size, result_size))

    # Perform convolution
    for i in range(size1):
        for j in range(size1):
            result[i:i+size2, j:j+size2] += kernel1[i, j] * kernel2

    return result


def rotate_filter(filter, angle):
    """
    Rotate a 2D filter to a specific angle using a rotation matrix.
    """
    size = filter.shape[0]
    center = size // 2
    rotated_filter = np.zeros_like(filter)

    # Rotation matrix
    angle_rad = np.deg2rad(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    for x in range(size):
        for y in range(size):
            # Translate to origin
            x0 = x - center
            y0 = y - center

            # Rotate
            x_rot = cos_theta * x0 - sin_theta * y0
            y_rot = sin_theta * x0 + cos_theta * y0

            # Translate back
            x_new = int(round(x_rot + center))
            y_new = int(round(y_rot + center))

            if 0 <= x_new < size and 0 <= y_new < size:
                rotated_filter[x, y] = filter[x_new, y_new]

    return rotated_filter


def create_lm_filter_bank(small=True):
    """
    Create the LM Filter Bank (LMS or LML).
    """
    # Define scales
    if small:
        scales = [1, np.sqrt(2), 2, 2 * np.sqrt(2)]
    else:
        scales = [np.sqrt(2), 2, 2 * np.sqrt(2), 4]

    orientations = 6
    elongation_factor = 3
    filter_bank = []

    # 1. First and Second Derivatives of Gaussian
    for sigma in scales[:3]:  # Only the first three scales
        for orientation in range(orientations):
            theta = orientation * (np.pi / orientations)
            g_kernel = gaussian_anisotropic_kernel(49, sigma, elongation_factor * sigma, theta)   #15

            # First derivative
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            dog_x = convolve(g_kernel, sobel_x)
            filter_bank.append(dog_x)

            # Second derivative
            sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            dog_y = convolve(g_kernel, sobel_y)
            filter_bank.append(dog_y)

    # 2. Laplacian of Gaussian (LoG)
    for sigma in scales:  # Only the first two scales
        log = laplacian_of_gaussian_kernel(49, sigma)     #15
        filter_bank.append(log)
        log_3sigma = laplacian_of_gaussian_kernel(49, 3 * sigma)  #15
        filter_bank.append(log_3sigma)

    # 3. Gaussians
    for sigma in scales:
        gaussian = gaussian_anisotropic_kernel(49, sigma, sigma, 0) #15
        filter_bank.append(gaussian)

    return filter_bank


def gabor_filter(size, wavelength, orientation, sigma, phase=0):
    """
    Generate a Gabor filter.

    Parameters:
        size (int): Size of the filter (output will be size x size).
        wavelength (float): Wavelength of the sinusoidal component.
        orientation (float): Orientation of the filter in degrees.
        sigma (float): Standard deviation of the Gaussian envelope.
        phase (float): Phase offset of the sinusoid (default is 0).

    Returns:
        np.ndarray: The Gabor filter.
    """
    # Create coordinate grids
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)

    # Rotate coordinates
    theta = np.deg2rad(orientation)
    X_theta = X * np.cos(theta) + Y * np.sin(theta)
    Y_theta = -X * np.sin(theta) + Y * np.cos(theta)

    # Create the Gabor filter
    gaussian_envelope = np.exp(-(X_theta**2 + Y_theta**2) / (2 * sigma**2))
    sinusoidal_component = np.cos(2 * np.pi * X_theta / wavelength + phase)
    gabor = gaussian_envelope * sinusoidal_component

    return gabor

def visualize_gabor_filters(num_scales, num_orientations, filter_size=31,visualize=False):
    """
    Generate and visualize Gabor filters with multiple scales and orientations.

    Parameters:
        num_scales (int): Number of scales.
        num_orientations (int): Number of orientations.
        filter_size (int): Size of each Gabor filter.
    """
    # Define parameters
    wavelengths = np.linspace(10, 20, num_scales)  # Wavelengths for scales
    orientations = np.linspace(0, 180, num_orientations, endpoint=False)  # Orientations
    sigma = filter_size / 6  # Standard deviation for Gaussian envelope

    # filter bank
    gabor_filter_bank = []
    for wavelength in wavelengths:
        for orientation in orientations:
            # Generate Gabor filter
            gabor = gabor_filter(filter_size, wavelength, orientation, sigma)
            gabor_filter_bank.append(gabor)

    # Create a figure
    if visualize:
      fig, axes = plt.subplots(num_scales, num_orientations, figsize=(12, 8))
      for i, wavelength in enumerate(wavelengths):
          for j, orientation in enumerate(orientations):
              # Generate Gabor filter
              gabor = gabor_filter(filter_size, wavelength, orientation, sigma)

              # Plot the filter
              ax = axes[i, j]
              ax.imshow(gabor, cmap='gray')
              ax.axis('off')

      plt.tight_layout()
      plt.savefig('Gabor.png')
      plt.show()

    return gabor_filter_bank


def gen_half_disc_masks(scales):
	half_discs = []
	angles = [0, 180, 30, 210, 45, 225, 60, 240, 90, 270, 120, 300, 135, 315, 150, 330]           #rotation angles (not equally spaced)
	no_of_disc = len(angles)
	for radius in scales:
		kernel_size = 2*radius + 1
		cc = radius
		kernel = np.zeros([kernel_size, kernel_size])
		for i in range(radius):
			for j in range(kernel_size):
				a = (i-cc)**2 + (j-cc)**2                                     #to create one disc
				if a <= radius**2:
					kernel[i,j] = 1

		for i in range(0, no_of_disc):                                       #rotate to make other discs
			mask = tf.rotate(kernel, angles[i])
			mask[mask<=0.5] = 0
			mask[mask>0.5] = 1
			half_discs.append(mask)
	return half_discs


def apply_filter_bank(image, filter_bank):
    """
    Apply each filter in the filter bank to the input image.

    Args:
        image: Grayscale input image (2D numpy array).
        filter_bank: List of filters (2D numpy arrays).

    Returns:
        A numpy array of shape (H, W, N), where N is the total number of filters.
    """
    H, W = image.shape
    N = len(filter_bank)
    filter_responses = np.zeros((H, W, N))

    for i, filter_ in enumerate(filter_bank):
        # Apply the filter using convolution
        response = cv2.filter2D(image, -1, filter_)
        filter_responses[:, :, i] = response

    return filter_responses

def create_texton_map(image, filter_banks, num_clusters=64):
    """
    Create a texton map from an input image using multiple filter banks.

    Args:
        image: Grayscale input image (2D numpy array).
        filter_banks: List of filter banks, where each bank is a list of filters.
        num_clusters: Number of clusters (K) for KMeans.

    Returns:
        Texton map (2D numpy array with cluster IDs for each pixel).
    """
    # Apply all filter banks and concatenate responses
    all_filter_responses = []
    for filter_bank in filter_banks:
        responses = apply_filter_bank(image, filter_bank)
        all_filter_responses.append(responses)

    # Concatenate responses along the last axis (depth)
    combined_filter_responses = np.concatenate(all_filter_responses, axis=-1)  # Shape: (H, W, Total Filters)

    # Flatten the filter responses to create pixel-wise feature vectors
    H, W, N = combined_filter_responses.shape
    feature_vectors = combined_filter_responses.reshape(-1, N)  # Shape: (H*W, N)

    # Normalize feature vectors (optional but recommended for clustering)
    feature_vectors -= feature_vectors.mean(axis=0)
    feature_vectors /= feature_vectors.std(axis=0) + 1e-5

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(feature_vectors)

    # Assign each pixel to a cluster (texton ID)
    texton_ids = kmeans.labels_.reshape(H, W)

    return texton_ids


# Function to compute the chi-square distance for a single pair of masks
def compute_chi_square(img, bins, left_mask, right_mask):
    chi_sqr_dist = np.zeros_like(img, dtype=np.float32)

    # Loop over bins
    for i in range(bins):
        # Create a binary mask for pixels in the current bin
        tmp = (img == i).astype(np.float32)

        # Convolve with the left and right half-disk masks
        g_i = convolve2d(tmp, left_mask, mode='same', boundary='symm')
        h_i = convolve2d(tmp, right_mask, mode='same', boundary='symm')

        # Update chi-squared distance
        numerator = (g_i - h_i) ** 2
        denominator = g_i + h_i + 1e-10  # Add a small value to avoid division by zero
        chi_sqr_dist += numerator / denominator

    return 0.5 * chi_sqr_dist

# Function to compute Tg for all orientations and scales
def compute_texture_gradient(img, bins, half_disk_masks):
    m, n = img.shape  # Image dimensions
    num_filters = len(half_disk_masks) // 2  # Number of left/right mask pairs

    Tg = np.zeros((m, n, num_filters), dtype=np.float32)

    # Loop over each pair of left and right masks
    for idx in range(num_filters):
        left_mask = half_disk_masks[2 * idx]
        right_mask = half_disk_masks[2 * idx + 1]

        Tg[:, :, idx] = compute_chi_square(img, bins, left_mask, right_mask)

    return Tg


def generate_brightness_map(image, num_clusters=16):
    """
    Generate a brightness map by clustering grayscale brightness values using k-means.

    Args:
    - image (numpy.ndarray): Input RGB image (HxWxC).
    - num_clusters (int): Number of clusters for k-means (default is 16).

    Returns:
    - brightness_map (numpy.ndarray): Brightness map with clustered values (HxW).
    """
    # Convert the image to grayscale
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reshape the image into a 2D array of pixels (HxW becomes (H*W)x1)
    pixels = image.reshape(-1, 1)

    # Apply k-means clustering to the pixel values
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    kmeans.fit(pixels)

    # Get cluster labels for each pixel
    labels = kmeans.labels_

    # Reshape the labels back to the original image shape
    brightness_map = labels.reshape(image.shape)

    return brightness_map


def compute_brightness_gradient(img, bins, half_disk_masks):
    m, n = img.shape  # Image dimensions
    num_filters = len(half_disk_masks) // 2  # Number of left/right mask pairs

    Bg = np.zeros((m, n, num_filters), dtype=np.float32)

    # Loop over each pair of left and right masks
    for idx in range(num_filters):
        left_mask = half_disk_masks[2 * idx]
        right_mask = half_disk_masks[2 * idx + 1]

        Bg[:, :, idx] = compute_chi_square(img, bins, left_mask, right_mask)

    return Bg


def generate_color_map(image, num_clusters=16):
    """
    Generate a color map by clustering RGB values using KMeans clustering.

    Args:
        image (numpy.ndarray): Input image in RGB format (height x width x 3).
        num_clusters (int): Number of clusters for KMeans.

    Returns:
        color_map (numpy.ndarray): Single-channel color map with cluster IDs.
    """
    # Reshape the image into a 2D array of RGB pixels
    h, w, c = image.shape
    pixels = image.reshape((-1, c))  # Shape: (height*width, 3)

    # Perform KMeans clustering on the RGB pixel values
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(pixels)

    # Get cluster IDs for each pixel and reshape back to image dimensions
    cluster_ids = kmeans.labels_.reshape(h, w)

    return cluster_ids


def compute_color_gradient(img, bins, half_disk_masks):
    m, n = img.shape  # Image dimensions
    num_filters = len(half_disk_masks) // 2  # Number of left/right mask pairs

    Cg = np.zeros((m, n, num_filters), dtype=np.float32)

    # Loop over each pair of left and right masks
    for idx in range(num_filters):
        left_mask = half_disk_masks[2 * idx]
        right_mask = half_disk_masks[2 * idx + 1]

        Cg[:, :, idx] = compute_chi_square(img, bins, left_mask, right_mask)

    return Cg


def compute_pb_lite(Tg, Bg, Cg, cannyPb, sobelPb, w1=0.5, w2=0.5):
    """
    Compute the PbLite output based on the provided gradients and baseline edge maps.

    Args:
        Tg (numpy.ndarray): Texture gradient (2D array).
        Bg (numpy.ndarray): Brightness gradient (2D array).
        Cg (numpy.ndarray): Color gradient (2D array).
        cannyPb (numpy.ndarray): Canny edge detection result (2D array).
        sobelPb (numpy.ndarray): Sobel edge detection result (2D array).
        w1 (float): Weight for Canny edge detection.
        w2 (float): Weight for Sobel edge detection.

    Returns:
        PbEdges (numpy.ndarray): Final PbLite edge map (2D array).
    """
    # Ensure w1 and w2 sum to 1
    assert np.isclose(w1 + w2, 1.0), "Weights w1 and w2 must sum to 1."

    # Compute the average gradient (Tg + Bg + Cg) / 3
    avg_gradient = (Tg + Bg + Cg) / 3

    # Compute the baseline term: w1 * cannyPb + w2 * sobelPb
    baseline = w1 * cannyPb + w2 * sobelPb

    # Compute the PbLite result using Hadamard product
    PbEdges = avg_gradient * baseline

    return PbEdges

def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	DoG_filter_bank = gen_DoG_filter_bank(scales=2, orientations=16, size=9)

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	LMS_filter_bank = create_lm_filter_bank(small=True)  
	LML_filter_bank = create_lm_filter_bank(small=False)

	# Visualize the LMS filter bank
	plt.figure(figsize=(16, 12))
	for i, filt in enumerate(LMS_filter_bank):
		plt.subplot(4, 12, i + 1)
		plt.imshow(filt, cmap='gray')
		plt.axis('off')
	plt.suptitle("LMS Filter Bank", fontsize=16)
	plt.tight_layout()
	plt.savefig('LMS.png')
	plt.show()

	# Visualize the LML filter bank
	plt.figure(figsize=(16, 12))
	for i, filt in enumerate(LML_filter_bank):
		plt.subplot(4, 12, i + 1)
		plt.imshow(filt, cmap='gray')
		plt.axis('off')
	plt.suptitle("LML Filter Bank", fontsize=16)
	plt.tight_layout()
	plt.savefig('LML.png')
	plt.show()

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	Gabor_filter_bank = visualize_gabor_filters(num_scales=5, num_orientations=8, filter_size=49,visualize=True)

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	# Define scales and orientations
	scales = [3,4,5, 6, 7]  # Example scales (radii)
	orientations = [0,180, 45,225, 90,270, 135,315]  # 8 orientations

	# Generate masks
	half_disk_masks = gen_half_disc_masks(scales)

	# Visualize some masks
	plt.figure(figsize=(12, 8))
	for i in range(len(half_disk_masks)):
		plt.subplot(5,16,i+1)
		plt.imshow(half_disk_masks[i], cmap='gray')
		# plt.title(f"Scale={scale}, Orient={orientation}°")
		plt.axis('off')

	plt.tight_layout()
	plt.savefig('HDMasks.png')
	plt.show()

	for i in range(1,11):

		print(f"Processing image {i} ..")

		"""
		Generate Texton Map
		Filter image using oriented gaussian filter bank
		"""
		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""

		image = cv2.imread(f"../BSDS500/Images/{i}.jpg", cv2.IMREAD_GRAYSCALE)
		
		# Combine all filter banks into a list
		all_filter_banks = [DoG_filter_bank]

		# Create the texton map
		num_clusters = 64
		texton_map = create_texton_map(image, all_filter_banks, num_clusters)
		texton_img_path = "TextonMap_"+str(i)+".png"

		# Visualize the texton map
		plt.figure(figsize=(8, 6))
		plt.imshow(texton_map, cmap="jet")
		plt.colorbar(label="Texton ID")
		plt.title("Texton Map")
		plt.axis("off")
		plt.savefig(texton_img_path)
		plt.show()
		


		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		# Compute Tg
		bins = 8  # Number of histogram bins
		Tg = compute_texture_gradient(texton_map, bins, half_disk_masks)

		# Compute Mean across channels
		Tg_mean = np.mean(Tg,axis=2)

		Tg_img_name = "Tg_"+str(i)+".png"

		plt.figure(figsize=(8, 6))
		plt.imshow(Tg_mean, cmap='viridis')
		plt.title("Texton Gradient")
		plt.colorbar(label="Cluster ID")
		plt.axis("off")
		plt.savefig(Tg_img_name)
		plt.show()

		"""
		Generate Brightness Map
		Perform brightness binning 
		"""
		brightness_map = generate_brightness_map(image, num_clusters=16)

		brightness_map_img_name = "BrightnessMap_"+str(i)+".png"

		# Visualize the brightness map
		plt.figure(figsize=(10, 10))
		plt.title("Brightness Map")
		plt.imshow(brightness_map, cmap='viridis')
		plt.colorbar(label='Cluster ID')
		plt.savefig(brightness_map_img_name)
		plt.show()

		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		Bg = compute_brightness_gradient(brightness_map,8, half_disk_masks)

		# Compute Mean across channels
		Bg_mean = np.mean(Bg,axis=2)

		Bg_img_name = "Bg_"+str(i)+".png"

		plt.figure(figsize=(8, 6))
		plt.imshow(Bg_mean, cmap='viridis')
		plt.title("Brightness Gradient")
		plt.colorbar(label="Cluster ID")
		plt.axis("off")
		plt.savefig(Bg_img_name)
		plt.show()


		"""
		Generate Color Map
		Perform color binning or clustering
		"""

		# Generate the color map
		image2 = cv2.imread(f"../BSDS500/Images/{i}.jpg")  
		color_map = generate_color_map(image2, num_clusters=16)

		color_map_img_name = "ColorMap_"+str(i)+".png"

		# Visualize the color map
		plt.figure(figsize=(8, 6))
		plt.imshow(color_map, cmap='viridis')
		plt.title("Color Map")
		plt.colorbar(label="Cluster ID")
		plt.axis("off")
		plt.savefig(color_map_img_name)
		plt.show()

		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		
		Cg = compute_color_gradient(brightness_map,8, half_disk_masks)

		# Compute Mean across channels
		Cg_mean = np.mean(Cg,axis=2)

		Cg_img_name = "Cg_"+str(i)+".png"

		plt.figure(figsize=(8, 6))
		plt.imshow(Cg_mean, cmap='viridis')
		plt.title("Color Gradient")
		plt.colorbar(label="Cluster ID")
		plt.axis("off")
		plt.savefig(Cg_img_name)
		plt.show()

		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
		sobel_baseline = cv2.imread("../BSDS500/SobelBaseline/"+str(i)+".png", cv2.IMREAD_GRAYSCALE)

		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""
		canny_baseline = cv2.imread("../BSDS500/CannyBaseline/"+str(i)+".png", cv2.IMREAD_GRAYSCALE)

		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		pb_lite_img = compute_pb_lite(Tg_mean, Bg_mean, Cg_mean, canny_baseline, sobel_baseline)

		cv2.imwrite("PbLite_"+str(i)+".png", pb_lite_img)
    
if __name__ == '__main__':
    main()
 


