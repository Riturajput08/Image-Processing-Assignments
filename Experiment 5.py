import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(r"C:\Users\DELL\OneDrive\Desktop\python\sample 3.jpg", 0) 

if img is None:
    print("Error: Image not loaded. Check the path!")
    exit()


#  Add Gaussian noise to the image

row, col = img.shape
mean = 0
sigma = 25

gauss = np.random.normal(mean, sigma, (row, col)).reshape(row, col)
noisy_img = img + gauss
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)


#  Apply Mean Filter (Averaging)

mean_filtered = cv2.blur(noisy_img, (5,5))  # kernel size 5x5


# Apply Median Filter

median_filtered = cv2.medianBlur(noisy_img, 5)  # kernel size 5


# Apply Gaussian Filter

gaussian_filtered = cv2.GaussianBlur(noisy_img, (5,5), 0)  # kernel size 5x5, sigma=0


#  Display results

plt.figure(figsize=(12,6))

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(noisy_img, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(mean_filtered, cmap='gray')
plt.title('Mean Filtered')
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(median_filtered, cmap='gray')
plt.title('Median Filtered')
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(gaussian_filtered, cmap='gray')
plt.title('Gaussian Filtered')
plt.axis('off')

plt.show()