import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image (update the path if needed)
img_path = r"C:\Users\DELL\OneDrive\Desktop\python\sample 3.jpg"  # Use raw string for Windows paths
img = cv2.imread(img_path, 0)

# Check if the image loaded successfully
if img is None:
    print(f"Error: Could not load image from {img_path}. Please check the file path and ensure the image exists.")
    exit(1)  # Exit the script if image loading fails

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

rows, cols = img.shape
crow, ccol = rows//2, cols//2

mask_lp = np.zeros((rows, cols), np.uint8)
mask_lp[crow-30:crow+30, ccol-30:ccol+30] = 1

mask_hp = 1 - mask_lp

lp = np.fft.ifft2(np.fft.ifftshift(fshift * mask_lp))
hp = np.fft.ifft2(np.fft.ifftshift(fshift * mask_hp))

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original'), plt.axis('off')
plt.subplot(1, 3, 2), plt.imshow(np.abs(lp), cmap='gray'), plt.title('Low Pass Filtered'), plt.axis('off')
plt.subplot(1, 3, 3), plt.imshow(np.abs(hp), cmap='gray'), plt.title('High Pass Filtered'), plt.axis('off')

plt.show()