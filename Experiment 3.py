import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("C:\Users\DELL\OneDrive\Desktop\python\sample 2.jpg", 0)

if img is None:
    print("Image not loaded!")
    exit()

#  Contrast Stretching

min_val, max_val = np.min(img), np.max(img)

if max_val > min_val:
    cs_img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
else:
    cs_img = img.copy()


#  Histogram Equalization

he_img = cv2.equalizeHist(img)


# Display ONLY images

plt.figure(figsize=(10,6))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Contrast Stretching")
plt.imshow(cs_img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Histogram Equalization")
plt.imshow(he_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()