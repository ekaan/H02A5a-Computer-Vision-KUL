# -- Imports --
import numpy as np
from matplotlib import pyplot as plt
import cv2

# -- -- -- --

# DIFFERENT TRIALS 
org_img = cv2.imread("abc.png")
cv2.imshow('org',org_img)

img = cv2.cvtColor(org_img, cv2.COLOR_BGR2HSV)

# -- SMOOTHING ZONE --

kernel_blur = np.ones((5,5),np.float32)/25

blur1 = cv2.filter2D(img,-1,kernel_blur)

blur2 = cv2.blur(img,(100,100))

blur3 = cv2.GaussianBlur(img,(25,25),cv2.BORDER_DEFAULT)

blur4 = cv2.bilateralFilter(img,9,0,0)

# depends on the param, blur changes

# -- SHARPENING HAPPENS -- 

ker_sh = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])

ker_sh = np.array([[0 ,-1, 0],
                   [-1, 5,-1],
                   [0 ,-1, 0]])

ker_sh = np.array([[-1, -1,0],
                   [-1, 6, 0],
                   [-1 ,-1,0]])


ker_sh = np.array([[0 ,-1, 0],
                   [-1, 5,-1],
                   [0 ,-1, 0]])

sharp_img = cv2.filter2D(img, 0, ker_sh)


# -- GRAYSCALE CHANGING ZONE --

b_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

s_b_img = cv2.blur(b_img,(5,5))  # BLURRING THE GRAY IMG



# -- PRINTING ZONE --

hsv = cv2.cvtColor(org_img, cv2.COLOR_BGR2HSV)
        
low_h = 100
high_h = 200
low_s = 30
high_s = 255
low_v = 0
high_v = 255
        
hsv_thr = cv2.inRange(hsv, (low_h, low_s, low_v), (high_h, high_s, high_v))
        
result = cv2.cvtColor(hsv_thr, cv2.COLOR_GRAY2BGR)

cv2.imshow('result',result)

cv2.waitKey(1)

plt.imshow(blur1)
plt.imshow(blur2)
plt.imshow(blur3)
plt.imshow(blur4)
plt.imshow(sharp_img)
plt.imshow(s_b_img, cmap="gray")
print(b_img)
