
import cv2

import numpy as np
import os



aaa = cv2.imread('seed0075.png')
aaa2 = cv2.imread('seed00752.png')

ddd = np.mean((aaa2 - aaa)**2)
print('ddd=%.6f' % ddd)


print()









