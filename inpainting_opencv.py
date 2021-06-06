import cv2
import numpy as np

img = cv2.imread("ISIC_0000095.jpg")
img = cv2.pyrDown(img)
image = cv2.pyrDown(img)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mask = cv2.Canny(gray, 40, 200)
kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
dilation=cv2.dilate(mask,kernel2,iterations=2)
cv2.imshow("Edge", dilation)

dst = cv2.inpaint(image, dilation,5,cv2.INPAINT_TELEA)
cv2.imshow("Orijinal", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
