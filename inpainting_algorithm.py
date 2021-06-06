import cv2
import numpy as np

img = cv2.imread("ISIC_0000095.jpg")
img = cv2.pyrDown(img)
image = cv2.pyrDown(img)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray, 50, 200)
contours, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contoured = cv2.drawContours(image, contours, -1, (255, 255, 255), 0)

imgheight=image.shape[0]
imgwidth=image.shape[1]

M = imgheight//15
N = imgwidth//15

sonuc = contoured.copy()
for y in range(0, imgheight, M):
    for x in range(0, imgwidth, N):
        y1 = y + M
        x1 = x + N
        grid = contoured[y:y+M,x:x+N]
        roi  = ((x, y), (x1, y1))
        roi_resim = contoured[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
        
        B, G, R = cv2.split(roi_resim)
        
        B255 = (np.count_nonzero(B>=250))//len(B)
        G255 = (np.count_nonzero(G>=255))//len(G)
        R255 = (np.count_nonzero(R>=255))//len(R)
        
        B[B>=250] = np.average(B) - B255
        G[G>=250] = np.average(G) - G255
        R[R>=250] = np.average(R) - R255
        
        cikti = cv2.merge((B,G,R))
        sonuc[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]] = cikti

kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
dilation=cv2.dilate(sonuc,kernel2,iterations=1)

cv2.imshow("Original", cv2.pyrDown(img))
cv2.imshow("Dilation", dilation)
cv2.waitKey(0)
