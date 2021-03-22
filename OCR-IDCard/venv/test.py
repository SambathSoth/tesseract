import cv2
import numpy as np
import pytesseract
import os

per = 25
pixelThreshold=500

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread('C:\\Users\\Asus\\tesseract\\OCR-IDCard\\venv\\Query.jpg')
h,w,c = imgQ.shape

orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(imgQ, None)
impKp1 = cv2.drawKeypoints(imgQ, kp1, None)

cv2.imshow("KeyPointsQuery", impKp1)
cv2.imshow("Ouput", imgQ)
cv2.waitKey(0)