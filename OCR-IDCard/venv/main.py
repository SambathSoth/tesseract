import cv2
import numpy as np
import pytesseract
import os

per = 25
pixelThreshold=500

roi = [
    [(948, 10), (1264, 86), 'text', 'ID'],
    [(592, 58), (956, 142), 'khm', 'Name'], 
    [(576, 144), (1098, 182), 'text', 'Latin Name'], 
    [(566, 186), (874, 230), 'khm', 'DOB'], 
    [(962, 184), (1094, 236), 'khm', 'Gender'], 
    [(1208, 174), (1316, 230), 'khm', 'Height'], 
    [(560, 234), (1438, 300), 'khm', 'POB'], 
    [(528, 302), (1436, 372), 'khm', 'Address1'], 
    [(320, 378), (1438, 446), 'khm', 'Address2'], 
    [(512, 452), (762, 502), 'khm', 'From'], 
    [(872, 456), (1128, 500), 'khm', 'To'],
    [(6, 52), (316, 454), 'face', 'Face']
]

# roi = [
#     [(948, 10), (1264, 86), 'text', 'ID'],
#     [(592, 58), (956, 142), 'khm', 'Name'],

# ]

# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread('C:\\Users\\Asus\\tesseract\\OCR-IDCard\\venv\\Query.jpg')
h, w, c = imgQ.shape

orb = cv2.ORB_create(30000)
kp1, des1 = orb.detectAndCompute(imgQ, None)

path = 'C:\\Users\\Asus\\tesseract\\OCR-IDCard\\UserIDs'	
myPicList = os.listdir(path)
print(myPicList)
for j, y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)
    img = cv2.resize(img, (1440, 963))
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key= lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w,h))

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []


    for x, r in enumerate(roi):

        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]),(0, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask,0.1,0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]

        if r[2] == 'text':
            custom_config = r'-l eng --psm 6'
            field_text = pytesseract.image_to_string(imgCrop, config=custom_config)
            print('{} :{}'.format(r[3], field_text))
            myData.append(field_text)

        if r[2] == 'khm':
            custom_config = r'-l khm --psm 6'
            field_text = pytesseract.image_to_string(imgCrop, config=custom_config)
            print('{} :{}'.format(r[3], field_text))
            myData.append(field_text)
            
        if r[2] == 'address':
            custom_config = r'-l khm+eng --psm 6'
            field_text = pytesseract.image_to_string(imgCrop, config=custom_config)
            print('{} :{}'.format(r[3], field_text))
            myData.append(field_text)

        cv2.putText(imgShow,str(myData[x]), (r[0][0], r[0][1]),
                    cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 4)

    with open('DataOutput.csv', 'a+', encoding="utf-8") as f:
        for data in myData:
            f.write((str(data) + ','))
        f.write('\n')

    print(myData)
    cv2.imshow(y + "2", imgShow)
    cv2.imwrite(y, imgShow)

cv2.imshow("Output", imgQ)
cv2.waitKey(0)
