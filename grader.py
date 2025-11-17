from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

answer_key={0: 3, 1: 1, 2: 0, 3: 2, 4: 1, 5: 3, 6: 0, 7: 1, 8: 2, 9: 0,
    10: 3, 11: 0, 12: 1, 13: 3, 14: 2, 15: 1, 16: 0, 17: 3, 18: 2, 19: 1,
    20: 3, 21: 2, 22: 0, 23: 1, 24: 3, 25: 0, 26: 2, 27: 1, 28: 3, 29: 0,
    30: 1, 31: 3, 32: 2, 33: 0, 34: 1, 35: 3, 36: 2, 37: 0, 38: 1, 39: 3,
    40: 0, 41: 2, 42: 1, 43: 3, 44: 0, 45: 2, 46: 1, 47: 3, 48: 0, 49: 2,
    50: 1, 51: 3, 52: 0, 53: 2, 54: 1, 55: 3, 56: 0, 57: 2, 58: 1, 59: 3}

bubbles_per_question=4
total_graded_questions=len(answer_key)

image=cv2.imread("omi.jpg")
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred=cv2.GaussianBlur(gray, (5,5), 0)
edged=cv2.Canny(blurred, 75, 200)

cnts=cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
docCnt=None

if len(cnts)>0:
    cnts= sorted(cnts, key=cv2.contourArea,reverse=True)

    for c in cnts:
        peri=cv2.arcLength(c, True)
        approx= cv2.approxPolyDP(c, 0.02*peri, True)

        if(len(approx)==4):
            docCnt= approx
            break
paper= four_point_transform(image, docCnt.reshape(4, 2))
warped= four_point_transform(gray, docCnt.reshape(4,2))
thresh= cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)[1]
cnts= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
questionCnt=[]

for c in cnts:
    (x, y, w, h)=cv2.boundingRect(c)
    ar=w/float(h)
    if w>=10 and w<=20 and h>=10 and h<=20 and ar>=0.9 and ar<=1.1:
        questionCnt.append(c)
questionCnt= contours.sort_contours(questionCnt, method="top-to-bottom")[0]
correct= 0

for (q, i) in enumerate(np.arange(0, len(questionCnt),5)):
    #5 bubbles in one ques
    cnts= contours.sort_contours(questionCnt[i: i+5],method="left-to-right")[0]
    bubbled= None

    for (j,c) in enumerate(cnts):
        mask= np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask,[c], -1, 255, -1)
        mask= cv2.bitwise_and(thresh, thresh, mask=mask)
        total= cv2.countNonZero(mask)

        if bubbled is None or total>bubbled[0]:
            bubbled= (total, j)
    color=(0,0,255)
    k= answer_key[q]
    if k==bubbled[1]:
        color=(0,255,0)
        correct+=1
    cv2.drawContours(paper,[cnts[k]],-1,color, 3)
#print(questionCnt)
score=(correct/60.0)*100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper,"{:.2f}%".format(score),(10,30), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
cv2.imshow("Exam",paper)
cv2.waitKey(0)




cv2.imshow("war",thresh)
cv2.waitKey(0)





