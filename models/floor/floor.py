# import sys
import numpy as np
import cv2
import time
import math
import copy
# import winsound
# import pygame

# import pymssql
from datetime import datetime

class Box(object):
    def __init__(self, l, t, r, b, img):
        # self.x = [l, l, r, r]
        # self.y = [b, t, t, b]
        self.lft = max(l,0)
        self.top = max(t,0)
        self.rgt = min(r,width-1)
        self.bot = min(b,height-1)
        # Limit 초기화
        lLim = []
        tLim = []
        rLim = []
        bLim = []
        for x in range(width):
            tLim.append(height)
            bLim.append(-1)
        for y in range(height):
            lLim.append(width)
            rLim.append(-1)
        for x in range(l, r + 1):
            tLim[x] = self.top # min(tLim[x], b)
            bLim[x] = self.bot # max(bLim[x], t)
        for y in range(t, b + 1):
            lLim[y] = self.lft # min(lLim[y], l)
            rLim[y] = self.rgt # max(rLim[y], r)

        self.rgtLim = np.array(rLim)
        self.lftLim = np.array(lLim)
        self.topLim = np.array(tLim)
        self.botLim = np.array(bLim)

        self.pixNormal = np.array(img)  # self.imgNormal = copy.deepcopy(img) #청결한 상태의 이미지 세팅
        self.lastNormalTime = time.time()

        self.nPixel = 0
        self.sumB = 0
        self.sumG = 0
        self.sumR = 0
        for y in range(self.top, self.bot + 1):
            for x in range(self.lftLim[y], self.rgtLim[y] + 1):
                self.nPixel = self.nPixel + 1
                b, g, r = self.pixNormal[y][x]
                self.sumB = self.sumB + int(b)
                self.sumG = self.sumG + int(g)
                self.sumR = self.sumR + int(r)

        self.nPixel = max(self.nPixel, 1) # divide 0 error prevention           
        self.meanB = self.sumB / self.nPixel
        self.meanG = self.sumG / self.nPixel
        self.meanR = self.sumR / self.nPixel

        varB = 0
        varG = 0
        varR = 0
        for y in range(self.top, self.bot + 1):
            for x in range(self.lftLim[y], self.rgtLim[y] + 1):
                b, g, r = self.pixNormal[y][x]
                varB = varB + (int(b) - self.meanB) * (int(b) - self.meanB)
                varG = varG + (int(g) - self.meanG) * (int(g) - self.meanG)
                varR = varR + (int(r) - self.meanR) * (int(r) - self.meanR)

        self.StdB = max(math.sqrt(varB / self.nPixel), 1) # divide 0 error prevention
        self.StdG = max(math.sqrt(varG / self.nPixel), 1) # divide 0 error prevention
        self.StdR = max(math.sqrt(varR / self.nPixel), 1) # divide 0 error prevention
        # self.drawInside()
        # print("box created")

        # 박스가 생성된 시간
        self.setTime = time.time()
        

    def same(self, img):
        # height1, width1, channel1 = self.imgNormal.shape
        # height2, width2, channel2 = img.shape
        # if height1 != height2 or width1 != width2 or channel1 != channel2: return False
        pix = np.array(img)  # 픽셀 정보를 떼 내어서 어레이로 만든다

        nPixel = 0
        sumB = 0
        sumG = 0
        sumR = 0

        # y_step = max((self.bot +1 - self.top)//10, 1)

        for y in range(self.top, self.bot + 1):# , y_step):
            # x_step = 1 # max((self.rgtLim[y] + 1 - self.lftLim[y]) // 10, 1)

            for x in range(self.lftLim[y], self.rgtLim[y] + 1):
                nPixel = nPixel + 1
                b, g, r = pix[y][x]
                sumB = sumB + int(b)
                sumG = sumG + int(g)
                sumR = sumR + int(r)

        nPixel = max(nPixel, 1) # divide 0 error prevention       
        meanB = sumB / nPixel
        meanG = sumG / nPixel
        meanR = sumR / nPixel

        varB = 0
        varG = 0
        varR = 0
        for y in range(self.top, self.bot + 1):# , y_step):
            # x_step = 1 # max((self.rgtLim[y] + 1 - self.lftLim[y]) // 10, 1)

            for x in range(self.lftLim[y], self.rgtLim[y] + 1):
                b, g, r = pix[y][x]
                varB = varB + (int(b) - meanB) * (int(b) - meanB)
                varG = varG + (int(g) - meanG) * (int(g) - meanG)
                varR = varR + (int(r) - meanR) * (int(r) - meanR)

        StdB = max(math.sqrt(varB / nPixel), 1) # divide 0 error prevention
        StdG = max(math.sqrt(varG / nPixel), 1) # divide 0 error prevention
        StdR = max(math.sqrt(varR / nPixel), 1) # divide 0 error prevention

        nDifB = 0
        nDifG = 0
        nDifR = 0
        # imgNormal = copy.deepcopy(self.pixNormal)
        # cv2.imshow("Kitchen", imgNormal)
        # cv2.waitKey(1)
        # imgcopy = copy.deepcopy(img)
        # cv2.imshow("Kitchen", imgcopy)
        # cv2.waitKey(1)

        # imgN = copy.deepcopy(self.pixNormal)
        # imgR = copy.deepcopy(self.pixNormal)
        for y in range(self.top, self.bot + 1):# , y_step):
            # x_step = 1 # max((self.rgtLim[y] + 1 - self.lftLim[y]) // 10, 1)

            for x in range(self.lftLim[y], self.rgtLim[y] + 1):
                b, g, r = pix[y][x]
                sb, sg, sr = self.pixNormal[y][x]

                zB = (int(b) - meanB) / StdB
                zG = (int(g) - meanG) / StdG
                zR = (int(r) - meanR) / StdR
                zSelB = (int(sb) - self.meanB) / self.StdB
                zSelG = (int(sg) - self.meanG) / self.StdG
                zSelR = (int(sr) - self.meanR) / self.StdR
                diffB = abs(zB - zSelB)
                diffG = abs(zG - zSelG)
                diffR = abs(zR - zSelR)

                '''
                imgB = int(b)-meanB+255
                imgG = int(g)-meanG+255
                imgR = int(r)-meanR+255
                selB = int(sb)-self.meanB+255
                selG = int(sg)-self.meanG+255
                selR = int(sr)-self.meanR+255
                diffB = abs(imgB - selB)       
                diffG = abs(imgG - selG)       
                diffR = abs(imgR - selR)
                '''

                if diffB > 0.5:  # selB*0.1:
                    nDifB = nDifB + 1
                    # imgcopy[y, x] = (0, 0, 255)
                if diffG > 0.5:  # selG*0.1:
                    nDifG = nDifG + 1
                    # imgcopy[y, x] = imgcopy[y, x] + (0, 255, 0)
                if diffR > 0.5:  # selR*0.1:
                    nDifR = nDifR + 1
                    # imgcopy[y, x] = imgcopy[y, x] + (255, 0, 0)

                '''
                if diffB > 20:#selB*0.1:
                    nDifB = nDifB+1
                    imgcopy[y,x] = (0,0,255)
                if diffG > 20:#selG*0.1:
                    nDifG = nDifG+1
                    imgcopy[y,x] = imgcopy[y,x] + (0,255,0)
                if diffR > 20:#selR*0.1:
                    nDifR = nDifR+1
                    imgcopy[y,x] = imgcopy[y,x] + (255,0,0)                
                '''

        # cv2.imshow("Kitchen", img)
        # cv2.waitKey(1)
        # cv2.imshow("Kitchen", imgcopy)
        # cv2.waitKey(1)

        nDif = nDifB + nDifG + nDifR
        threshold = 3 * nPixel * 0.1
        if nDif > threshold:
            # cv2.imshow("Kitchen", imgcopy)
            # cv2.waitKey(1)
            return False
        else:
            return True
# CCTV      
# cap = cv2.VideoCapture('rtsp://admin:torch439@@59.1.62.68:554/trackID=1')
# if not cap.isOpened():
#     print("camera open failed")
#     exit()
# ret, imgReal = cap.read()  # 실제 현장 이미지
# if not ret:
#     print("Can't read camera")
#     exit()

# TEST
imgReal = cv2.imread('wanjoo1.jpg')
imgEdit = copy.deepcopy(imgReal)  # 박스와 텍스트 등이 그려진 이미지
height, width, channel = imgReal.shape

# box = [Box(800, 300, 1200, 700, imgReal)]
box = [Box(0, 0, width-1, height-1, imgReal)]

input_img = cv2.imread('wanjoo3.jpg')

if box[0].same(input_img):
    print("정상")
else:
    print("비정상")
    
# if box[0].same(imgReal):
#     print("정상")
# else:
#     print("비정상")