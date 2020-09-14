from imutils import face_utils
from imutils.video import FileVideoStream
from imutils.video import FPS
from scipy.spatial import distance as dist
import datetime
import argparse
import imutils
from breakdownClasses import *
import numpy as np
import time
import dlib
import cv2
import math
from collections import OrderedDict

class videoClass(object):

    def __init__(self, software, video, statsColor, circleSize, jawDA, eyeBDA, noseDA, eyesDA, mouthDA, landmarkDivision=[2,2,2,2,2]):

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.video = video
        self.software = software
        self.landmarkDivision = landmarkDivision
        print(self.landmarkDivision)
        self.currentFrame = None
        self.currentShape = None
        self.currentCompare = None
        self.grayScale = None
        self.FACIAL_LANDMARKS_IDXS = OrderedDict([
        	("mouth", (48, 68)),
        	("inner_mouth", (60, 68)),
        	("right_eyebrow", (17, 22)),
        	("left_eyebrow", (22, 27)),
        	("right_eye", (36, 42)),
        	("left_eye", (42, 48)),
        	("nose", (27, 36)),
        	("jaw", (0, 17))
        ])
        self.faces = None
        self.shapeList = []
        self.actionAreaCount = {"Jaw":0, "Eyebrows":0, "Nose":0, "Eyes":0, "Mouth":0}
        self.statsValues = {"Jaw":0, "Eyebrows":0, "Nose":0, "Eyes":0, "Mouth":0}
        self.statsColor = statsColor
        self.jawDA = jawDA
        self.eyeBDA = eyeBDA
        self.noseDA = noseDA
        self.eyesDA = eyesDA
        self.mouthDA = mouthDA
        self.circleSize = []
        if circleSize == "Vs":
            self.circleSize = [1,2,3,4,5]
        elif circleSize == "S":
            self.circleSize = [2,4,6,8,10]
        elif circleSize == "M":
            self.circleSize = [4,6,8,10,12]
        elif circleSize == "L":
            self.circleSize = [6,8,10,12,14]


        self.beginStream()

    def detectFaces(self):

        self.currentFrame = imutils.resize(self.currentFrame, width=700)
        self.grayScale = cv2.cvtColor(self.currentFrame, cv2.COLOR_BGR2GRAY)
        self.faces = self.detector(self.grayScale, 1)

    def visualizeActionPoints(self):

        def determineTransparency(coordinates, alpha, circleSize, color):
            overlay = self.currentFrame.copy()
            cv2.circle(overlay, (coordinates[0], coordinates[1]), circleSize, color, -1)
            self.currentFrame = cv2.addWeighted(overlay, alpha, self.currentFrame, 1 - alpha, 0)

        def updateAACount(AACountVar, regionMovVar):
            oldCount = self.actionAreaCount[AACountVar]
            newCount = oldCount + self.shapeList[-1].regionMovement.get(regionMovVar)
            self.actionAreaCount[AACountVar] = newCount


        for i in self.shapeList:
            if self.shapeList[-1].shapeID - 20 >= i.shapeID >= self.shapeList[-1].shapeID - 25:
                for j in i.actionPoints:
                    determineTransparency((j.pointCords[0], j.pointCords[1]), 0.01, self.circleSize[0], (255,243,59))
            elif self.shapeList[-1].shapeID - 15 >= i.shapeID >= self.shapeList[-1].shapeID - 20:
                for j in i.actionPoints:
                    determineTransparency((j.pointCords[0], j.pointCords[1]), 0.02, self.circleSize[1], (253,199,12))
            elif self.shapeList[-1].shapeID - 10 >= i.shapeID >= self.shapeList[-1].shapeID - 15:
                for j in i.actionPoints:
                    determineTransparency((j.pointCords[0], j.pointCords[1]), 0.03, self.circleSize[2], (243,144,63))
            elif self.shapeList[-1].shapeID - 5 >= i.shapeID >= self.shapeList[-1].shapeID - 10:
                for j in i.actionPoints:
                    determineTransparency((j.pointCords[0], j.pointCords[1]), 0.05, self.circleSize[3], (237,104,60))
            elif self.shapeList[-1].shapeID - 1 >= i.shapeID >= self.shapeList[-1].shapeID - 5:
                for j in i.actionPoints:
                    determineTransparency((j.pointCords[0], j.pointCords[1]), 0.1, self.circleSize[4], (233,62,58))

        updateAACount("Jaw", "jaw")
        updateAACount("Eyebrows", "eyebrows")
        updateAACount("Nose", "nose")
        updateAACount("Eyes", "eyes")
        updateAACount("Mouth", "mouth")

    def visualizeActionPoints2(self):

        def updateCountDict(pointID, lower, upper, dictKey):

            if lower <= pointID <= upper:
                newValue = self.currentShape.countDict[dictKey]
                newValue[0] += 1
                self.currentShape.countDict[dictKey] = newValue

        for i in (self.currentShape.actionPoints):
            updateCountDict(i.pointID, 1, 17, "jaw")
            updateCountDict(i.pointID, 18, 22, "rEB")
            updateCountDict(i.pointID, 23, 27, "lEB")
            updateCountDict(i.pointID, 28, 36, "nose")
            updateCountDict(i.pointID, 37, 42, "rEye")
            updateCountDict(i.pointID, 43, 48, "lEye")
            updateCountDict(i.pointID, 49, 60, "oMouth")
            updateCountDict(i.pointID, 60, 68, "iMouth")

        def determineTransparency(alpha, shapeID):
            overlay = self.currentFrame.copy()
            for (i, name) in enumerate(self.FACIAL_LANDMARKS_IDXS.keys()):
                (j, k) = self.FACIAL_LANDMARKS_IDXS[name]
                pts = self.shapeList[shapeID].pureShape[j:k]
                if name == "mouth" and self.shapeList[shapeID].countDict.get("oMouth")[0] > self.shapeList[shapeID].countDict.get("oMouth")[1] / self.landmarkDivision[4]:
                    hull = cv2.convexHull(pts)
                    cv2.drawContours(overlay, [hull], -1, (233,62,58), -1)
                    self.currentFrame = cv2.addWeighted(overlay, alpha, self.currentFrame, 1 - alpha, 0)
                elif name == "inner_mouth" and self.shapeList[shapeID].countDict.get("iMouth")[0] > self.shapeList[shapeID].countDict.get("iMouth")[1] / self.landmarkDivision[4]:
                    hull = cv2.convexHull(pts)
                    cv2.drawContours(overlay, [hull], -1, (233,62,58), -1)
                    self.currentFrame = cv2.addWeighted(overlay, alpha, self.currentFrame, 1 - alpha, 0)
                elif name == "right_eyebrow" and self.shapeList[shapeID].countDict.get("rEB")[0] > self.shapeList[shapeID].countDict.get("rEB")[1] / self.landmarkDivision[3]:
                    hull = cv2.convexHull(pts)
                    cv2.drawContours(overlay, [hull], -1, (233,62,58), -1)
                    self.currentFrame = cv2.addWeighted(overlay, alpha, self.currentFrame, 1 - alpha, 0)
                elif name == "left_eyebrow" and self.shapeList[shapeID].countDict.get("lEB")[0] > self.shapeList[shapeID].countDict.get("lEB")[1] / self.landmarkDivision[3]:
                    hull = cv2.convexHull(pts)
                    cv2.drawContours(overlay, [hull], -1, (233,62,58), -1)
                    self.currentFrame = cv2.addWeighted(overlay, alpha, self.currentFrame, 1 - alpha, 0)
                elif name == "nose" and self.shapeList[shapeID].countDict.get("nose")[0] > self.shapeList[shapeID].countDict.get("nose")[1] / self.landmarkDivision[2]:
                    hull = cv2.convexHull(pts)
                    cv2.drawContours(overlay, [hull], -1, (233,62,58), -1)
                    self.currentFrame = cv2.addWeighted(overlay, alpha, self.currentFrame, 1 - alpha, 0)
                elif name == "right_eye" and self.shapeList[shapeID].countDict.get("rEye")[0] > self.shapeList[shapeID].countDict.get("rEye")[1] / self.landmarkDivision[1]:
                    hull = cv2.convexHull(pts)
                    cv2.drawContours(overlay, [hull], -1, (233,62,58), -1)
                    self.currentFrame = cv2.addWeighted(overlay, alpha, self.currentFrame, 1 - alpha, 0)
                elif name == "left_eye" and self.shapeList[shapeID].countDict.get("lEye")[0] > self.shapeList[shapeID].countDict.get("lEye")[1] / self.landmarkDivision[1]:
                    hull = cv2.convexHull(pts)
                    cv2.drawContours(overlay, [hull], -1, (233,62,58), -1)
                    self.currentFrame = cv2.addWeighted(overlay, alpha, self.currentFrame, 1 - alpha, 0)
                elif name == "jaw" and self.shapeList[shapeID].countDict.get("jaw")[0] > self.shapeList[shapeID].countDict.get("jaw")[1] / self.landmarkDivision[0]:
                    for l in range(1, len(pts)):
                        ptA = tuple(pts[l - 1])
                        ptB = tuple(pts[l])
                        cv2.line(overlay, ptA, ptB, (233,62,58), 2)
                        self.currentFrame = cv2.addWeighted(overlay, alpha, self.currentFrame, 1 - alpha, 0)



        def updateAACount(AACountVar, regionMovVar):
            oldCount = self.actionAreaCount[AACountVar]
            newCount = oldCount + self.shapeList[-1].regionMovement.get(regionMovVar)
            self.actionAreaCount[AACountVar] = newCount


        for i in self.shapeList:
            if self.shapeList[-1].shapeID - 20 >= i.shapeID >= self.shapeList[-1].shapeID - 25:
                shapeID = self.shapeList[-1].shapeID - i.shapeID
                determineTransparency(0.01, -shapeID)
            elif self.shapeList[-1].shapeID - 15 >= i.shapeID >= self.shapeList[-1].shapeID - 20:
                shapeID = self.shapeList[-1].shapeID - i.shapeID
                determineTransparency(0.02, -shapeID)
            elif self.shapeList[-1].shapeID - 10 >= i.shapeID >= self.shapeList[-1].shapeID - 15:
                shapeID = self.shapeList[-1].shapeID - i.shapeID
                determineTransparency(0.03, -shapeID)
            elif self.shapeList[-1].shapeID - 5 >= i.shapeID >= self.shapeList[-1].shapeID - 10:
                shapeID = self.shapeList[-1].shapeID - i.shapeID
                determineTransparency(0.05, -shapeID)
            elif self.shapeList[-1].shapeID - 1 >= i.shapeID >= self.shapeList[-1].shapeID - 5:
                shapeID = self.shapeList[-1].shapeID - i.shapeID
                determineTransparency(0.08, -shapeID)

        updateAACount("Jaw", "jaw")
        updateAACount("Eyebrows", "eyebrows")
        updateAACount("Nose", "nose")
        updateAACount("Eyes", "eyes")
        updateAACount("Mouth", "mouth")

    def statisticsOverlay(self):

        def percentage(part, whole):
            if part == 0:
                return 0
            else:
                return 100 * float(part)/float(whole)

        def displayText(text, percent, height, width, heightMinus):
            cv2.putText(self.currentFrame, text + str(percent) + "%",
            (int((width/100) * 5), height - heightMinus), 0, 0.5, self.statsColor)

        def displayVisual(statsOverlay, visualValue, height, width, heightMinus):
            cv2.line(statsOverlay, (int((width/100) * 40), height - heightMinus),
            (int((width/100) * 40) +
            (visualValue * 2), height - heightMinus), self.statsColor, 12)


        total = 0
        for x, y in self.actionAreaCount.items():
            total += y

        roundedValues = []

        for x, y in self.actionAreaCount.items():
            visualPercent = math.ceil(percentage(y, total))
            roundedValues.append(visualPercent)

        frameHeight, frameWidth = self.currentFrame.shape[:2]

        displayText("Jaw ", roundedValues[0], frameHeight, frameWidth, 22)
        displayText("Eyebrows ", roundedValues[1], frameHeight, frameWidth, 47)
        displayText("Nose ", roundedValues[2], frameHeight, frameWidth, 72)
        displayText("Eyes ", roundedValues[3], frameHeight, frameWidth, 97)
        displayText("Mouth ", roundedValues[4], frameHeight, frameWidth, 122)

        newImage = self.currentFrame.copy()

        displayVisual(newImage, roundedValues[0], frameHeight, frameWidth, 25)
        displayVisual(newImage, roundedValues[1], frameHeight, frameWidth, 50)
        displayVisual(newImage, roundedValues[2], frameHeight, frameWidth, 75)
        displayVisual(newImage, roundedValues[3], frameHeight, frameWidth, 100)
        displayVisual(newImage, roundedValues[4], frameHeight, frameWidth, 125)

        alpha = 0.4

        cv2.addWeighted(newImage, alpha, self.currentFrame, 1 - alpha, 0, self.currentFrame)
        self.statsValues["Jaw"] = roundedValues[0]
        self.statsValues["Eyebrows"] = roundedValues[1]
        self.statsValues["Nose"] = roundedValues[2]
        self.statsValues["Eyes"] = roundedValues[3]
        self.statsValues["Mouth"] = roundedValues[4]

    def determineActionPoints(self):

        self.detectFaces()

        for (i, self.faces) in enumerate(self.faces):
            shape = self.predictor(self.grayScale, self.faces)
            shape = face_utils.shape_to_np(shape)
            self.currentShape = shapeClass(shape, len(self.shapeList) + 1)
            self.shapeList.append(self.currentShape)
            if len(self.shapeList) > 10:
                self.currentCompare = shapeCompareClass(self.shapeList[-1], self.shapeList[-10],
                self.jawDA, self.eyeBDA, self.noseDA, self.eyesDA, self.mouthDA)
                for i in self.shapeList[-1].shapePointsArray:
                    if self.currentCompare.returnCPDDifference(i.pointID) == True or self.currentCompare.returnCPADifference(i.pointID) == True:
                        self.shapeList[-1].actionPoints.append(i)
                        for j in self.shapeList[-1].regionMovement:
                            if i.facialRegion == j:
                                self.shapeList[-1].regionMovement[j] = self.shapeList[-1].regionMovement[j] + 1



            if self.software == "Point":
                self.visualizeActionPoints()
            elif self.software == "Landmark":
                self.visualizeActionPoints2()
            self.statisticsOverlay()


    def beginStream(self):

        videoStream = FileVideoStream(self.video).start()
        time.sleep(1.0)
        fps = FPS().start()
        count = 0
        while videoStream.Q.qsize() != 1:
            self.currentFrame = videoStream.read()
            self.determineActionPoints()
            cv2.imshow("Image", self.currentFrame)
            key = cv2.waitKey(1) & 0xFF
            cv2.imwrite("images/" + str(count) + ".jpg", self.currentFrame)
            count += 1
            fps.update()
        for x, y in self.statsValues.items():
            print(x, y)
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}". format(fps.fps()))
        cv2.destroyAllWindows()
        videoStream.stop()
