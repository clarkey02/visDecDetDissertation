from scipy.spatial import distance as dist
import math
import numpy as np

class shapeClass(object):
    def __init__(self, shapePoints, shapeID=0):

        self.shapePointsArray = []
        self.pureShape = shapePoints
        self.shapeCenterCords = shapePoints[31-1]
        self.shapeID = shapeID
        self.actionPoints = []
        self.rightPointInclusion = []
        self.leftPointInclusion = []
        self.sensitivityBias = ""
        self.sensitivityCorrection = 0
        self.countDict = {"jaw": [0,17], "rEB": [0,5], "lEB": [0,5],
        "nose": [0,9], "rEye": [0,6], "lEye": [0,6],
        "oMouth": [0,12], "iMouth": [0,9] }
        self.regionMovement = {"jaw":0, "eyebrows":0, "nose":0, "eyes":0, "mouth":0}

        count = 1

        for i in shapePoints:
            p = pointClass(i, count, self.shapeCenterCords)
            count += 1
            self.shapePointsArray.append(p)

        self.defineSensitivityCorr()

    def defineSensitivityCorr(self):

        for i in self.shapePointsArray:
            if i.pointID == 2:
                rightSideDist = i.distanceFromCenter()
            elif i.pointID == 16:
                leftSideDist = i.distanceFromCenter()

        leftSidePercent = 100 * (float(leftSideDist) / float(leftSideDist + rightSideDist))
        rightSidePercent = 100 * (float(rightSideDist) / float(leftSideDist + rightSideDist))

        if leftSidePercent > rightSidePercent:
            self.sensitivityBias = "Right"
            self.sensitivityCorrection = abs(leftSidePercent / rightSidePercent)
        elif rightSidePercent > leftSidePercent:
            self.sensitivityBias = "Left"
            self.sensitivityCorrection = abs(rightSidePercent / leftSidePercent)

class pointClass(object):
    def __init__(self, pointCords, pointID, relativeCenterPoint):

        self.pointCords = pointCords
        self.pointID = pointID
        self.relativeCenterPoint = relativeCenterPoint
        self.facialRegion = ""
        self.facialHalf = "Middle"

        if 1 <= self.pointID <= 17:
            self.facialRegion = "jaw"
        elif 18 <= self.pointID <= 27:
            self.facialRegion = "eyebrows"
        elif 28 <= self.pointID <= 36:
            self.facialRegion = "nose"
        elif 37 <= self.pointID <= 48:
            self.facialRegion = "eyes"
        elif 49 <= self.pointID <= 68:
            self.facialRegion = "mouth"

        if 1 <= self.pointID <= 8:
            self.facialHalf = "Right"
        elif 10 <= self.pointID <= 17:
            self.facialHalf = "Left"
        elif 18 <= self.pointID <= 22:
            self.facialHalf = "Right"
        elif 23 <= self.pointID <= 27:
            self.facialHalf = "Left"
        elif self.pointID == 32:
            self.facialHalf = "Right"
        elif self.pointID == 36:
            self.facialHalf = "Left"
        elif 37 <= self.pointID <= 42:
            self.facialHalf  = "Right"
        elif 43 <= self.pointID <= 48:
            self.facialHalf  = "Left"
        elif 49 <= self.pointID <= 51:
            self.facialHalf  = "Right"
        elif 53 <= self.pointID <= 57:
            self.facialHalf  = "Left"
        elif 59 <= self.pointID <= 62:
            self.facialHalf  = "Right"
        elif 64 <= self.pointID <= 66:
            self.facialHalf  = "Left"
        elif self.pointID == 68:
            self.facialHalf  = "Right"



    def distanceFromCenter(self):
        return dist.euclidean((self.pointCords), (self.relativeCenterPoint))

    def angleFromCenter(self):
        myradians = math.atan2(self.pointCords[1]-self.relativeCenterPoint[1], self.pointCords[0]-self.relativeCenterPoint[0])
        degs = math.degrees(myradians)
        return degs

class shapeCompareClass(object):
    def __init__(self, previousShape, currentShape, jawDA, eyeBDA, noseDA, eyesDA, mouthDA):

        self.previousShape = previousShape
        self.currentShape = currentShape
        self.jawDA = jawDA
        self.eyeBDA = eyeBDA
        self.noseDA = noseDA
        self.eyesDA = eyesDA
        self.mouthDA = mouthDA

    def returnShapeIDs(self):
        return self.previousShape.shapeID, self.currentShape.shapeID

    def returnCPDDifference(self, pointID):
        currentShapeComparisonPoint = 0
        previousShapeComparisonPoint = 0

        for i in self.currentShape.shapePointsArray:
            if i.pointID == pointID:
                currentShapeComparisonPoint = i
        for i in self.previousShape.shapePointsArray:
            if i.pointID == pointID:
                previousShapeComparisonPoint = i

        jawComp = self.jawDA[0]
        eyeBComp = self.eyeBDA[0]
        noseComp = self.noseDA[0]
        eyesComp = self.eyesDA[0]
        mouthComp = self.mouthDA[0]

        if currentShapeComparisonPoint.facialHalf == self.currentShape.sensitivityBias:
            jawComp = jawComp * self.currentShape.sensitivityCorrection
            eyeBComp = eyeBComp * self.currentShape.sensitivityCorrection
            eyeBComp = eyeBComp - ((5 * eyeBComp) / 100.0)
            noseComp = noseComp * self.currentShape.sensitivityCorrection
            eyesComp = eyesComp * self.currentShape.sensitivityCorrection
            eyesComp = eyesComp - ((0 * eyesComp) / 100.0)
            mouthComp = mouthComp * self.currentShape.sensitivityCorrection
            mouthComp = mouthComp - ((5 * mouthComp) / 100.0)

        distDifference = abs(currentShapeComparisonPoint.distanceFromCenter() - previousShapeComparisonPoint.distanceFromCenter())
        if currentShapeComparisonPoint.facialRegion == "jaw":
            if distDifference > ((previousShapeComparisonPoint.distanceFromCenter() / 100) * jawComp):
                return True
        elif currentShapeComparisonPoint.facialRegion == "eyebrows":
            if distDifference > ((previousShapeComparisonPoint.distanceFromCenter() / 100) * eyeBComp):
                return True
        elif currentShapeComparisonPoint.facialRegion == "nose":
            if distDifference > ((previousShapeComparisonPoint.distanceFromCenter() / 100) * noseComp):
                return True
        elif currentShapeComparisonPoint.facialRegion == "eyes":
            if distDifference > ((previousShapeComparisonPoint.distanceFromCenter() / 100) * eyesComp):
                return True
        elif currentShapeComparisonPoint.facialRegion == "mouth":
            if distDifference > ((previousShapeComparisonPoint.distanceFromCenter() / 100) * mouthComp):
                return True

    def returnCPADifference(self, pointID):
        currentShapeComparisonPoint = 0
        previousShapeComparisonPoint = 0

        for i in self.currentShape.shapePointsArray:
            if i.pointID == pointID:
                currentShapeComparisonPoint = i
        for i in self.previousShape.shapePointsArray:
            if i.pointID == pointID:
                previousShapeComparisonPoint = i

        jawComp = self.jawDA[1]
        eyeBComp = self.eyeBDA[1]
        noseComp = self.noseDA[1]
        eyesComp = self.eyesDA[1]
        mouthComp = self.mouthDA[1]

        if currentShapeComparisonPoint.facialHalf == self.currentShape.sensitivityBias:
            jawComp = jawComp * self.currentShape.sensitivityCorrection
            eyeBComp = eyeBComp * self.currentShape.sensitivityCorrection
            eyeBComp = eyeBComp - ((5 * eyeBComp) / 100.0)
            noseComp = noseComp * self.currentShape.sensitivityCorrection
            eyesComp = eyesComp * self.currentShape.sensitivityCorrection
            eyesComp = eyesComp - ((0 * eyesComp) / 100.0)
            mouthComp = mouthComp * self.currentShape.sensitivityCorrection
            mouthComp = mouthComp - ((5 * mouthComp) / 100.0)

        angleDifference = abs(currentShapeComparisonPoint.angleFromCenter() - previousShapeComparisonPoint.angleFromCenter())
        if currentShapeComparisonPoint.facialRegion == "jaw":
            if angleDifference > jawComp:
                return True
        elif currentShapeComparisonPoint.facialRegion == "eyebrows":
            if angleDifference > eyeBComp:
                return True
        elif currentShapeComparisonPoint.facialRegion == "nose":
            if angleDifference > noseComp:
                return True
        elif currentShapeComparisonPoint.facialRegion == "eyes":
            if angleDifference > eyesComp:
                return True
        elif currentShapeComparisonPoint.facialRegion == "mouth":
            if angleDifference > mouthComp:
                return True
