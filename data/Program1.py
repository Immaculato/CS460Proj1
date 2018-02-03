#Tristan Basil
#Assignment: Project 1 - cS460G Machine Learning, Dr. Harrison

import math

FILENAME='synthetic-1.csv'

#this class is only designed to work for the data in this project.
class TestData:
    debug = False
    featureList1 = list()
    featureList2 = list()
    binList1 = list()
    binList2 = list()
    classLabelList = list()
    distinctClassLabels = set()

    def __debugPrint(self):
        if self.debug:
            print " --- DEBUG INFO --- "
            print "class labels: " + str(self.distinctClassLabels)
            print "max feature 1: " + str(max(self.featureList1))
            print "min feature 1: " + str(min(self.featureList1))
            print "discretized bins feature 1:",
            for i in range(len(self.binList1)):
                print(self.binList1[i]),
                print "-",
            print ""
            print "max feature 2: " + str(max(self.featureList2))
            print "min feature 2: " + str(min(self.featureList2))
            print "discretized bins feature 2:",
            for i in range(len(self.binList2)):
                print(self.binList2[i]),
                print "-",
            print ""
            print " --- END DEBUG --- "

    def __discretizeFeaturesEquidistant(self, numBins):
        binMin1 = min(self.featureList1)
        binMin2 = min(self.featureList2)
        binMax1 = max(self.featureList1)
        binMax2 = max(self.featureList2)
        binSize1 = (binMax1 - binMin1) / float(numBins)
        binSize2 = (binMax2 - binMin2) / float(numBins)
        #using the distance we found for each bin, append these values to the binList.
        for i in range(numBins):
            self.binList1.append(binMin1+(i*binSize1))
            self.binList2.append(binMin2+(i*binSize2))
        #small errors can mean we miss the end value, so append those on manually.
        self.binList1.append(binMax1)
        self.binList2.append(binMax2)

    def __entropy(self, featureList, featureIndex):
        binList = None
        if (featureIndex == 1):
            binList = self.binList1
        elif (featureIndex == 2):
            binList = self.binList2

        print(len(featureList))
        entropyTot = 0.0
        #for each bin in the list
        for i in range(len(binList)-1):
            numInBin = 0.0
            #for each feature value, see if it fits in the current bin.
            for k in range(len(featureList)):
                if (featureList[k] >= binList[i] and featureList[k] <= binList[i+1]):
                    numInBin+=1
            #add to the total entropy for each bin.
            ratioInBin = (numInBin/len(featureList))
            print(ratioInBin)
            if ratioInBin == 0:
                entropyTot+=0
            else:
                entropyTot += -(ratioInBin*math.log(ratioInBin, 2))

        return entropyTot


    #initialization takes a filename.
    def __init__(self, filename, numBins, debug):
        self.debug = debug
        file = None
        try:
            file = open(FILENAME, "r")
        except:
            invalidNumFields = True
            print(FILENAME, 'not found')
            exit -1
        #for each line in the file, parse the features and class labels into parallel lists.
        for line in file:
            parsedLine = line.split(',')
            self.featureList1.append(float(parsedLine[0]))
            self.featureList2.append(float(parsedLine[1]))
            classLabel = parsedLine[2].rstrip()
            self.classLabelList.append(classLabel)
            self.distinctClassLabels.add(classLabel)

        self.__discretizeFeaturesEquidistant(numBins)
        self.__debugPrint()
        print("entropy",self.__entropy(self.featureList1, 1))

    def printList(self):
        for i in range(len(self.featureList1)):
            print(self.featureList1[i]+','+self.featureList2[i]+','+self.classLabelList[i])

    
class FeatureBin:
    binMin = 0.0
    binMax = 0.0

    def __init__(self, min, max):
        self.binMax = max
        self.binMin = min

    def printBin(self):
        print(self.binMin, self.binMax),



def main():
    #initialize the TestData object
    isDebugMode = True
    numBins = 10
    rawData = TestData(FILENAME, numBins, isDebugMode)
    print 'done'

main()