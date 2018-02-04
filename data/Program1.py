#Tristan Basil
#Assignment: Project 1 - cS460G Machine Learning, Dr. Harrison

import math
import sys

class Node:
    children = list()
    label = -1

    def label1(self):
        self.label = 1
        return self

    def label0(self):
        self.label = 0
        return self

#this class is only designed to work for the data in this project.
class TestData:
    debug = False
    featureLists = list()
    #featureList1 = list()
    #featureList2 = list()
    binLists = list()
    #binList1 = list()
    #binList2 = list()
    classLabelList = list()
    distinctClassLabels = set()
    unbranchedFeatures = {0, 1}

    def __debugPrint(self):
        if self.debug:
            print " --- DEBUG INFO --- "
            print "class labels: " + str(self.distinctClassLabels)
            for i in range(len(self.featureLists)):
                print "max feature " + str(i) + ": " + str(max(self.featureLists[i]))
                print "min feature " + str(i) + ": " + str(min(self.featureLists[i]))
                print "discretized bins feature 1:",
                for j in range(len(self.binLists[i])):
                    print(self.binLists[i][j]),
                    print "-",
                print ""
            print " --- END DEBUG --- "

    def __discretizeFeaturesEquidistant(self, numBins):
        for i in range(len(self.featureLists)):
            binMin = min(self.featureLists[i])
            binMax = max(self.featureLists[i])
            binSize = (binMax - binMin) / float(numBins)
            for j in range(numBins):
                self.binLists[i].append(binMin+(j*binSize))
            self.binLists[i].append(binMax)


        '''
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
        '''

    def __entropy(self, distinctClassLabels, classLabelList):

        entropyTot = 0.0
        #for each class label in the list
        for i in distinctClassLabels:
            #contains the number of instances for each class label
            numEquals=0.0
            #mark the number of values for each class label.
            for k in range(len(classLabelList)):
                if (int(classLabelList[k]) == int(i)):
                    numEquals+=1
            #print(i, numEquals)
            #add to the total entropy for each bin (making sure it can be evaluated)
            if (numEquals == 0 or len(classLabelList) == 0):
                entropyTot+=0
            else:
                #print(numEquals)
                ratioInBin = (numEquals/len(classLabelList))
                entropyTot += -(ratioInBin*math.log(ratioInBin, 2))

        return entropyTot

    def __informationGain(self, featureList, featureIndex, classLabelList):
        binList = self.binLists[featureIndex]

        informationGainTot = self.__entropy(self.distinctClassLabels, classLabelList)
        #count the instances in each bin
        #2d list containing indexes of contents for each bin
        binContents = list()
        for i in range(len(binList)-1):
            numInBin = 0.0
            #for each feature value, see if it fits in the current bin.
            for k in range(len(featureList)):
                binContents.append(list())
                if (featureList[k] >= binList[i] and featureList[k] <= binList[i+1]):
                    numInBin+=1
                    #add its class label to the bin contents
                    binContents[i].append(self.classLabelList[k])
            #continue calculating information gain
            informationGainTot -= (numInBin/len(featureList))*self.__entropy(self.distinctClassLabels, binContents[i])

        return informationGainTot

    def ID3(self, classLabelList, unbranchedFeatures):
        root = Node()
        num1 = 0
        num0 = 0
        #count the instances of 0's and 1's to check for uniformity.
        for i in range(len(classLabelList)):
            if classLabelList[i] == 0:
                num0+=1
            elif classLabel[i] == 1:
                num1+=1
        #if we have a uniform set, we're done! label the node.
        if num0==len(classLabelList):
            return root.label0()
        elif num1==len(classLabelList):
            return root.label1()
        #if we're out of features to branch on, return the most common label (with 0 breaking ties)
        elif len(unbranchedFeatures) == 0:
            if num0>=num1:
                return root.label0()
            elif num1>num0:
                return root.label1()
        #begin the algorithm!
        maxInfoGain = 0
        for i in range(len(unbranchedFeatures)):
            #need to do information gain
            print "just coded myself into a corner and need to make feature list extensible"


        



    #initialization takes a filename.
    def __init__(self, filename, numBins, debug):
        self.debug = debug
        file = None
        try:
            file = open(filename, "r")
        except:
            invalidNumFields = True
            print(filename, 'not found')
            exit -1
        #quickly get the number of features from the file, and initialize the lists.
        fileBeginning = file.tell()
        numFeatures = len(file.readline().split(',')) - 1
        for i in range(numFeatures):
            self.featureLists.append(list())
            self.binLists.append(list())
        file.seek(fileBeginning)

        #for each line in the file, parse the features and class labels into parallel lists.
        for line in file:
            parsedLine = line.split(',')
            for i in range(len(self.featureLists)):
                self.featureLists[i].append(float(parsedLine[i]))
            classLabel = parsedLine[len(parsedLine)-1].rstrip()
            self.classLabelList.append(classLabel)
            self.distinctClassLabels.add(classLabel)

        self.__discretizeFeaturesEquidistant(numBins)
        self.__debugPrint()
        #self.printList()
        self.__informationGain(self.featureLists[0], 0, self.classLabelList)
        #print("entropy",self.__entropy(self.distinctClassLabels, self.classLabelList))

    def printList(self):
        for j in range(len(self.featureLists[0])):
            print str(self.featureLists[0][j]), str(self.featureLists[1][j]),
            print self.classLabelList[j]



def main():
    if (len(sys.argv) != 2):
        print "Takes 1 command line argument: the name of the csv file."
        exit -1;
    filename = sys.argv[1]
    #initialize the TestData object
    isDebugMode = True
    numBins = 10
    rawData = TestData(filename, numBins, isDebugMode)
    print 'done'

main()