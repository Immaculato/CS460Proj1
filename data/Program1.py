#Tristan Basil
#Assignment: Project 1 - cS460G Machine Learning, Dr. Harrison

import math
import sys
import copy
import pdb

class Node:
    children = None
    label = None
    featureIndex = None
    featureMax = None
    featureMin = None

    def printNode(self):
        print "featureIndex:", self.featureIndex, "featureMin/Max:", self.featureMin, self.featureMax, "label:", self.label,

#this class is only designed to work for the data in this project.
class ID3Tree:
    debug = False
    featureLists = list()
    #featureList1 = list()
    #featureList2 = list()
    binLists = list()
    #binList1 = list()
    #binList2 = list()
    classLabelList = list()
    distinctClassLabels = set()
    unbranchedFeatures = set()
    rootTreeNode = Node()

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
        #for each feature
        for i in range(len(self.featureLists)):
            #find the max and min of the feature, and use this to calculate the bin size.
            binMin = min(self.featureLists[i])
            binMax = max(self.featureLists[i])
            binSize = (binMax - binMin) / float(numBins)
            #create the specified number of bins.
            for j in range(numBins):
                self.binLists[i].append(binMin+(j*binSize))
            #put the end on manually, since small errors can arise when adding floats.
            self.binLists[i].append(binMax)

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

    def __informationGain(self, featureList, featureIndex, classLabelIndexList):
        binList = self.binLists[featureIndex]
        #rebuild the classLabelList from the indexes
        classLabelList = list()
        for i in classLabelIndexList:
            classLabelList.append(self.classLabelList[i])


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

    def __ID3(self, examplesIndexList, unbranchedFeatures, root):

        num1 = 0
        num0 = 0
        #count the instances of 0's and 1's to check for uniformity.
        for i in examplesIndexList:
            if self.classLabelList[i] == 0:
                num0+=1
            elif self.classLabelList[i] == 1:
                num1+=1

        #if we have a uniform set, we're done! label the node.
        if num0==len(examplesIndexList):
            root.label = 0
            return root
        if num1==len(examplesIndexList):
            root.label = 1
            return root

        #if we're out of features to branch on, return the most common label (with 0 breaking ties)

        if len(unbranchedFeatures) == 0:
            if num0>=num1:
                root.label = 0
                return root
            elif num1>num0:
                root.label = 1
                return root
        #begin the algorithm!
        maxInfoGain = -1
        #for each unbranched feature, get the information gain for that feature.
        for i in unbranchedFeatures:
            infoGain = self.__informationGain(self.featureLists[i], i, examplesIndexList)
            #if we found a larger info gain, set the root to that index and note the new max.
            if infoGain > maxInfoGain:
                root.featureIndex = i
                maxInfoGain = infoGain
        #create a new UnbranchedFeatures list where we have branched on the new feature.
        newUnbranchedFeatures = copy.deepcopy(unbranchedFeatures)
        newUnbranchedFeatures.remove(root.featureIndex)
        root.children = list()
        #for each bin for the best feature, 
        for j in range(len(self.binLists[root.featureIndex])-1):
            #create a child node with the bin's min and max values.
            root.children.append(Node())
            root.children[j].featureMin=self.binLists[root.featureIndex][j]
            root.children[j].featureMax=self.binLists[root.featureIndex][j+1]
            #root.children[j].printNode()
            #print()
        #for each child
        for child in root.children:
            #create a new examples index list
            newExamplesIndexList = list()
            #for each example left in the dataset
            for exampleIndex in examplesIndexList:
                #if the example value fits the bin, add it into the new examples list.
                if self.featureLists[root.featureIndex][exampleIndex] >= child.featureMin and self.featureLists[root.featureIndex][exampleIndex] <= child.featureMax:
                    newExamplesIndexList.append(exampleIndex)
            #print newExamplesIndexList
            #if the new example list is empty, then we just make a leaf with the most common feature in the examples. we already found these values earlier.
            if len(newExamplesIndexList) == 0:
                if num0>=num1:
                    child.label = 0
                elif num1>num0:
                    child.label = 1
            #otherwise, we recursively call ID3 and keep on going.
            else:
                self.__ID3(newExamplesIndexList, newUnbranchedFeatures, child)

        return root

    #this function in particular is very specific to testing this project, although the rest of the code might be possible to modify for a similar case.
    def testAgainstSelf(self):
        #we've already parsed all the data from the file, so we can reuse it here and run through the tree.
        successes = 0.0
        #for each row and feature,
        for i in range(len(self.classLabelList)):
            features = list()
            classValue = self.classLabelList[i]
            for j in range(len(self.featureLists)):
                #pivot our parallel lists from the 2d array into a 1d array.
                features.append(self.featureLists[j][i])
            #traverse the tree to find the result. 
            successes += self.__isTreeCorrect(self.rootTreeNode, features, classValue)

        return 1 - (successes/len(self.classLabelList))

    def __isTreeCorrect(self, node, features, classValue):
        #if the node is labeled, we're done.
        if node.label != None:
            if classValue == node.label:
                return 1
            else:
                return 0

        #if the node has a feature index, we split on that.
        if node.featureIndex != None:
            featureValue = float(features[node.featureIndex])
            #for each child, find the correct bin.
            for child in node.children:
                if featureValue >= child.featureMin and featureValue <= child.featureMax:
                    return self.__isTreeCorrect(child, features, classValue)




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
            self.unbranchedFeatures.add(i)
        file.seek(fileBeginning)

        #for each line in the file, parse the features and class labels into parallel lists.
        for line in file:
            parsedLine = line.split(',')
            for i in range(len(self.featureLists)):
                self.featureLists[i].append(float(parsedLine[i]))
            classLabel = int(parsedLine[len(parsedLine)-1].rstrip())
            self.classLabelList.append(classLabel)
            self.distinctClassLabels.add(classLabel)

        self.__discretizeFeaturesEquidistant(numBins)
        self.__debugPrint()
        #self.printList()
        #self.__informationGain(self.featureLists[0], 0, self.classLabelList)

        #make a list of the example indexes to start ID3 (all of them)
        examplesIndexList = list()
        for i in range(len(self.classLabelList)):
            examplesIndexList.append(i)
        #set a root node
        self.rootTreeNode = Node()
        #kick off ID3!
        self.__ID3(examplesIndexList, self.unbranchedFeatures, self.rootTreeNode)
        #print("entropy",self.__entropy(self.distinctClassLabels, self.classLabelList))

    def printList(self):
        for j in range(len(self.featureLists[0])):
            print str(self.featureLists[0][j]), str(self.featureLists[1][j]),
            print self.classLabelList[j]

    def printTree(self, root, depth):
        root.printNode()
        print()
        depth+=1
        if root.children != None:
            for child in root.children:     
                print depth*'---',
                #child.printNode()
                #print child.children
                self.printTree(child, depth)
        



def main():
    if (len(sys.argv) != 2):
        print "Takes 1 command line argument: the name of the csv file."
        exit(-1)
    filename = sys.argv[1]
    #initialize the TestData object
    isDebugMode = False
    numBins = 10
    #initialize tree
    treeObj = ID3Tree(filename, numBins, isDebugMode)
    #print the tree. spoilers: its REALLY ugly
    treeObj.printTree(treeObj.rootTreeNode, 0)
    #test it against the training data
    error = treeObj.testAgainstSelf()

    print "error: " + str(error)

main()