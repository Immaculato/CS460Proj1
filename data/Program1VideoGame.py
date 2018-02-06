#Tristan Basil
#Assignment: Project 1 - cS460G Machine Learning, Dr. Harrison
#https://stackoverflow.com/questions/32796531/how-to-get-the-most-common-element-from-a-list-in-python


import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import copy
import pdb
from collections import Counter


class Node:
    children = None
    label = None
    featureIndex = None
    featureMax = None
    featureMin = None
    featureString = None

    def printNode(self):
        print "featureIndex:", self.featureIndex, "featureString:", self.featureString, "featureMin/Max:", self.featureMin, self.featureMax, "label:", self.label

#this class is only designed to work for the data in this project.
class ID3Tree:
    debug = False
    maxDepth = 0
    featureLists = list()
    #declare how each data type should be handled.
    featureTypes = ['s', 's', 's', 's', 'f', 'f', 'f', 'f', 'f', 's', 's']
    binLists = list()
    classLabelList = list()
    distinctClassLabels = set()
    #contains bins for the class labels.
    distinctClassBins = list([(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)])
    unbranchedFeatures = set()
    rootTreeNode = Node()

    def __debugPrint(self):
        if self.debug:
            print " --- DEBUG INFO --- "
            print "class labels: " + str(self.distinctClassLabels)
            print "class bins: " + str(self.distinctClassBins)
            for i in range(len(self.featureLists)):
                if self.featureTypes[i] == 'f':
                    print "max feature " + str(i) + ": " + str(max(self.featureLists[i]))
                    print "min feature " + str(i) + ": " + str(min(self.featureLists[i]))
                    print "discretized bins feature " + str(i) + ":",
                    for j in range(len(self.binLists[i])):
                        print(self.binLists[i][j]),
                        print "-",
                    print ""
                elif self.featureTypes[i] == 's':
                    print "string values feature " + str(i) +": " + str(min(self.featureLists[i])) + " ... " + str(max(self.featureLists[i]))
            print " --- END DEBUG --- "

    def __discretizeFeaturesEquidistant(self, numBins):
        #for each feature
        for i in range(len(self.featureTypes)):
            #if we are dealing with a float,
            if self.featureTypes[i] == 'f':
                #find the max and min of the feature, and use this to calculate the bin size.
                binMin = min(self.featureLists[i])
                binMax = max(self.featureLists[i])
                binSize = (binMax - binMin) / float(numBins)
                #create the specified number of bins.
                for j in range(numBins):
                    self.binLists[i].append(binMin+(j*binSize))
                #put the end on manually, since small errors can arise when adding floats.
                self.binLists[i].append(binMax)
            #if we're dealing with a string, just put the distinct string values in the bin.
            elif self.featureTypes[i] == 's':
                #for each string value,
                for stringValue in self.featureLists[i]:
                    #if the string isn't already in the list,
                    if stringValue not in self.binLists[i]:
                        #put it in there.
                        self.binLists[i].append(stringValue)


    def __entropy(self, distinctClassBins, classLabelList):

        entropyTot = 0.0
        indexesUsed = list()
        #for each class label bin in the list
        for i in distinctClassBins:
            #contains the number of instances for each class label bin
            numEquals=0.0
            #mark the number of values for each class label.
            for k in range(len(classLabelList)):
                if k not in indexesUsed and self.__valueInBin(int(classLabelList[k]), i):
                    numEquals+=1
                    indexesUsed.append(k)
            #print(i, numEquals)
            #add to the total entropy for each bin (making sure it can be evaluated)
            if (numEquals == 0 or len(classLabelList) == 0):
                entropyTot+=0
            else:
                #print(numEquals)
                ratioInBin = (numEquals/len(classLabelList))
                entropyTot += -(ratioInBin*math.log(ratioInBin, 2))
        #print 'entropy', entropyTot
        return entropyTot

    def __informationGain(self, featureList, featureIndex, classLabelIndexList):
        binList = self.binLists[featureIndex]
        #rebuild the classLabelList from the indexes
        classLabelList = list()
        for i in classLabelIndexList:
            classLabelList.append(self.classLabelList[i])

        informationGainTot = self.__entropy(self.distinctClassBins, classLabelList)
        #print'-----'
        #print 'before',informationGainTot
        #print 'featureIndex', featureIndex
        #count the instances in each bin
        #2d list containing indexes of contents for each bin
        binContents = list()
        #for each bin (there are 1 less than the total number of entries)
        for i in range(len(binList)):
            numInBin = 0.0
            #for each row on the current feature, see if it fits in the current bin.
            for k in classLabelIndexList:
                #if we're dealing with floats, we need to fit into the discretized bins
                if self.featureTypes[featureIndex] == 'f':
                    binContents.append(list())
                    if (i < len(binList)-1 and featureList[k] >= binList[i] and featureList[k] <= binList[i+1]):
                        numInBin+=1
                        #add its class label to the bin contents
                        binContents[i].append(self.classLabelList[k])
                
                #if we're dealing with strings, we just need to match a string to fit in the 'bin'.
                elif self.featureTypes[featureIndex] == 's':
                    binContents.append(list())
                    if (featureList[k] == binList[i]):
                        numInBin+=1
                        binContents[i].append(self.classLabelList[k])

            #continue calculating information gain
            informationGainTot -= (numInBin/len(classLabelList))*self.__entropy(self.distinctClassBins, binContents[i])
            #print informationGainTot

        return informationGainTot

    def __ID3(self, examplesIndexList, unbranchedFeatures, root, depth):
        #create a new dictionary, keyed on the class bins.
        classLabelInstances = dict()
        for i in self.distinctClassBins:
            classLabelInstances[i] = 0

        #print len(classLabelInstances)
        #count the instances of each class label to check for uniformity.
        #for each class label index, and distinct class bin,
        for i in examplesIndexList:
            valueSorted = False
            for j in self.distinctClassBins:
                #see if we have a match. if we do, mark it.
                if not valueSorted and self.__valueInBin(self.classLabelList[i], j):
                    classLabelInstances[j]+=1
                    valueSorted=True
        
        #if every example has the same class label, we're done! label the node.
        for j in self.distinctClassBins:
            if classLabelInstances[j]==len(examplesIndexList):
                root.label=j
                return root

        
        #if we're out of features to branch on, or at the maximum depth, return the most common label (with 0 breaking ties)
        if len(unbranchedFeatures) == 0 or depth == self.maxDepth:
            countLabels = Counter(classLabelInstances)
            root.label = countLabels.most_common(1)[0][0]
            return root

        #begin the algorithm! we're going down the tree now.
        depth += 1
        maxInfoGain = -1
        #for each unbranched feature, get the information gain for that feature.
        for i in unbranchedFeatures:
            infoGain = self.__informationGain(self.featureLists[i], i, examplesIndexList)
            #print 'infogain',infoGain
            #if we found a larger info gain, set the root to that index and note the new max.
            if infoGain > maxInfoGain:
                root.featureIndex = i
                maxInfoGain = infoGain
        #create a new UnbranchedFeatures list where we have branched on the new feature.
        #print root.featureIndex, maxInfoGain
        #print 'unbranchedFeatures', unbranchedFeatures
        newUnbranchedFeatures = copy.deepcopy(unbranchedFeatures)
        newUnbranchedFeatures.remove(root.featureIndex)
        root.children = list()
        #for each bin for the best feature, 
        for j in range(len(self.binLists[root.featureIndex])):
            #if it's a number feature,
            if self.featureTypes[root.featureIndex] == 'd':
                #create a child node with the bin's min and max values.
                if j < len(self.binLists[root.featureIndex])-1:
                    root.children.append(Node())
                    root.children[j].featureMin=self.binLists[root.featureIndex][j]
                    root.children[j].featureMax=self.binLists[root.featureIndex][j+1]
            #if it's a string, just change the feature for a string.
            elif self.featureTypes[root.featureIndex] == 's':
                root.children.append(Node())
                root.children[j].featureString=self.binLists[root.featureIndex][j]

            #root.children[j].printNode()
            #print()
        #for each child
        for child in root.children:
            #create a new examples index list
            newExamplesIndexList = list()
            #for each example left in the dataset
            for exampleIndex in examplesIndexList:
                #if the example value fits the bin, add it into the new examples list.
                if self.featureTypes[root.featureIndex] == 'd':
                    if self.featureLists[root.featureIndex][exampleIndex] >= child.featureMin and self.featureLists[root.featureIndex][exampleIndex] <= child.featureMax:
                        newExamplesIndexList.append(exampleIndex)
                elif self.featureTypes[root.featureIndex] == 's':
                    if self.featureLists[root.featureIndex][exampleIndex] == child.featureString:
                        newExamplesIndexList.append(exampleIndex)
            #print newExamplesIndexList
            #if the new example list is empty, then we just make a leaf with the most common feature in the examples.
            if len(newExamplesIndexList) == 0:
                countLabels = Counter(classLabelInstances)
                #print countLabels.most_common(1)[0][0]
                child.label = countLabels.most_common(1)[0][0]
            #otherwise, we recursively call ID3 and keep on going.
            else:
                #print newUnbranchedFeatures
                self.__ID3(newExamplesIndexList, newUnbranchedFeatures, child, depth)

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
            resultBin = self.__isTreeCorrect(self.rootTreeNode, features)
            if self.__valueInBin(classValue, resultBin):
                successes += 1

        return 1 - (successes/len(self.classLabelList))

    def __isTreeCorrect(self, node, features):
        #if the node is labeled, we're done.
        if node.label != None:
            return node.label

        #if the node has a feature index, we split on that.
        if node.featureIndex != None:
            featureValue = features[node.featureIndex]
            #for each child, find the correct bin.
            for child in node.children:
                if self.featureTypes[node.featureIndex] == 'f':
                    if featureValue >= child.featureMin and featureValue <= child.featureMax:
                        return self.__isTreeCorrect(child, features)
                elif self.featureTypes[node.featureIndex] == 's':
                    if featureValue == child.featureString:
                        return self.__isTreeCorrect(child, features)

    #helper function to determine if value is in a distinctClassBin
    def __valueInBin(self, value, bin):
        if value>=bin[0] and value<=bin[1]:
            return True
        else:
            return False


    #initialization takes a filename.
    def __init__(self, filename, numBins, maxDepth, debug):
        self.maxDepth = maxDepth
        self.debug = debug
        file = None
        try:
            file = open(filename, "r")
        except:
            invalidNumFields = True
            print(filename, 'not found')
            exit -1
        #quickly get the number of features from the file, and initialize the lists.
        numFeatures = len(file.readline().split(',')) - 1
        for i in range(numFeatures):
            self.featureLists.append(list())
            self.binLists.append(list())
            self.unbranchedFeatures.add(i)

        #for each line in the file, parse the features and class labels into parallel lists.
        for line in file:
            #data can have commas inside quotes. change them to a back slash to make parsing simpler
            quoteToBeFound = True
            searchedIndexesTo = 0
            #while there's still quotes being found in the line,
            while quoteToBeFound:
                quoteIndex = line.find('\"', searchedIndexesTo)
                #look for a quote. if we found one,
                if (quoteIndex != -1):
                    #try to find a comma in the pair of quotes.
                    nextQuote = line.find('\"', quoteIndex+1)
                    nextComma = line.find(',', quoteIndex+1)
                    #if we found a comma, and it comes before the next quote, change it to a front slash and keep looking.
                    if nextComma != -1 and nextComma < nextQuote:
                        linelist = list(line)
                        linelist[nextComma] = '/'
                        line = "".join(linelist)
                    #otherwise, we're done with this pair of quotes. we can move on to the next one (if it exists).
                    else:
                        searchedIndexesTo = nextQuote+1
                #not finding a quote anymore means this line is cleaned and good to go.     
                else:
                    quoteToBeFound = False

            parsedLine = line.split(',')
            for i in range(len(self.featureLists)):
                if self.featureTypes[i] == 's':
                    if parsedLine[i] == '':
                        #N/A is a little cleaner than just empty strings
                        self.featureLists[i].append('N/A')
                    else:
                        self.featureLists[i].append(parsedLine[i])
                elif self.featureTypes[i] == 'f':
                    self.featureLists[i].append(float(parsedLine[i]))
            classLabel = int(parsedLine[len(parsedLine)-1].rstrip())
            self.classLabelList.append(classLabel)
            self.distinctClassLabels.add(classLabel)

        self.__discretizeFeaturesEquidistant(numBins)
        self.__debugPrint()

        #make a list of the example indexes to start ID3 (all of them)
        examplesIndexList = list()
        for i in range(len(self.classLabelList)):
            examplesIndexList.append(i)
        #print self.__informationGain(self.featureLists[0], 0, examplesIndexList)
        #set a root node
        self.rootTreeNode = Node()
        #kick off ID3!
        self.__ID3(examplesIndexList, self.unbranchedFeatures, self.rootTreeNode, 0)
        #print("entropy",self.__entropy(self.distinctClassLabels, self.classLabelList))


    def printTree(self, root, depth):
        root.printNode()
        if root.label == None and root.featureIndex == None:
            print '*************************************RED ALERT*********************************************'
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
    isDebugMode = True
    numBins = 10
    maxDepth = 2
    #initialize tree
    treeObj = ID3Tree(filename, numBins, maxDepth, isDebugMode)
    #print the tree. spoilers: its REALLY ugly
    treeObj.printTree(treeObj.rootTreeNode, 0)
    #test it against the training data
    error = treeObj.testAgainstSelf()
    print "error: " + str(error)



main()