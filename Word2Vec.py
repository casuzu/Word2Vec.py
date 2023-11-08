# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 21:40:11 2023

@author: j.casuzu
"""
#The dog saw a cat. The dog chased the cat. The cat climbed a tree.
#the dog saw a cat chased climbed tree
#the ->     [00000001]
#dog ->     [00000010]
#saw ->     [00000011]
# a  ->     [00000100]
#cat ->     [00000101]
#chased ->  [00000110]
#climbed -> [00000111]
#tree ->    [00001000]

#SkipGram
#windowSize = 3

import numpy as np
import matplotlib.pyplot as plt

sentence = ["the", "dog", "saw", "a", "cat",
            "the", "dog", "chased","the", "cat",
            "the", "cat", "climbed", "a", "tree"]

sentencedict = {
 "the":[0, 0, 0, 0, 0, 0, 0, 1],
  "dog":[0, 0, 0, 0, 0, 0, 1, 0],
  "saw":[0, 0, 0, 0, 0, 0, 1, 1],
   "a": [0, 0, 0, 0, 0, 1, 0, 0],
  "cat":[0,0, 0, 0, 0, 1, 0, 1],
  "chased":[0, 0, 0, 0, 0, 1, 1, 0],
  "climbed":[0, 0, 0, 0, 0, 1, 1, 1],
  "tree":[0,0,0,0,1,0,0,0]
  }

windowSize = 3
window = []

def WindowDict_Getter(currentIndex):
    currentWordkey = sentence[currentIndex]
    
    if(currentIndex == (len(sentence)-1)):
        relateWord1key = sentence[currentIndex -1]
        relateWord2key = sentence[currentIndex - 2]
            
    elif(currentIndex > 0):
        relateWord1key = sentence[currentIndex -1]
        
        relateWord2key = sentence[currentIndex + 1]
        
    elif(currentIndex == 0):
        relateWord1key = sentence[currentIndex + 1]
        relateWord2key = sentence[currentIndex + 2]
     
    return sentencedict[relateWord1key], sentencedict[currentWordkey], sentencedict[relateWord2key]


def Soft_Max(matrix):
    outMatrix = np.zeros(matrix.size).reshape(matrix.shape)
    numeratorMatrix = np.exp(matrix)
    denominator = np.sum(numeratorMatrix)
    
    for i, element in enumerate(numeratorMatrix):
        outMatrix[i] = (element/denominator)
    return outMatrix
    
def Generate_Weights(row, col):
    weights = np.random.rand(row * col).reshape(row, col)    
    return weights     
 
    
def Feed_Forward(inputMatrix, weights):
    hiddenLayerMatrix = np.matmul(inputMatrix, weights)
    return hiddenLayerMatrix


def Train(inputMat, inputWeights, word1Mat, word1_Weights, word2Mat, word2_Weights):
    #Feed-Forward 
    hiddenL_Mat = Feed_Forward(inputMat, inputWeights)
    word1_outMat = Feed_Forward(hiddenL_Mat, word1_Weights)
    word2_outMat = Feed_Forward(hiddenL_Mat, word2_Weights)
    
    word1_outMat = Soft_Max(word1_outMat)
    word2_outMat = Soft_Max(word2_outMat)
    
    word1_Target = Soft_Max(word1Mat)
    word2_Target = Soft_Max(word2Mat)
    
    word1_Error = Cost_Function(word1_outMat, word1_Target)
    word2_Error = Cost_Function(word2_outMat, word2_Target)
    
    Error = 0.5 * (word1_Error + word2_Error)
    
    inputWeights, word1_Weights,  word2_Weights = BackPropagation(inputMat, inputWeights, hiddenL_Mat, word1_Weights, word1_outMat, word1_Target, word2_Weights, word2_outMat, word2_Target)
    return Error, inputWeights, word1_Weights,  word2_Weights, word1_outMat, word2_outMat

def Cost_Function(outputMatrix, correctMatrix):
    #Cost function is MSE(Mean Square Error)
    summationMatrix= np.square(outputMatrix - correctMatrix)
    MSE = np.sum(summationMatrix)
    return MSE



def BackPropagation(inputMat, inputWeights, hiddenL_Mat, word1_Weights, word1_outMat, word1_Target, word2_Weights, word2_outMat, word2_Target):

    dCtotal_dW1 = 2 * np.matmul(np.transpose(hiddenL_Mat), (word1_outMat - word1_Target)) 
    dCtotal_dW2 = 2 * np.matmul(np.transpose(hiddenL_Mat), (word2_outMat - word2_Target)) 
    
    temp1 = np.matmul(word1_Weights, np.transpose(word1_outMat - word1_Target))
    temp2 = np.matmul(word2_Weights, np.transpose(word2_outMat - word2_Target))
    temp3 = temp1 + temp2
    dCtotal_dW0 = 2 * np.matmul(temp3, inputMat)
    
    learning_rate = 0.00001
    
    newWord1_Weights = word1_Weights - (learning_rate * dCtotal_dW1)
    newWord2_Weights = word1_Weights - (learning_rate * dCtotal_dW2)
    newInputWeights = inputWeights - np.transpose(learning_rate * dCtotal_dW0)
    
    return newInputWeights, newWord1_Weights, newWord2_Weights
      

def Epoch_Train():
    inputWeights = Generate_Weights(8, 3) 
    word1_Weights = Generate_Weights(3, 8)
    word2_Weights = Generate_Weights(3, 8)

    best_inputWeights = 0
    best_word1Weights = 0
    best_word2Weights = 0
    
    errorList = []
    leasterror = 10 
    epoch = 100000
    errorSkip = epoch/10
    
    for i in range(0, len(sentence)):        
        #Get the words matrices related to the current word  
        relateW1_Mat, currentW_Mat, relateW2_Mat = WindowDict_Getter(i)#related word 1 matrix and so on.
        #print(currentW_Mat, " ", relateW1_Mat, " ", relateW2_Mat)
        
       
        for j in range(0, epoch):
            inputMat = np.array(currentW_Mat).reshape(1, 8)
            word1Mat = np.array(relateW1_Mat).reshape(1, 8)
            word2Mat = np.array(relateW2_Mat).reshape(1, 8)
            
            Error, inputWeights, word1_Weights, word2_Weights, word1_outMat, word2_outMat = Train(inputMat, inputWeights, word1Mat, word1_Weights, word2Mat, word2_Weights)
            
            if(Error < leasterror):
                leasterror = Error
                best_inputWeights = inputWeights
                best_word1Weights = word1_Weights
                best_word2Weights = word2_Weights
                
            if((j % errorSkip) == 0):
                errorList.append(Error)
                
    errorList = np.array(errorList)
    maxerror = np.max(errorList)
    minerror = np.min(errorList)
    
    z = int(len(sentence)*(epoch/errorSkip))
    y = errorList
    plot(minerror,maxerror, y, z)
    
    print("Min Error =", minerror)
    print("Max Error =", maxerror)
    print("Least Error =", leasterror)
    print("best_inputWeights\n", best_inputWeights)
    print("best_word1Weights\n", best_word1Weights)
    print("best_word2Weights\n", best_word2Weights)
    
    return  leasterror, best_inputWeights, best_word1Weights, best_word2Weights

def Test(inputword, best_inputWeights, best_word1Weights, best_word2Weights):
    inputMat = sentencedict[inputword]
    inputMat = np.array(inputMat).reshape(1, 8)
    
    hiddenMat = Feed_Forward(inputMat, best_inputWeights)
    
    relateword1_Mat = Feed_Forward(hiddenMat, best_word1Weights)
    relateword2_Mat = Feed_Forward(hiddenMat, best_word2Weights)
    
    outword1_Mat = Soft_Max(relateword1_Mat)
    outword2_Mat = Soft_Max(relateword2_Mat)
    
    
    return outword1_Mat, outword2_Mat
  
    
def plot(minerror, maxerror, y, z):
    x = np.linspace(minerror, maxerror, z)
    plt.plot(x, y)
    plt.show()


inputword = "cat"
leasterror, best_inputWeights, best_word1Weights, best_word2Weights = Epoch_Train()
outword1_Mat, outword2_Mat = Test(inputword, best_inputWeights, best_word1Weights, best_word2Weights)
print("\noutword1_Mat\n", outword1_Mat)
print("Word 1 Probabilities")
print("-----------------")
print("the: ", (outword1_Mat[0][7]) * 100)
print("dog: ", (outword1_Mat[0][6]) * 100)
print("saw: ", (outword1_Mat[0][6] + outword1_Mat[0][7]) * 100)
print("a: ", (outword1_Mat[0][5]) * 100)
print("cat: ", (outword1_Mat[0][5] + outword1_Mat[0][7]) * 100)
print("chased: ", (outword1_Mat[0][5] + outword1_Mat[0][6]) * 100)
print("climbed: ", (outword1_Mat[0][5] + outword1_Mat[0][6] + outword1_Mat[0][7]) * 100)
print("tree: ", (outword1_Mat[0][4]) * 100)

print("\noutword2_Mat\n", outword2_Mat)
print("Word 2 Probabilities")
print("-----------------")
print("the: ", (outword2_Mat[0][7]) * 100)
print("dog: ", (outword2_Mat[0][6]) * 100)
print("saw: ", (outword2_Mat[0][6] + outword2_Mat[0][7]) * 100)
print("a: ", (outword2_Mat[0][5]) * 100)
print("cat: ", (outword2_Mat[0][5] + outword2_Mat[0][7]) * 100)
print("chased: ", (outword2_Mat[0][5] + outword2_Mat[0][6]) * 100)
print("climbed: ", (outword2_Mat[0][5] + outword2_Mat[0][6] + outword2_Mat[0][7]) * 100)
print("tree: ", (outword2_Mat[0][4]) * 100)











