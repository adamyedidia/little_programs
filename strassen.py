import matplotlib.pyplot as p
import numpy as np
from math import log
import pickle

matrixSize = 2
numExamples = 10000


def softLog(x):
    if x <= 0:
        return -75
    else:
        return log(x)


listOfAs = [np.random.normal(size=(matrixSize, matrixSize)) for _ in range(numExamples)]
listOfBs = [np.random.normal(size=(matrixSize, matrixSize)) for _ in range(numExamples)]

listOfCs = [np.dot(A, B) for A, B in zip(listOfAs, listOfBs)]

X = []
T = []
As = []
Bs = []

for i in range(numExamples):
    A = listOfAs[i]
    B = listOfBs[i]

    As.append(A.flatten())
    Bs.append(B.flatten())

    ex = np.concatenate([A.flatten(), B.flatten()], 0)

    X.append(ex)

    C = listOfCs[i]
    tar = C.flatten()

    T.append(tar)

X = np.array(X).transpose()
T = np.array(T).transpose()
As = np.array(As).transpose()
Bs = np.array(Bs).transpose()


strassenWA = np.array([[1,0,0,1],
                       [0,0,1,1],
                       [1,0,0,0],
                       [0,0,0,1],
                       [1,1,0,0],
                       [-1,0,1,0],
                       [0,1,0,-1]])

strassenWB = np.array([[1,0,0,1],
                       [1,0,0,0],
                       [0,1,0,-1],
                       [-1,0,1,0],                       
                       [0,0,0,1],
                       [1,1,0,0],
                       [0,0,1,1]])

strassenWFinal = np.array([[1,0,0,1,-1,0,1],
                           [0,0,1,0,1,0,0],
                           [0,1,0,1,0,0,0],
                           [1,-1,1,0,0,1,0]])




def customMatrixMultMaker(WA, WB, WFinal):
    def customMatrixMult(exampleA, exampleB):
        return np.reshape(np.dot(WFinal, np.multiply(np.dot(WA, exampleA.flatten()), np.dot(WB, exampleB.flatten()))), (matrixSize, matrixSize))
    return customMatrixMult

#print(X)
#print(T)

#print(np.dot(X[0][:4].reshape(2,2), X[0][4:].reshape(2,2)))
#print(T[0].reshape(2,2))

#W = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), T)

numMultiplies = 6

WA = np.random.normal(size=(numMultiplies, matrixSize*matrixSize))
WB = np.random.normal(size=(numMultiplies, matrixSize*matrixSize))
WFinal = np.random.normal(size=(matrixSize*matrixSize, numMultiplies))

noise = 0.1

#WA = strassenWA.astype(float) + np.random.normal(0, noise, size=(numMultiplies, matrixSize*matrixSize))
#WB = strassenWB.astype(float) + np.random.normal(0, noise, size=(numMultiplies, matrixSize*matrixSize))
#WFinal = strassenWFinal.astype(float) + np.random.normal(0, noise, size=(matrixSize*matrixSize, numMultiplies))

numIter = 200000
learningRate = 1e-6

errors = []
logErrors = []

for iteration in range(numIter):

    # FEEDFORWARD STAGE

#    print(WA.shape)
#    print(As.shape)

    AOutputs = np.dot(WA, As)
    BOutputs = np.dot(WB, Bs)

    finalInputs = np.multiply(AOutputs, BOutputs)
    finalOutputs = np.dot(WFinal, finalInputs)

    # BACKPROP STAGE

    outputErrorWFinal = finalOutputs - T

    outputErrorWA = np.dot(np.transpose(WFinal), outputErrorWFinal) * BOutputs
    outputErrorWB = np.dot(np.transpose(WFinal), outputErrorWFinal) * AOutputs

#   actFinal = finalOutputs
#   actMid = finalInputs

    gradientsA = np.transpose(np.dot(As, np.transpose(outputErrorWA)))
    gradientsB = np.transpose(np.dot(Bs, np.transpose(outputErrorWB)))
    gradientsFinal = np.transpose(np.dot(finalInputs, np.transpose(outputErrorWFinal)))


    WA -= gradientsA * learningRate
    WB -= gradientsB * learningRate
    WFinal -= gradientsFinal * learningRate

#    print(WA, WB, WFinal)

    error = np.linalg.norm(finalOutputs - T)

    errors.append(error)
    logErrors.append(softLog(error))

print(error)

print(WA)
print(WB)
print(WFinal)

exampleA = np.random.normal(size=(matrixSize,matrixSize))
exampleB = np.random.normal(size=(matrixSize,matrixSize))

exampleC1 = np.dot(exampleA, exampleB)
exampleC2 = customMatrixMultMaker(WA, WB, WFinal)(exampleA, exampleB)

print(exampleC1) 
print(exampleC2)


print(customMatrixMultMaker(strassenWA, strassenWB, strassenWFinal)(exampleA, exampleB))

pickle.dump((strassenWA, strassenWB, strassenWFinal), open("strassens.p", "wb"))

p.plot(logErrors)
p.show()



#print(W)

#print(X.shape)

#print(T - np.dot(X, W))

#W = 