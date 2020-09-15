import sys
import matplotlib.pyplot as p
import numpy as np
from math import log, sqrt
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


T = np.array([[1,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0],
              [0,1,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,0],
              [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,1,0],
              [0,0,0,0, 0,0,0,0, 0,1,0,0, 0,0,0,1]])



numMultiplies = 4


def customMatrixMultMaker(WA, WB, WFinal):
    def customMatrixMult(exampleA, exampleB):
        return np.reshape(np.dot(WFinal, np.multiply(np.dot(WA, exampleA.flatten()), np.dot(WB, exampleB.flatten()))), (matrixSize, matrixSize))
    return customMatrixMult

def multiplyOut(a, b):
    returnMat = []

    for i in range(len(a)):
        returnMat.append([])

        for j in range(len(a[i])):
            for k in range(len(b[i])):
                returnMat[-1].append(a[i][j] * b[i][k])

    return np.array(returnMat)

def multiplyOutPrime(a, b, error):
    AErrorMat = np.zeros((numMultiplies, matrixSize*matrixSize))
    BErrorMat = np.zeros((numMultiplies, matrixSize*matrixSize))

    for i in range(len(a)):
        for j in range(len(a[i])):
            for k in range(len(b[i])):
                AErrorMat[i][j] += error[i][matrixSize*matrixSize*j + k] * b[i][k]
                BErrorMat[i][k] += error[i][matrixSize*matrixSize*j + k] * a[i][j]

    return AErrorMat, BErrorMat


#print(X)
#print(T)

#print(np.dot(X[0][:4].reshape(2,2), X[0][4:].reshape(2,2)))
#print(T[0].reshape(2,2))

#W = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), T)


numIter = 100000
learningRate = 3e-4
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

errors = []
logErrors = []

#WA = strassenWA.astype(float)
#WB = strassenWB.astype(float)
#WFinal = strassenWFinal.astype(float)

noise = 0.1

#WA = strassenWA.astype(float) + np.random.normal(0, noise, size=(numMultiplies, matrixSize*matrixSize))
#WB = strassenWB.astype(float) + np.random.normal(0, noise, size=(numMultiplies, matrixSize*matrixSize))
#WFinal = strassenWFinal.astype(float) + np.random.normal(0, noise, size=(matrixSize*matrixSize, numMultiplies))

WA = np.random.normal(size=(numMultiplies, matrixSize*matrixSize))
WB = np.random.normal(size=(numMultiplies, matrixSize*matrixSize))
WFinal = np.random.normal(size=(matrixSize*matrixSize, numMultiplies))

#WA, WB, WFinal = pickle.load(open("strassens_5.p", "rb"))

print(WA)
print(WB)
print(WFinal)

vA = np.ones(WA.shape)
vB = np.ones(WB.shape)
vFinal = np.ones(WFinal.shape)

mA = np.zeros(WA.shape)
mB = np.zeros(WB.shape)
mFinal = np.zeros(WFinal.shape)

for iteration in range(numIter):

    # FEEDFORWARD STAGE

#    print(WA.shape)
#    print(As.shape)

    multOutput = multiplyOut(WA, WB)
    finalOutput = np.dot(WFinal, multOutput)

#    print(np.linalg.norm(T - finalOutput))

    # BACKPROP STAGE

    outputErrorWFinal = finalOutput - T

    AErrorMat, BErrorMat = multiplyOutPrime(WA, WB, np.dot(np.transpose(WFinal), outputErrorWFinal))

    outputErrorWA = AErrorMat #* WA
    outputErrorWB = BErrorMat #* WB

#   actFinal = finalOutputs
#   actMid = finalInputs

#    gradientsA = np.transpose(np.dot(As, np.transpose(outputErrorWA)))
#    gradientsB = np.transpose(np.dot(Bs, np.transpose(outputErrorWB)))
#    gradientsFinal = np.transpose(np.dot(finalInputs, np.transpose(outputErrorWFinal)))

    gradientsA = outputErrorWA
    gradientsB = outputErrorWB
    gradientsFinal = np.transpose(np.dot(multOutput, np.transpose(outputErrorWFinal)))

    mA = beta1 * mA + (1 - beta1) * gradientsA
    mB = beta1 * mB + (1 - beta1) * gradientsB
    mFinal = beta1 * mFinal + (1 - beta1) * gradientsFinal

    vA = beta2 * vA + (1 - beta2) * gradientsA**2
    vB = beta2 * vB + (1 - beta2) * gradientsB**2
    vFinal = beta2 * vFinal + (1 - beta2) * gradientsFinal**2


    WA -= learningRate * mA / np.sqrt(vA + eps)
    WB -= learningRate * mB / np.sqrt(vB + eps)
    WFinal -= learningRate * mFinal / np.sqrt(vFinal + eps)

#    WA -= gradientsA * learningRate
#    WB -= gradientsB * learningRate
#    WFinal -= gradientsFinal * learningRate

#    print(WA, WB, WFinal)

    error = np.linalg.norm(finalOutput - T)

#    print(finalOutput[0][0])

    errors.append(error)
    logErrors.append(softLog(error))

#    learningRate *= 0.99

print(error)

#print(WA)
#print(WB)
#print(WFinal)

print(finalOutput)
p.matshow(finalOutput)
p.colorbar()
p.show()

exampleA = np.random.normal(size=(matrixSize,matrixSize))
exampleB = np.random.normal(size=(matrixSize,matrixSize))

exampleC1 = np.dot(exampleA, exampleB)
exampleC2 = customMatrixMultMaker(WA, WB, WFinal)(exampleA, exampleB)

print(exampleC1) 
print(exampleC2)


print(customMatrixMultMaker(strassenWA, strassenWB, strassenWFinal)(exampleA, exampleB))

#pickle.dump((WA, WB, WFinal), open("strassens_5_refined.p", "wb"))

p.plot(logErrors)
p.show()



#print(W)

#print(X.shape)

#print(T - np.dot(X, W))

#W = 