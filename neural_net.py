import random
import sys
import pickle
from scipy.special import softmax
from math import log, sqrt, floor, exp
import numpy as np
import matplotlib.pyplot as p
import time

ORIGINAL_PROGRAM = False
TRADITIONAL_NET = False
HYBRID_APPROACH = False
ADAM_APPROACH = False
INTERESTING_PLOTS = False
MULTI_SGD = False
MULTI_K_TINY_SGD = True
MULTI_K_LITTLE_SGD = False
ADMM_WITH_SOFTMAX = False

t = time.time()

totalNumExamples = 1000
#n = 100
#totalNumExamples = 2

def padIntegerWithZeros(x, maxLength):
    if x == 0:
        return "0"*maxLength

    eps = 1e-8


    assert log(x+0.0001, 10) < maxLength

    return "0"*(maxLength-int(floor(log(x, 10)+eps))-1) + str(x)

def randomPermutationMat(n):
    i = np.identity(n)
#   rr = range(n)
    np.random.shuffle(i)
    return i



numExamples = 1000

assert totalNumExamples % numExamples == 0

numBatches = int(totalNumExamples / numExamples)

#numTeacherLayers = 2
numLayers = 2

#inertiaFraction = (numBatches - 1)/numBatches

def softPlus(x):
    return np.logaddexp(0, x)   

def softPlusPrime(x):
    #return softmax([x, 0])[0]
    if x >= 100:
        return 1
    if x <= -100:
        return 0
    return 1/(1+exp(-x))

def softPlusPrimeSofter(x):
    #return softmax([x, 0])[0]
    return 1/(1+relu(-x))


def computeChangeInWs(Ws, prevWs):
    totalSum = 0
    totalNumVals = 0

    for layer in range(numLayers):
        for cohort in range(numCohorts):
            w = Ws[layer][cohort]
            prevW = prevWs[layer][cohort]

            diffW = w - prevW
            totalSum += np.sum(np.multiply(diffW, diffW))
            totalNumVals += w.shape[0]#np.sum(np.ones(w.shape))

#    return totalSum / totalNumVals
    return totalSum

def relu(x):
    return x*(x>=0)

def reluPrime(x):
    return 1*(x>=0)

def softRelu(x):
    return relu(x) + softSignAbs(x)

def softReluPrime(x):
    return reluPrime(x) + softSignAbsPrime(x)

def leakyReluMaker(a):
    def leakyRelu(x):
        return x*(x>=0) + a*x*(x<0)
    return leakyRelu

def leakyReluPrimeMaker(a):
    def leakyReluPrime(x):
        return 1*(x>=0) + a*(x<0)
    return leakyReluPrime

def softSign(x):
    return x/(abs(x) + 1)

def softSignAbs(x):
    return (-abs(x)/(abs(x)+2) + 1)/2

def softSignAbsPrime(x):
    return -x/(abs(x) * (abs(x) + 2)**2)

def softSignPrime(x):
    return 1/(1 + abs(x))**2

def softSignPrime3(x):
    return 4/(2 + abs(x))**2

def softSign3(x):
    return x/(abs(x) + 2)

def softSignZeroOne3(x):
    return (softSign3(x)+1)/2

def softSignZeroOne(x):
    return (softSign(x)+1)/2

def softSignZeroOnePrime(x):
    return softSignPrime(x)/2

def softSignZeroOnePrime3(x):
    return softSignPrime3(x)/2

def softmaxByColumnMaker(k):
    def softmaxByColumn(mat):
#        print("input mat", mat)
        returnMat = []
        for col in np.transpose(mat):
#            print(col)
            returnMat.append(softmax(k*np.array(col)))
#            print(softmax(k*np.array(col)))
        return np.transpose(np.array(returnMat))

    return softmaxByColumn

#def softmaxPrime(vectorX):
#    return softmaxByColumn(vectorX) * (1 - softmaxByColumn(vectorX))

def softmaxPrimeMaker(k):
    softmaxByColumn = softmaxByColumnMaker(k)
    def softmaxPrime(mat, grad_output):
#        print("mat", mat)
#        print("grad_output", grad_output)

        outputK = softmaxByColumn(mat)
#        print("outputK", outputK)

        outputK = np.transpose(outputK)
    #                    print(grad_output.shape)
        tensor1 = np.einsum('ij,ik->ijk', outputK, outputK)  # (m, n, n)
#        print(tensor1.shape)
        # Second we need to create an (n,n) identity of the feature vector

    #                    print(tensor1.shape)
        # ( p1  0  0  ...  )
        # ( 0   p2 0  ...  )
        # ( ...            )
        tensor2 = np.einsum('ij,jk->ijk', outputK, np.identity(tensor1.shape[1]))  # (m, n, n)
#        tensor2 = np.einsum('ij,ik->ijk', outputK, np.identity(tensor1.shape[0]))  # (m, n, n)
#        print("t2", tensor2)
#        print(tensor2.shape)
        # Then we need to subtract the first tensor from the second
        # ( p1 - p1^2   -p1*p2   -p1*p3  ... )
        # ( -p1*p2     p2 - p2^2   -p2*p3 ...)
        # ( ...                              )
 #       print(tensor2.shape)
        dSoftmax = tensor2 - tensor1
#        print("dsoftmax", dSoftmax)
#        print("grad_output", grad_output)

        dSoftmax = np.swapaxes(dSoftmax, 1, 2)

        # Finally, we multiply the dSoftmax (da/dz) by da (dL/da) to get the gradient w.r.t. Z
#        dz = np.einsum('ijk,ki->ji', dSoftmax, grad_output)  # (m, n)
        dz = np.transpose(np.einsum('ijk,ik->ij', dSoftmax, np.transpose(grad_output)))  # (m, n)

#        print("dz", dz)
        return k * dz    

#    def softmaxPrime(mat, grad_output)

    return softmaxPrime

def softmaxPrimeMakerNoTranspose(k):
    softmaxByColumn = softmaxByColumnMaker(k)
    def softmaxPrime(mat, grad_output):
        outputK = softmaxByColumn(mat)
#        print("outputK", outputK)
    #                    print(grad_output.shape)
        tensor1 = np.einsum('ij,ik->ijk', outputK, outputK)  # (m, n, n)
#        print(tensor1.shape)
        # Second we need to create an (n,n) identity of the feature vector

    #                    print(tensor1.shape)
        # ( p1  0  0  ...  )
        # ( 0   p2 0  ...  )
        # ( ...            )
        tensor2 = np.einsum('ij,jk->ijk', outputK, np.identity(tensor1.shape[1]))  # (m, n, n)
#        tensor2 = np.einsum('ij,ik->ijk', outputK, np.identity(tensor1.shape[0]))  # (m, n, n)
#        print("t2", tensor2)
#        print(tensor2.shape)
        # Then we need to subtract the first tensor from the second
        # ( p1 - p1^2   -p1*p2   -p1*p3  ... )
        # ( -p1*p2     p2 - p2^2   -p2*p3 ...)
        # ( ...                              )
 #       print(tensor2.shape)
        dSoftmax = tensor2 - tensor1
#        print("dsoftmax", dSoftmax)
#        print("grad_output", grad_output)

        dSoftmax = np.swapaxes(dSoftmax, 1, 2)

        # Finally, we multiply the dSoftmax (da/dz) by da (dL/da) to get the gradient w.r.t. Z
#        dz = np.einsum('ijk,ki->ji', dSoftmax, grad_output)  # (m, n)
        dz = np.einsum('ijk,ik->ij', dSoftmax, grad_output)  # (m, n)


#ji -> ik



#ki -> ji

#        print("deriv", dz)

#        print("grad_output", grad_output)
#        print("mat", mat)
#        print("dz", dz)


        return k * dz    

#    def softmaxPrime(mat, grad_output)

    return softmaxPrime

def sigmoid(x):
    if x >= 100:
        return 1
    if x <= -100:
        return 0
    return 1/(1+exp(-x))

def sigmoidMaker(k):
    def sigmoid(x):
        if k*x >= 100:
            return 1
        if k*x <= -100:
            return 0
        return 1/(1+exp(-k*x))
    return sigmoid

def sigmoidPrimeMaker(k):
    def sigmoidPrime(x):
        sigmoid = sigmoidMaker(k)
        return k * sigmoid(x) * sigmoid(-x) 
    return sigmoidPrime

def sigmoidPrimeNormalizedMaker(k):
    def sigmoidPrime(x):
        sigmoid = sigmoidMaker(k)
        return sigmoid(x) * sigmoid(-x) 
#        return 1
    return sigmoidPrime

def softPlusMaker(k):
    def softPlus(x):
        if k*x > 100:
            return x
        return log(1 + exp(k*x))/k
    return softPlus

def softPlusPrimeMaker(k):
    def softPlusPrime(x):
        if k*x > 100:
            return 1
        return exp(k*x)/(1+exp(k*x))
    return softPlusPrime

def softPlusPrimeMaker2(k):
    def softPlusPrime(x):
        kx = k*x
        return exp(kx - np.logaddexp(0, kx))
    return softPlusPrime


def quadMaker(k):
    def quad(x):
        if abs(k*x) > 1e20:
            return 1e40
        return (k*x)**2
    return quad

def quadPrimeMaker(k):
    def quadPrime(x):
        if abs(k*x) > 1e20:
            return 1e20
        return 2*x*k**2
    return quadPrime

def sigmoidPrime(x):
    return sigmoid(x) * sigmoid(-x) 

def sigmoidOneMinusOne(x):
    return 2*(sigmoid(x)-0.5)

def sigmoidOneMinusOnePrime(x):
    return 2*sigmoidPrime(x)

def doubleSigmoid(x):
    return 2*sigmoid(x)

def line(x):
    return x
def linePrime(x):
    return 1

f, fPrime, fName = relu, reluPrime, "relu"
#f, fPrime, fName = relu, softPlusPrimeSofter, "relu"
#f, fPrime, fName = softPlus, softPlusPrime, "softPlus"
#f, fPrime, fName = line, linePrime, "line"

#f, fPrime, fName = softSign, softSignPrime, "softSign"
#f, fPrime, fName = sigmoid, sigmoidPrime, "sigmoid"
#f, fPrime, fName = sigmoid, softSignZeroOnePrime, "sigmoid"
#f, fPrime, fName = softSignZeroOne, sigmoid, "sigmoid"
#f, fPrime, fName = softSignZeroOne, softSignZeroOnePrime, "sigmoid"
#f, fPrime, fName = softSign, sigmoidPrime, "sigmoid"
#f, fPrime, fName = doubleSigmoid, softSignPrime, "sigmoid"
#f, fPrime, fName = sigmoidOneMinusOne, sigmoidOneMinusOnePrime, "sigmoidOneMinusOne"
#f, fPrime, fName = sigmoidOneMinusOne, softSignPrime, "sigmoidOneMinusOne"

def classicEvaluateNetwork(f, ws, X, showLayers=False, softmaxFinalLayer=False, softmaxByColumn=None):
    numLayers = len(ws)
    layerOutput = X
#    print("ws", ws)
    if showLayers:
        print("lo", layerOutput)
#        print(np.dot(ws[0], augmentWithBias(layerOutput)))
    for i in range(numLayers):
        if softmaxFinalLayer and i == numLayers - 1:
            layerOutput = softmaxByColumn(np.dot(ws[i], augmentWithBias(layerOutput)))
        else:
            layerOutput = np.vectorize(f)(np.dot(ws[i], augmentWithBias(layerOutput)))
        if showLayers:
            print("lo", i, layerOutput)

    return layerOutput


def evaluateNetwork(ws, X, matrixF, softmaxFinalLayer=False, showLayers=False):
    numLayers = len(ws)
    layerOutput = X


    for i in range(numLayers):
        if showLayers:
            p.matshow(matrixF(WsDotXs(ws[i], layerOutput)))
            p.colorbar()
            p.show()            
    


        if i < numLayers-1:
#           layerOutput = splitAndBiasX(np.vectorize(f)(WsDotXs(ws[i], layerOutput)))

            layerOutput = splitAndBiasX(matrixF(WsDotXs(ws[i], layerOutput)))
        else:
#           layerOutput = np.vectorize(f)(WsDotXs(ws[i], layerOutput))
            if softmaxFinalLayer:
#                print("layerInput", WsDotXs(ws[i], layerOutput))

                layerOutput = softmaxByColumnMaker(1)(WsDotXs(ws[i], layerOutput))
            else:
                layerOutput = matrixF(WsDotXs(ws[i], layerOutput))

#    print("layerOutput", layerOutput)


    return layerOutput

def augmentWithBias(X):
#   print(X)
#   print(np.ones((1, numExamples)))

    biasAugmentedX = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)
    return biasAugmentedX

def augmentWithZeros(X):
    biasAugmentedX = np.concatenate([X, np.zeros((1, numExamples))], axis=0)
    return biasAugmentedX


def diminishWithBias(W):
    biasDiminishedW = W[:,:-1]
    return biasDiminishedW

def selectSubset(overallX, overallT, index, numExamples):
    return overallX[:, index*numExamples:(index+1)*numExamples], \
        overallT[:, index*numExamples:(index+1)*numExamples]

def selectRandomSubset(overallX, overallT, numExamples):
    indices = np.random.choice(overallX.shape[1], numExamples, replace=False)


#   print("hi")

#   print(indices)
#   for i in range(numExamples):
#       print(np.transpose(overallT[:, indices])[i])
#       showExample(overallX[:, indices], i)

#   overallX[:, indices]

    return overallX[:, indices], overallT[:, indices]

def showExample(X, i):
    p.matshow(np.reshape(np.transpose(X)[i], (7,7)))
    p.show()

def repeatT(T, numRepeats):
    return np.repeat(T, numRepeats, axis=0)

#X = np.array([[1,2,3], [4,5,6], [7,8,9]])
#T = X
#T = np.array([[4,5,6], [7,8,9], [1,2,3]])

#print(selectRandomSubset(X, T, 2))
#sys.exit()


def batchArrayAlongZeroAxis(arr, batchSize):

    listOfBigFrames = []
    frameSum = np.zeros(arr[0].shape)
    numFrames = len(arr)

    for i in range(numFrames):
        frameSum += arr[i]

        if i % batchSize == batchSize - 1:
            listOfBigFrames.append(frameSum / batchSize)
            frameSum = np.zeros(arr[0].shape)

    if numFrames % batchSize != 0:
        listOfBigFrames.append(frameSum / (numFrames % batchSize))

#    print len(listOfBigFrames)
    return np.array(listOfBigFrames)

def batchArrayAlongAxis(arr, axis, batchSize):
    rearrangedArr = np.swapaxes(arr, 0, axis)
    batchedArray = batchArrayAlongZeroAxis(rearrangedArr, batchSize)
    return(np.swapaxes(batchedArray, 0, axis))

def vote(finalZ):
#   print(batchArrayAlongAxis(finalZ, 0, 5).shape)

    return np.argmax(batchArrayAlongAxis(finalZ, 0, 5), axis=0) 

mnistExample = False
if mnistExample:

    overallX, overallT = pickle.load(open("little_mnist_one_minus_one.p", "rb"))

#print(overallT)

#fName = "sigmoid"
#if fName == "relu" or fName == "softPlus" or fName == "sigmoid":
    overallT = np.maximum(overallT, np.zeros(overallT.shape))

    overallX = augmentWithBias(overallX)
    overallT = repeatT(overallT, 5)

else:
    n = 3
    numExamples = 3
    overallX = np.random.normal(size=(n, numExamples))

#    print(overallX)

    specialT=True
    if specialT:
        overallT = np.array([[0,0,0],[1,0,1],[0,1,0]])
    else:
        overallT = []
        for i in range(numExamples):
            newCol = [0]*n
            newCol[random.randint(0, n-1)] = 1
            overallT.append(newCol)

        overallT = np.transpose(np.array(overallT))

#    print(overallT)


#p.matshow(repeatT(overallT, 5))
#p.show()   

#p.plot(vote(repeatT(overallT, 5)))
#p.show()   



#print(X.shape)
#print(T.shape)

#X = np.random.normal(size=(n, numExamples))
#X = np.array([[1,-2]])
#X = np.identity(n)
#print(X.shape)

#teacherWs = [np.random.normal(0,1,size=(n,n+1)) for _ in range(numTeacherLayers)]

#print(teacherWs)

#T = evaluateNetwork(teacherWs, augmentWithBias(X))
#T = np.array([[1, 2]])
#T = np.array([[1, 2], [3, 4]])
#T = np.array([[1]])

#T = np.vectorize(softPlus)(np.random.normal(size=(n, numExamples)))
#T = np.vectorize(relu)(np.random.normal(size=(n, numExamples)))
#T = X


#print(X)
#print(T)

#layerOutput = X
#for i in range(numTeacherLayers):
#   layerOutput = np.dot(teacherWs[i], layerOutput)

#T = layerOutput




#T = np.array([[1]])
#T = np.random.exponential(size=(n, numExamples))


#print(T)


#X = T
#X = -T


if mnistExample:
    inputSize = 50
    middleSize = 50
    outputSize = 50 

    cohortSize = 50

else:
    inputSize = n
    middleSize = n
    outputSize = n
    cohortSize = n


numCohorts = 1


#randomPerm = randomPermutationMat(numCohorts*cohortSize)
randomPerm = np.identity(numCohorts*cohortSize)
#randomPerm = np.random.normal(0,1,size=(numCohorts*cohortSize, numCohorts*cohortSize))

#p.matshow(randomPerm)
#p.show()

#inputSize = 784
#middleSize = 784
#outputSize = 10

#layerSizes = [2,2]
layerSizes = [inputSize, middleSize, outputSize]
#layerSizes = [3, 3]
#layerSizes = [inputSize, middleSize, middleSize, outputSize]

#W = np.ones((n,n))

#                                     Looks like typo but it's not
#Ws = [np.random.normal(0,1,size=(layerSizes[i+1],layerSizes[i]+1)) for i in range(numLayers)]
Ws = [[np.random.normal(0,1,size=(cohortSize,cohortSize+1)) for _ in range(numCohorts)] for i in range(numLayers)]

#print(len(Ws[0]))

#Ws = [np.zeros((n, n+1)) for _ in range(numLayers)]
#Ws = [np.array([-1/3, 4/3]), np.array([1, 0])]

#Ys = [np.zeros((n, numExamples)) for _ in range(numLayers)]
Ys = [[np.random.normal(0,1,size=(layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]
lamdas = [[np.zeros((layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]

#Ys = [np.array([[-4,3]]), np.array([[1,2]])]
#Ys = [np.array([[1,-1]]), np.array([[-1,1]])]


#Zs = [np.ones((n, numExamples)) for _ in range(numLayers)]
Zs = [[np.random.normal(0,1,size=(layerSizes[i+1],numExamples)) for i in range(numLayers)] for _ in range(numBatches)]
mus = [[np.zeros((layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]

#Zs = [np.array([[1,2]]), np.array([[1,2]])]
#Y = T
#Z = Y


numIter = 500
#rhalpha = 1/2
#rhalpha = 1/((numExamples*n)**numLayers)
#rhalpha = 1/(numExamples*n)
#rhalpha = 1
#rho = 1/n
#alpha = 1/n**2
rho = 1
alpha = 1/inputSize
#alpha = 1/10
#rhalpha = 1e-3
#LR = 1/n

#bigWeightPenalty = 0
gamma = 100


def softLog(x):
    if x <= 0:
        return -75
    else:
        return log(x)

def splitAndBiasX(X):
    Xs = []

    for cohortIndex in range(numCohorts):
        Xs.append(augmentWithBias(X[cohortIndex*cohortSize:(cohortIndex+1)*cohortSize,:]))

    return Xs

def splitX(X):
    Xs = []

    for cohortIndex in range(numCohorts):
        Xs.append(X[cohortIndex*cohortSize:(cohortIndex+1)*cohortSize,:])

    return Xs   

def WsDotXs(Ws, Xs):#, permute=False):
#   print(len(Ws))
##  print([W.shape for W in Ws])
#   print(np.array(Ws).shape)
#   print([np.dot(W, X).shape for W, X in zip(Ws, Xs)])

#   if permute:
#       return np.dot(randomPerm, np.concatenate([np.dot(W, X) for W, X in zip(Ws, Xs)], axis=0))
#   else:
#       print([np.dot(W, X).shape for W, X in zip(Ws, Xs)])
    return np.concatenate([np.dot(W, X) for W, X in zip(Ws, Xs)], axis=0)

def lamdaStep(prevLamda, prevY, prevWs, Xs):
#   print(prevY.shape)
#   print(prevW.shape)
#   print(X.shape)
#   return prevLamda + alpha*(prevY - np.dot(prevW, X))
    return prevLamda + alpha*(prevY - WsDotXs(prevWs, Xs))


def muStep(prevMu, prevZ, fOfPrevY):
    return prevMu + alpha*(prevZ - fOfPrevY)

def finalZStep(T, mu, fOfPrevY, prevZ):
#   print("last Z", T - mu - rhalpha*(prevZ - fOfPrevY))

#   return T - mu - rho*(prevZ - fOfPrevY)
    if False:
        print("")
        print("-mu/rho", -mu/rho)
        print("fOfPrevY", fOfPrevY)
        print("T", T)
        print("")

#    print("Z", (-mu/rho + fOfPrevY + T)/2)

    return (-mu/rho + fOfPrevY + T)/2
#   return -mu/rho + fOfPrevY + T - prevZ

def ZStepNotLast(muThisLayer, fOfPrevYThisLayer, WsNextLayer, lamdaNextLayer, YNextLayer, prevZsThisLayer):
#   print("uh-oh!")
#   print("not last Z", -muThisLayer + fOfPrevYThisLayer + np.dot(np.transpose(WNextLayer), lamdaNextLayer) + \
#       rhalpha*np.dot(np.transpose(WNextLayer), YNextLayer - np.dot(WNextLayer, prevZThisLayer)))
    if False:
        print("")

        print("mu term", -muThisLayer/rho)
        print("fOfPrevY", fOfPrevYThisLayer)
        print("W transpose lambda", np.dot(np.transpose(WNextLayer), lamdaNextLayer)/rho)
        print("WT(Y-WZ) term", np.dot(np.transpose(WNextLayer), YNextLayer - np.dot(WNextLayer, prevZThisLayer)))
        print("overall", -muThisLayer/rho + fOfPrevYThisLayer + np.dot(np.transpose(diminishWithBias(WNextLayer)), lamdaNextLayer)/rho + \
        np.dot(np.transpose(diminishWithBias(WNextLayer)), YNextLayer - np.dot(WNextLayer, prevZThisLayer)))

        print("")

    diminishedTransposedWs = [np.transpose(diminishWithBias(W)) for W in WsNextLayer]

#np.dot(np.transpose(diminishWithBias(WNextLayer)), lamdaNextLayer)

#   return -muThisLayer/rho + fOfPrevYThisLayer + WsDotXs(diminishedTransposedWs, splitX(lamdaNextLayer))/rho + \
#       WsDotXs(diminishedTransposedWs, splitX(YNextLayer - WsDotXs(WsNextLayer, prevZsThisLayer)))

#   splitX(np.dot(np.transpose(randomPerm), lamdaNextLayer))

    return -muThisLayer/rho + fOfPrevYThisLayer + WsDotXs(diminishedTransposedWs, splitX(lamdaNextLayer))/rho + \
        WsDotXs(diminishedTransposedWs, splitX(YNextLayer - WsDotXs(WsNextLayer, prevZsThisLayer)))


#   wLamdaPlusYMat = np.dot(np.transpose(WNextLayer), np.transpose(augmentWithBias(np.transpose(lamdaNextLayer/rho + YNextLayer - np.dot(WNextLayer, prevZThisLayer)))))


#   return -muThisLayer/rho + fOfPrevYThisLayer + wLamdaPlusYMat

#   return fOfPrevYThisLayer + rhalpha*np.dot(np.transpose(diminishWithBias(WNextLayer)), lamdaNextLayer) + \
#       rhalpha*np.dot(np.transpose(diminishWithBias(WNextLayer)), YNextLayer - np.dot(WNextLayer, prevZThisLayer))


#   return -muThisLayer + fOfPrevYThisLayer + np.dot(np.transpose(WNextLayer), lamdaNextLayer) + \
#       rhalpha*np.dot(np.transpose(WNextLayer), YNextLayer - np.dot(WNextLayer, prevZThisLayer))

def ZStepNotLastInvertStyle(muThisLayer, fOfPrevYThisLayer, WNextLayer, lamdaNextLayer, YNextLayer, prevZThisLayer):
    matToInv = np.dot(np.transpose(diminishWithBias(WNextLayer)), WNextLayer) + np.transpose(augmentWithZeros(np.identity(n)))
#   invMat = np.dot(np.linalg.inv(np.dot(np.transpose(matToInv), matToInv)), np.transpose(diminishWithBias(matToInv)))
    invMat = np.dot(np.transpose(diminishWithBias(matToInv)), np.linalg.inv(np.dot(matToInv, np.transpose(matToInv))))
#   invMat = np.dot(np.linalg.inv(np.dot(np.transpose(matToInv), matToInv)), np.transpose(matToInv))


    wLamdaPlusYMat = np.dot(np.transpose(diminishWithBias(WNextLayer)), lamdaNextLayer/rho + YNextLayer)


    print(invMat.shape)
    print(muThisLayer.shape)
    print(wLamdaPlusYMat.shape)
    print(fOfPrevYThisLayer.shape)

    return np.dot(invMat, -muThisLayer/rho + fOfPrevYThisLayer + wLamdaPlusYMat)


def YStep(prevWs, Xs, mu, fPrimeOfPrevY, lamda, Z, fOfPrevY, softmaxPrime=False, T=None):
    if False:
        print("")
        print("Y exploration")
        print("WX term", np.dot(prevW, X))
        print("mu fprime term", mu*fPrimeOfPrevY/rho)
        print("lamda term", -lamda/rho)
        print("fPrime term", fPrimeOfPrevY*(Z - fOfPrevY))
        print("fPrimeOfPrevY", fPrimeOfPrevY)
        print("")

    if softmaxPrime:
        # we feed prevY in as fPrimeOfPrevY in this case
#        print("fPrimeOfPrevY", fPrimeOfPrevY)
#        print("fOfPrevY", fOfPrevY)

#        fPrimeOfPrevY = np.zeros((3,3))
#        fOfPrevY = softmaxByColumnMaker(1)(fPrimeOfPrevY)
#        return WsDotXs(prevWs, Xs) + mu*fPrimeOfPrevY/rho - lamda/rho + softmaxPrimeMaker(1)(fPrimeOfPrevY, fOfPrevY - T)

        return WsDotXs(prevWs, Xs) + mu*fPrimeOfPrevY/rho - lamda/rho + softmaxPrimeMaker(1)(fPrimeOfPrevY, Z - fOfPrevY)
    else:
        return WsDotXs(prevWs, Xs) + mu*fPrimeOfPrevY/rho - lamda/rho + fPrimeOfPrevY*np.dot(np.transpose(randomPerm), (Z - fOfPrevY))

def plotFunc(f):
    xs = np.linspace(-10, 10, 100)
    p.plot(xs, [f(x) for x in xs])
    p.show()

def WStepPerfect(lamda, prevW, Y, X):

#   print(X)

#   print(X)

#   if n >= numExamples:
    if X.shape[0] >= X.shape[1]:
        gramX = np.dot(np.transpose(X), X)
    else:
        gramX = np.dot(X, np.transpose(X))

#   print(X)

    frobeniusGramX = np.sum(gramX*gramX)

#   print(frobeniusGramX)

#   print(gramX + \
#           bigWeightPenalty*frobeniusGramX*np.identity(numExamples))

#   if n >= numExamples:
    if X.shape[0] >= X.shape[1]:
        pseudoInv = np.dot(np.linalg.inv(gramX + \
            bigWeightPenalty*np.identity(numExamples)), np.transpose(X))
    else:
#       print(np.dot(X, np.transpose(X)) + \
#           bigWeightPenalty*np.identity(n))
        pseudoInv = np.dot(np.transpose(X), np.linalg.inv(gramX + \
            bigWeightPenalty*np.identity(gramX.shape[0])))

#   print("lamda plus Y", lamda + Y)
#   print("pseudoInv", pseudoInv)


    return inertiaFraction*prevW + (1-inertiaFraction)*np.dot((lamda/rho + Y), pseudoInv)   



def WStepNew(lamda, prevW, Y, X):
#   print(np.array(X).shape)
    inputLayerSize = X.shape[0] # this is like n+1


    gramX = np.dot(X, np.transpose(X))
#    print(gramX)

    gamma = 10*max(np.sum(gramX), 1)
    print("gcond1", gamma)
    gamma = np.linalg.cond(gramX + gamma*np.identity(inputLayerSize), "fro")
    print("gcond", gamma)

    firstMat = gamma*prevW + np.dot((lamda/rho + Y), np.transpose(X))
    secondMat = np.linalg.inv(gramX + gamma*np.identity(inputLayerSize))

#    if np.sum(np.dot(secondMat, np.transpose(secondMat)) > 

    return np.dot(firstMat, secondMat)

def jointWStepNew(lamda, prevWs, Y, Xs):
    lamdas = splitX(lamda)
    Ys = splitX(Y)

    return [WStepNew(lamda, prevW, Y, X) for lamda, prevW, Y, X in zip(lamdas, prevWs, Ys, Xs)]

def doSGD(f, fPrime, numIter=500, learningRate=3e-3, loadFromPickle=False, startingVals=None, movie=False,
    softmaxFinalLayer=False, trainingError=True, k1=1, k2=1, n=3, numExamples=5):

#   plotFunc(f)
#   plotFunc(fPrime)

    if not trainingError:
        overallXtest, overallTtest = pickle.load(open("little_mnist_one_minus_one_test.p", "rb"))

        overallTtest = np.maximum(overallTtest, np.zeros(overallTtest.shape))
        overallTtest = repeatT(overallTtest, 5)
        overallXtest = augmentWithBias(overallXtest)

#   overallT = np.maximum(overallT, np.zeros(overallT.shape))

#    cohortSize=n
#    layerSizes=[n,n,n]
#    numLayers=2
#    n = n

    small=True
    tiny=False

    if small:
        n = 10
        cohortSize=n
#        layerSizes=[n,n,n]
#        numLayers=2
        layerSizes=[n,n,n,n]
        numLayers=3
#        print(n)
        n = n
        numExamples = 5
    if tiny:
        cohortSize=1
        layerSizes=[1,1]
        numLayers=1
        n = 1
        numExamples = 1

    overallX = np.random.normal(size=(n,numExamples))

#    overallX = np.array([[0]])
#    overallT = np.array([[0.75]])

    overallT = []
    for i in range(numExamples):
        newCol = [0]*n
        newCol[random.randint(0, n-1)] = 1
        overallT.append(newCol)

    overallT = np.transpose(np.array(overallT))



    if loadFromPickle:
        if startingVals == None:
            Ws = pickle.load(open("ws3.p", "rb"))
            Ys = pickle.load(open("ys3.p", "rb"))
            Zs = pickle.load(open("zs3.p", "rb"))
            overallT = pickle.load(open("t3.p", "rb"))
        #        overallT = np.array([[0,0,0],[0,1,0],[1,0,1]])
            overallX = pickle.load(open("x3.p", "rb"))

        else:
            Ws, Ys, Zs, overallT, overallX = startingVals
#        print(Ws)
#        print(Ys)
#        print(Zs)
#        print(overallT)


    else:
        Ws = [np.random.normal(0,1,size=(cohortSize,cohortSize+1)) for i in range(numLayers)]       
#        Ws = [np.zeros((cohortSize,cohortSize+1)) for i in range(numLayers)]       
#        Ws = [1.1*np.ones((cohortSize,cohortSize+1)) for i in range(numLayers)]       
#        Ws = [np.array([[0., 4.]])]

        Ys = [np.random.normal(0,1,size=(layerSizes[i+1], numExamples)) for i in range(numLayers)]
        Zs = [np.random.normal(0,1,size=(layerSizes[i+1],numExamples)) for i in range(numLayers)]

        if tiny: 
            overallX = np.array([[0]])
            overallT = np.array([[0.75]])

#        print(Ws)
#        print(Ys)
#        print(Zs)
#        print(overallT)
        overwrite=False
        if overwrite and not tiny:
            pickle.dump(Ws, open("ws3.p", "wb"))
            pickle.dump(Ys, open("ys3.p", "wb"))
            pickle.dump(Zs, open("zs3.p", "wb"))
            pickle.dump(overallX, open("x3.p", "wb"))
            pickle.dump(overallT, open("t3.p", "wb"))

    logErrors = []

    updates = []

    counter = 0

#   learningRate = 3e-3

#       if numIter >= 10:
#           moniker = "startup"

#       else:
#           moniker = "vanilla"

    moniker = "SGD"


    softmaxByColumn = softmaxByColumnMaker(k1)
    softmaxPrime = softmaxPrimeMaker(k2)


    for iteration in range(numIter):
#       print(Ws)

#        print("iter", iteration)

        # FEEDFORWARD STAGE
        layerOutput = overallX

        for i in range(numLayers):
        
#            print('w', Ws[i])           

            Ys[i] = np.dot(Ws[i], augmentWithBias(layerOutput))

            if softmaxFinalLayer and i == numLayers - 1:
                Zs[i] = softmaxByColumn(Ys[i])
#               p.matshow(softmaxByColumn(Ys[i]))
#               p.colorbar()
#               p.show()
#                print(Zs[i])


            else:
                Zs[i] = np.vectorize(f)(Ys[i])


#            print("y",Ys[i])
#            print("z",Zs[i])

            layerOutput = Zs[i]
 #           print("layerOutput", layerOutput)


        # BACKPROP STAGE

        outputErrors = []

        if softmaxFinalLayer:
            outputError = softmaxPrime(Ys[-1], Zs[-1] - overallT)
        else:
            outputError = (Zs[-1] - overallT) * np.vectorize(fPrime)(Ys[-1])
#           print(outputError.shape)
#        print('fprime of y', np.vectorize(fPrime)(Ys[-1]))

        outputErrors.append(outputError)
#        print(outputError)

#        print('OE',outputError)
#           print(numLayers)

        if numLayers > 1:
            for i in range(numLayers-2, -1, -1):

#                print("oe", outputError)
#                print("fPrime", np.vectorize(fPrime)(Ys[i]))
#                print("dot1", np.transpose(diminishWithBias(Ws[i+1])))
#                print("dot2", outputErrors[-1])

                outputError = np.dot(np.transpose(diminishWithBias(Ws[i+1])), outputErrors[-1]) * np.vectorize(fPrime)(Ys[i])
#                print("oe", outputError)

                outputErrors.append(outputError)
#                   print(outputErrors[-1].shape)
        outputErrors = outputErrors[::-1]
 #       print("err", outputErrors)

#           print([outputError.shape for outputError in outputErrors])


        As = [overallX] + [Zs[i] for i in range(numLayers)]

#        print(As)

  #      print('A',As)
#           print(As[0].shape, np.transpose(outputErrors[i]).shape)

#           print(outputErrors)

#        print(outputErrors)

        gradients = [np.transpose(np.dot(augmentWithBias(As[i]), np.transpose(outputErrors[i]))) for i in range(numLayers)]

    #    print('g',gradients)
   #     print('')
#           print(Zs[-1])

#           p.matshow(gradients[1])
#           p.colorbar()
#           p.show()

#           print([gradient.shape for gradient in gradients])

#        learningRate *= 0.999



#           print(Ws[i])
        if trainingError:
            output = classicEvaluateNetwork(f, Ws, overallX, showLayers=False, softmaxFinalLayer=softmaxFinalLayer, softmaxByColumn=softmaxByColumn)
    #           print(Zs[-1] - output)
  #          print(output)

            diffMat = overallT - output
#            print(overallT)
#            print(output)
#            print(diffMat)

            error = np.trace(np.dot(np.transpose(diffMat), diffMat))
        else:
            outputTest = classicEvaluateNetwork(f, Ws, overallXtest, showLayers=False, softmaxFinalLayer=softmaxFinalLayer, softmaxByColumn=softmaxByColumn)
            diffMatTest = overallTtest - outputTest
            error = np.trace(np.dot(np.transpose(diffMatTest), diffMatTest))            

#        print(gradients)

        for i in range(numLayers):
#            print(Ws[i])
#            print(gradients[i])
            Ws[i] -= gradients[i] * learningRate

#        print(overallT)
#        print("output", output)

#        print(Ws[0])
#            print(Ws[i])
#        print(gradients)


        updates.append(np.sum(gradients[0]) * learningRate)

        if movie:
            p.clf()
            p.matshow(Ws[0])
            p.colorbar()
            p.savefig("nn_movie/W0_" + moniker + "_frame_" + padIntegerWithZeros(counter,3) + ".png")
            p.close()

            p.clf()
            p.matshow(Ws[1])
            p.colorbar()
            p.savefig("nn_movie/W1_" + moniker + "_frame_" + padIntegerWithZeros(counter,3) + ".png")
            p.close()

        counter += 1



#       print(error)
        logErrors.append(softLog(error))
    #   logErrors.append(Ys[0][0])

    return logErrors, updates, counter

def compareManySGDs(k1s, k2s, n, numExamples, showIntermediates=False):
    
    errorMat = []
#    for k1 in [1/64,1/32,0.0625, 0.125, 0.25, 0.5, 1, 2,4,8,16]:
#    for k1 in k1s[::-1]:
#        print(k1)

#    overallX = np.array([[0]])
#    overallT = np.array([[0.75]])
    overallX = np.random.normal(size=(n,numExamples))

    overallT = []
    for i in range(numExamples):
        newCol = [0]*n
        newCol[random.randint(0, n-1)] = 1
        overallT.append(newCol)

    overallT = np.transpose(np.array(overallT))

    cohortSize = n
    layerSizes = [n,n,n]

    Ws = [np.zeros((cohortSize,cohortSize+1)) for i in range(numLayers)]       
#        Ws = [1.1*np.ones((cohortSize,cohortSize+1)) for i in range(numLayers)]       
#        Ws = [np.array([[0., 4.]])]

    Ys = [np.random.normal(0,1,size=(layerSizes[i+1], numExamples)) for i in range(numLayers)]
    Zs = [np.random.normal(0,1,size=(layerSizes[i+1],numExamples)) for i in range(numLayers)]


    for k2 in k2s[::-1]:
        print(k2)
        errorMat.append([])
        for k1 in k1s[::-1]:
#        for k2 in k2s[::-1]:

#        for k2 in [1/64,1/32,0.0625, 0.125, 0.25, 0.5, 1, 2]:
#            plotFunc(sigmoidMaker(k1))
#            plotFunc(sigmoidPrimeMaker(k2))
#            print(k2)
#            f, fPrime = quadMaker(k1), quadPrimeMaker(k2)
#            f, fPrime = sigmoidMaker(k1), sigmoidPrimeMaker(k2)
            f, fPrime = softPlusMaker(k1), softPlusPrimeMaker(k2)
#            f, fPrime = None, None
            logErrors, updates, counter = doSGD(f, fPrime, numIter=200, learningRate=1, loadFromPickle=True, \
                startingVals=(Ws, Ys, Zs, overallT, overallX), movie=False, softmaxFinalLayer=True, \
                k1=1, k2=1, n=n, numExamples=numExamples)

            p.plot(logErrors, label=str(k1) + " " + str(k2))
#            p.show()
            errorMat[-1].append(max(logErrors[-1], -10))
#            p.plot([softLog(abs(u)) for u in updates], label=str(k1) + " " + str(k2))
#            p.plot([u for u in updates], label=str(k1) + " " + str(k2))
        if showIntermediates:
            p.legend()
            p.show()




    p.matshow(errorMat)
    p.xlabel("k2")
    p.ylabel("k1")
    ax=p.gca()
    ax.set_xticklabels([""] + k1s)
    ax.set_yticklabels([""] + k2s)
    p.colorbar()
    p.show()    


def doADMMLearning(numLayers, numBatches, numExamples, numCohorts, layerSizes, f, fPrime, softmaxFinalLayer):
#    Ws = [[np.random.normal(0,1,size=(cohortSize,cohortSize+1)) for _ in range(numCohorts)] for i in range(numLayers)]
    Ws = [[np.zeros((cohortSize,cohortSize+1)) for _ in range(numCohorts)] for i in range(numLayers)]

    #print(len(Ws[0]))

    #Ws = [np.zeros((n, n+1)) for _ in range(numLayers)]
    #Ws = [np.array([-1/3, 4/3]), np.array([1, 0])]

    #Ys = [np.zeros((n, numExamples)) for _ in range(numLayers)]
    Ys = [[np.random.normal(0,1,size=(layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]
    lamdas = [[np.zeros((layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]

    #Ys = [np.array([[-4,3]]), np.array([[1,2]])]
    #Ys = [np.array([[1,-1]]), np.array([[-1,1]])]


    #Zs = [np.ones((n, numExamples)) for _ in range(numLayers)]
    Zs = [[np.random.normal(0,1,size=(layerSizes[i+1],numExamples)) for i in range(numLayers)] for _ in range(numBatches)]
    mus = [[np.zeros((layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]

    errors = []
#       logErrors = []

    percent = 0

    lamda0s = []
    mu0s = []
    W0s = []
    Y0s = []
    Z0s = []
    lamda1s = []
    mu1s = []
    W1s = []
    Y1s = []
    Z1s = []

    W2s = []

    batchIndex = 0

    numIterADMM = 2000

    movie=False
    counter = 0

    gammas = []

    logErrors = []

    scaleFactor = 1e3
    avgDiff = 1
    beta2 = 0.

    for i in range(numIterADMM):
        if i/numIter > percent/100 and (i-1)/numIter <= percent/100:
            print(i, "/", numIter)
            percent += 1

        X, T = selectSubset(overallX, overallT, batchIndex, numExamples)    
    #   X, T = selectRandomSubset(overallX, overallT, numExamples)  

#       biasAugmentedX = augmentWithBias(X)
        splitAndBiasedXs = splitAndBiasX(X)

        batchIndex = (batchIndex + 1) % numBatches

        prevLamdas = lamdas[:]
        prevMus = mus[:]
        prevZs = Zs[:]
        prevYs = Ys[:]
        prevWs = [W[:] for W in Ws]

    #   print(W)

#       fOfPrevYs = [[np.vectorize(f)(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]
#       fPrimeOfPrevYs = [[np.vectorize(fPrime)(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]

        matrixF = lambda Y: np.dot(randomPerm, np.vectorize(f)(Y))
        matrixFprime = lambda Y: np.dot(randomPerm, np.vectorize(fPrime)(Y))

#       matrixF = np.vectorize(f)
#       matrixFprime = np.vectorize(fPrime)

        if softmaxFinalLayer:
#            print(prevYs[0][-1])
#            print(softmaxByColumnMaker(1)(prevYs[0][-1]))

#            print("pys", prevYs)

#            print("")
 #           print("prevZs", prevZs[0][-2])
 #           print("prevWs", prevWs[0][-1])
 #           print("dot", np.dot(prevWs[0][-1], augmentWithBias(prevZs[0][-2])))
 #           print("")

 #           print("")
 #           print("bpys", prevYs[0][-1])
 #           print("smbpys", softmaxByColumnMaker(1)(prevYs[0][-1]))
 #           print("")

            fOfPrevYs = [[matrixF(prevY) for prevY in batchPrevYs[:-1]] + [softmaxByColumnMaker(1)(batchPrevYs[-1])] for batchPrevYs in prevYs]
#            print("fopys", fOfPrevYs)

            fPrimeOfPrevYs = [[matrixFprime(prevY) for prevY in batchPrevYs[:-1]] for batchPrevYs in prevYs] # Second part not necessary; special-cased

        else:
            fOfPrevYs = [[matrixF(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]
            fPrimeOfPrevYs = [[matrixFprime(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]


        for layer in range(numLayers-1, -1, -1):

            splitAndBiasedPrevZs = splitAndBiasX(prevZs[batchIndex][layer-1])


    #       print(layer)

            # Lambda step
            if layer == 0:
                lamdas[batchIndex][layer] = lamdaStep(prevLamdas[batchIndex][layer], prevYs[batchIndex][layer], prevWs[layer], splitAndBiasedXs)
            else:
#               print(np.array(prevWs[layer]).shape)

                lamdas[batchIndex][layer] = lamdaStep(prevLamdas[batchIndex][layer], prevYs[batchIndex][layer], \
                    prevWs[layer], splitAndBiasedPrevZs)

    #       print("lamda",layer, lamdas[layer])

            if layer == 0:
                lamda0s.append(lamdas[batchIndex][layer][0][0])
            if layer == 1:
                lamda1s.append(lamdas[batchIndex][layer][0][0])


            # Mu step
    #       print(layer)
    #       print(len(mus))
    #       print(len(prevMus))
    #       print(len(prevZs))
    #       print(len(fOfPrevYs))
            mus[batchIndex][layer] = muStep(prevMus[batchIndex][layer], prevZs[batchIndex][layer], fOfPrevYs[batchIndex][layer])

    #       print("mu", layer, mus[layer])

            if layer == 0:
                mu0s.append([mus[batchIndex][layer]][0][0][0])
            if layer == 1:
                mu1s.append([mus[batchIndex][layer]][0][0][0])


            # Z step
            if layer == numLayers - 1:
                Zs[batchIndex][layer] = finalZStep(T, mus[batchIndex][layer], fOfPrevYs[batchIndex][layer], prevZs[batchIndex][layer])
            else:
                Zs[batchIndex][layer] = ZStepNotLast(mus[batchIndex][layer], fOfPrevYs[batchIndex][layer], \
                    Ws[layer+1], lamdas[batchIndex][layer+1], Ys[batchIndex][layer+1], splitAndBiasX(prevZs[batchIndex][layer]))

    #       print("Y",layer, Ys[layer])
    #       print("Z",layer, Zs[layer])

            if layer == 0:
                Z0s.append([Zs[batchIndex][layer]][0][0][0])
            if layer == 1:
                Z1s.append([Zs[batchIndex][layer]][0][0][0])



            # Y step
            if layer == numLayers-1 and softmaxFinalLayer:
                if layer == 0:
                    # We intentionally feed prevYs in the place of fPrimeOfPrevYs
                    Ys[batchIndex][layer] = YStep(prevWs[layer], splitAndBiasedXs, mus[batchIndex][layer], prevYs[batchIndex][layer], \
                        lamdas[batchIndex][layer], Zs[batchIndex][layer], fOfPrevYs[batchIndex][layer], softmaxPrime=True, T=T)
                else:
                    # We intentionally feed prevYs in the place of fPrimeOfPrevYs
                    Ys[batchIndex][layer] = YStep(prevWs[layer], splitAndBiasX(prevZs[batchIndex][layer-1]), mus[batchIndex][layer], prevYs[batchIndex][layer], \
                        lamdas[batchIndex][layer], Zs[batchIndex][layer], fOfPrevYs[batchIndex][layer], softmaxPrime=True, T=T)                         
            else:
                if layer == 0:
                    Ys[batchIndex][layer] = YStep(prevWs[layer], splitAndBiasedXs, mus[batchIndex][layer], fPrimeOfPrevYs[batchIndex][layer], \
                        lamdas[batchIndex][layer], Zs[batchIndex][layer], fOfPrevYs[batchIndex][layer])
                else:
                    Ys[batchIndex][layer] = YStep(prevWs[layer], splitAndBiasX(prevZs[batchIndex][layer-1]), mus[batchIndex][layer], fPrimeOfPrevYs[batchIndex][layer], \
                        lamdas[batchIndex][layer], Zs[batchIndex][layer], fOfPrevYs[batchIndex][layer])         

    #       print("Y",layer, Ys[layer])

            if layer == 0:
                Y0s.append([Ys[batchIndex][layer]][0][0][0])
            if layer == 1:
                Y1s.append([Ys[batchIndex][layer]][0][0][0])


            # W step
            if layer == 0:
                Ws[layer] = jointWStepNew(lamdas[batchIndex][layer], prevWs[layer], Ys[batchIndex][layer], splitAndBiasedXs)
            else:
                Ws[layer] = jointWStepNew(lamdas[batchIndex][layer], prevWs[layer], Ys[batchIndex][layer], splitAndBiasX(prevZs[batchIndex][layer-1]))

            avgDiff = beta2*avgDiff + (1-beta2)*computeChangeInWs(Ws, prevWs)

#            print(gamma)

    #       print("W",layer, Ws[layer])

            if layer == 0:
                W0s.append([Ws[layer]][0][0][0])
            if layer == 1:
                W1s.append([Ws[layer]][0][0][0])
            if layer == 2:
                W2s.append([Ws[layer]][0][0][0])


        gamma = max(scaleFactor*sqrt(avgDiff), 1)
        print(gamma)
        gammas.append(gamma)


    #   mu = muStep(prevMu, prevZ, fOfPrevY)
    #   Z = ZStep(T, mu, prevZ, fOfPrevY)
    #   Y = YStep(prevW, X, mu, fPrimeOfPrevY, lamda, Z, fOfPrevY)  
    #   W = WStepPerfect(lamda, prevW, Y, X)

#       if i % 200 == 0:
        if False:
            output = evaluateNetwork(Ws, splitAndBiasedXs, matrixF, softmaxFinalLayer=softmaxFinalLayer, showLayers=True)
            p.matshow(T)
            p.colorbar()
            p.show()

            p.matshow(T - output)
            p.title("output - T")
            p.colorbar()
            p.show()
        else:
            output = evaluateNetwork(Ws, splitAndBiasedXs, matrixF, softmaxFinalLayer=softmaxFinalLayer, showLayers=False)

        if movie:
            p.clf()
            p.matshow(Ws[0][0])
            p.colorbar()
            p.savefig("nn_movie/W0_" + moniker + "_frame_" + padIntegerWithZeros(counter,3) + ".png")
            p.close()

            p.clf()
            p.matshow(Ws[1][0])
            p.colorbar()
            p.savefig("nn_movie/W1_" + moniker + "_frame_" + padIntegerWithZeros(counter,3) + ".png")
            p.close()

        counter += 1

#        print("output", output)
    #   print("T", T)

        diffMat = T - output

#       print(diffMat)

    #   print(T)
    #   print(output)

        error = np.trace(np.dot(np.transpose(diffMat), diffMat))

        errors.append(error)
        logErrors.append(softLog(error))

    numCorrect = 0


#       p.matshow(X)
#       p.colorbar()
#       p.show()                
    output = evaluateNetwork(Ws, splitAndBiasedXs, matrixF, showLayers=False)

#   p.matshow(output)
#   p.colorbar()
#   p.show()

    for batchIndex in range(numBatches):
        X, T = selectSubset(overallX, overallT, batchIndex, numExamples)    
    #   X = augmentWithBias(X)
    #   X, T = selectRandomSubset(overallX, overallT, numExamples)  
        voteT = vote(T)

        splitAndBiasedXs = splitAndBiasX(X)

        output = evaluateNetwork(Ws, splitAndBiasedXs, matrixF, softmaxFinalLayer=softmaxFinalLayer)
        voteOutput = vote(output)

#       print(output)

        for i in range(numExamples):
        #   print(np.transpose(output)[i])
        #   print(np.argmax(np.transpose(output)[i]))

        #   print(np.transpose(T)[i])
        #   print(np.argmax(np.transpose(T)[i]))

        #   print("System's guess:", np.argmax(np.transpose(output)[i]), \
        #       "Correct:", np.argmax(np.transpose(T)[i]))

#           print(voteOutput[i], voteT[i])

            if voteOutput[i] == voteT[i]:
                numCorrect += 1

    accuracy = numCorrect / totalNumExamples

    print("Training Accuracy:", accuracy)

    overallXtest, overallTtest = pickle.load(open("little_mnist_one_minus_one_test.p", "rb"))

    overallTtest = repeatT(overallTtest, 5)
    overallXtest = augmentWithBias(overallXtest)

    #testOutput = evaluateNetwork(Ws, augmentWithBias(Xtest))

    numCorrect = 0
    for batchIndex in range(numBatches):
    #   X, Ttest = selectRandomSubset(Xtest, Ttest, numExamples)    
        X, Ttest = selectSubset(overallXtest, overallTtest, batchIndex, numExamples)    
    #   X = augmentWithBias(X)

        splitAndBiasedXs = splitAndBiasX(X)

        testOutput = evaluateNetwork(Ws, splitAndBiasedXs, matrixF, softmaxFinalLayer=softmaxFinalLayer)

        voteTtest = vote(Ttest)
        voteTestOutput = vote(testOutput)

    #   print(batchIndex)
    #   print("a", testOutput.shape)
    #   print("b", Ttest.shape)


        for i in range(numExamples):
    #       print("System's guess:", np.argmax(np.transpose(testOutput)[i]), \
    #           "Correct:", np.argmax(np.transpose(Ttest)[i]))

    #       p.matshow(np.reshape(np.transpose(X)[i], (7,7)))
    #       p.show()



            if voteTestOutput[i] == voteTtest[i]:
                numCorrect += 1

    accuracy = numCorrect / totalNumExamples
    print("Test Accuracy:", accuracy)


    #   p.matshow(np.reshape(np.transpose(X)[i], (7,7)))
    #   p.show()



    #print("teacher weights", teacherWs)
    #print("student weights", Ws)

    #print("teacher output", evaluateNetwork(teacherWs, X))
    #print("student output", evaluateNetwork(Ws, X))
    #print(T)
    #print(output)
    #print(diffMat)

    #print(Ys)

#       pickle.dump(logErrors, open("log_errors.p", "wb"))
#    logErrors2 = pickle.load(open("log_errors.p", "rb"))
    p.plot(lamda0s, label='lamda0s')
    p.plot(mu0s, label='mu0s')
    p.plot(W0s, label='W0s')
    p.plot(Y0s, label='Y0s')
    p.plot(Z0s, label='Z0s')
    p.plot(lamda1s, label='lamda1s')
    p.plot(mu1s, label='mu1s')
    p.plot(W1s, label='W1s')
    p.plot(Y1s, label='Y1s')
    p.plot(Z1s, label='Z1s')
    #p.plot(W2s, label="W2s")

    p.legend()
    p.show()

    print(time.time() - t)
    p.plot(logErrors)
#    p.plot(logErrors2)
    p.ylabel("Log error")
    p.xlabel("Iteration")
    p.show()        

    p.plot(gammas)
    p.show()

#def rearrange(fullZ, cohortSize=cohortSize, numCohorts=numCohorts, numExamples=numExamples):
#   fullZ = np.reshape(fullZ, (cohortSize, numCohorts, numExamples))
#   fullZ = np.swapaxes(fullZ, 0, 1)
#   fullZ = np.reshape(fullZ, (cohortSize*numCohorts, numExamples))
#   return fullZ

#def rearrangeMaker():
#   return 

if __name__ == "__main__":

    if ORIGINAL_PROGRAM:

        errors = []
        logErrors = []

        percent = 0

        lamda0s = []
        mu0s = []
        W0s = []
        Y0s = []
        Z0s = []
        lamda1s = []
        mu1s = []
        W1s = []
        Y1s = []
        Z1s = []

        W2s = []

        batchIndex = 0

        for i in range(numIter):
            if i/numIter > percent/100 and (i-1)/numIter <= percent/100:
                print(i, "/", numIter)
                percent += 1

            X, T = selectSubset(overallX, overallT, batchIndex, numExamples)    
        #   X, T = selectRandomSubset(overallX, overallT, numExamples)  

    #       biasAugmentedX = augmentWithBias(X)
            splitAndBiasedXs = splitAndBiasX(X)

            batchIndex = (batchIndex + 1) % numBatches

            prevLamdas = lamdas[:]
            prevMus = mus[:]
            prevZs = Zs[:]
            prevYs = Ys[:]
            prevWs = [W[:] for W in Ws]

        #   print(W)

    #       fOfPrevYs = [[np.vectorize(f)(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]
    #       fPrimeOfPrevYs = [[np.vectorize(fPrime)(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]

            matrixF = lambda Y: np.dot(randomPerm, np.vectorize(f)(Y))
    #       matrixFprime = lambda Y: np.dot(randomPerm, np.vectorize(fPrime)(Y))

    #       matrixF = np.vectorize(f)
    #       matrixFprime = np.vectorize(fPrime)

            fOfPrevYs = [[matrixF(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]
            fPrimeOfPrevYs = [[np.vectorize(fPrime)(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]


            for layer in range(numLayers-1, -1, -1):

                splitAndBiasedPrevZs = splitAndBiasX(prevZs[batchIndex][layer-1])


        #       print(layer)

                # Lambda step
                if layer == 0:
                    lamdas[batchIndex][layer] = lamdaStep(prevLamdas[batchIndex][layer], prevYs[batchIndex][layer], prevWs[layer], splitAndBiasedXs)
                else:
    #               print(np.array(prevWs[layer]).shape)

                    lamdas[batchIndex][layer] = lamdaStep(prevLamdas[batchIndex][layer], prevYs[batchIndex][layer], \
                        prevWs[layer], splitAndBiasedPrevZs)

        #       print("lamda",layer, lamdas[layer])

                if layer == 0:
                    lamda0s.append(lamdas[batchIndex][layer][0][0])
                if layer == 1:
                    lamda1s.append(lamdas[batchIndex][layer][0][0])


                # Mu step
        #       print(layer)
        #       print(len(mus))
        #       print(len(prevMus))
        #       print(len(prevZs))
        #       print(len(fOfPrevYs))
                mus[batchIndex][layer] = muStep(prevMus[batchIndex][layer], prevZs[batchIndex][layer], fOfPrevYs[batchIndex][layer])

        #       print("mu", layer, mus[layer])

                if layer == 0:
                    mu0s.append([mus[batchIndex][layer]][0][0][0])
                if layer == 1:
                    mu1s.append([mus[batchIndex][layer]][0][0][0])


                # Z step
                if layer == numLayers - 1:
                    Zs[batchIndex][layer] = finalZStep(T, mus[batchIndex][layer], fOfPrevYs[batchIndex][layer], prevZs[batchIndex][layer])
                else:
                    Zs[batchIndex][layer] = ZStepNotLast(mus[batchIndex][layer], fOfPrevYs[batchIndex][layer], \
                        Ws[layer+1], lamdas[batchIndex][layer+1], Ys[batchIndex][layer+1], splitAndBiasX(prevZs[batchIndex][layer]))

        #       print("Y",layer, Ys[layer])
        #       print("Z",layer, Zs[layer])

                if layer == 0:
                    Z0s.append([Zs[batchIndex][layer]][0][0][0])
                if layer == 1:
                    Z1s.append([Zs[batchIndex][layer]][0][0][0])



                # Y step
                if layer == 0:
                    Ys[batchIndex][layer] = YStep(prevWs[layer], splitAndBiasedXs, mus[batchIndex][layer], fPrimeOfPrevYs[batchIndex][layer], \
                        lamdas[batchIndex][layer], Zs[batchIndex][layer], fOfPrevYs[batchIndex][layer])
                else:
                    Ys[batchIndex][layer] = YStep(prevWs[layer], splitAndBiasX(prevZs[batchIndex][layer-1]), mus[batchIndex][layer], fPrimeOfPrevYs[batchIndex][layer], \
                        lamdas[batchIndex][layer], Zs[batchIndex][layer], fOfPrevYs[batchIndex][layer])         

        #       print("Y",layer, Ys[layer])

                if layer == 0:
                    Y0s.append([Ys[batchIndex][layer]][0][0][0])
                if layer == 1:
                    Y1s.append([Ys[batchIndex][layer]][0][0][0])


                # W step
                if layer == 0:
                    Ws[layer] = jointWStepNew(lamdas[batchIndex][layer], prevWs[layer], Ys[batchIndex][layer], splitAndBiasedXs)
                else:
                    Ws[layer] = jointWStepNew(lamdas[batchIndex][layer], prevWs[layer], Ys[batchIndex][layer], splitAndBiasX(prevZs[batchIndex][layer-1]))

        #       print("W",layer, Ws[layer])

                if layer == 0:
                    W0s.append([Ws[layer]][0][0][0])
                if layer == 1:
                    W1s.append([Ws[layer]][0][0][0])
                if layer == 2:
                    W2s.append([Ws[layer]][0][0][0])


        #   mu = muStep(prevMu, prevZ, fOfPrevY)
        #   Z = ZStep(T, mu, prevZ, fOfPrevY)
        #   Y = YStep(prevW, X, mu, fPrimeOfPrevY, lamda, Z, fOfPrevY)  
        #   W = WStepPerfect(lamda, prevW, Y, X)

    #       if i % 200 == 0:
            if False:
                output = evaluateNetwork(Ws, splitAndBiasedXs, matrixF, showLayers=True)
                p.matshow(T)
                p.colorbar()
                p.show()

                p.matshow(T - output)
                p.title("output - T")
                p.colorbar()
                p.show()
            else:
                output = evaluateNetwork(Ws, splitAndBiasedXs, matrixF, showLayers=False)


        #   print("output", output)
        #   print("T", T)

            diffMat = T - output

    #       print(diffMat)

        #   print(T)
        #   print(output)

            error = np.trace(np.dot(np.transpose(diffMat), diffMat))

            errors.append(error)
            logErrors.append(softLog(error))

        numCorrect = 0


#       p.matshow(X)
#       p.colorbar()
#       p.show()                
        output = evaluateNetwork(Ws, splitAndBiasedXs, matrixF, showLayers=False)

    #   p.matshow(output)
    #   p.colorbar()
    #   p.show()

        for batchIndex in range(numBatches):
            X, T = selectSubset(overallX, overallT, batchIndex, numExamples)    
        #   X = augmentWithBias(X)
        #   X, T = selectRandomSubset(overallX, overallT, numExamples)  
            voteT = vote(T)

            splitAndBiasedXs = splitAndBiasX(X)

            output = evaluateNetwork(Ws, splitAndBiasedXs, matrixF)
            voteOutput = vote(output)

    #       print(output)

            for i in range(numExamples):
            #   print(np.transpose(output)[i])
            #   print(np.argmax(np.transpose(output)[i]))

            #   print(np.transpose(T)[i])
            #   print(np.argmax(np.transpose(T)[i]))

            #   print("System's guess:", np.argmax(np.transpose(output)[i]), \
            #       "Correct:", np.argmax(np.transpose(T)[i]))

    #           print(voteOutput[i], voteT[i])

                if voteOutput[i] == voteT[i]:
                    numCorrect += 1

        accuracy = numCorrect / totalNumExamples

        print("Training Accuracy:", accuracy)

        overallXtest, overallTtest = pickle.load(open("little_mnist_one_minus_one_test.p", "rb"))

        overallTtest = repeatT(overallTtest, 5)
        overallXtest = augmentWithBias(overallXtest)

        #testOutput = evaluateNetwork(Ws, augmentWithBias(Xtest))

        numCorrect = 0
        for batchIndex in range(numBatches):
        #   X, Ttest = selectRandomSubset(Xtest, Ttest, numExamples)    
            X, Ttest = selectSubset(overallXtest, overallTtest, batchIndex, numExamples)    
        #   X = augmentWithBias(X)

            splitAndBiasedXs = splitAndBiasX(X)

            testOutput = evaluateNetwork(Ws, splitAndBiasedXs)

            voteTtest = vote(Ttest)
            voteTestOutput = vote(testOutput)

        #   print(batchIndex)
        #   print("a", testOutput.shape)
        #   print("b", Ttest.shape)


            for i in range(numExamples):
        #       print("System's guess:", np.argmax(np.transpose(testOutput)[i]), \
        #           "Correct:", np.argmax(np.transpose(Ttest)[i]))

        #       p.matshow(np.reshape(np.transpose(X)[i], (7,7)))
        #       p.show()



                if voteTestOutput[i] == voteTtest[i]:
                    numCorrect += 1

        accuracy = numCorrect / totalNumExamples
        print("Test Accuracy:", accuracy)


        #   p.matshow(np.reshape(np.transpose(X)[i], (7,7)))
        #   p.show()



        #print("teacher weights", teacherWs)
        #print("student weights", Ws)

        #print("teacher output", evaluateNetwork(teacherWs, X))
        #print("student output", evaluateNetwork(Ws, X))
        #print(T)
        #print(output)
        #print(diffMat)

        #print(Ys)

        p.plot(lamda0s, label='lamda0s')
        p.plot(mu0s, label='mu0s')
        p.plot(W0s, label='W0s')
        p.plot(Y0s, label='Y0s')
        p.plot(Z0s, label='Z0s')
        p.plot(lamda1s, label='lamda1s')
        p.plot(mu1s, label='mu1s')
        p.plot(W1s, label='W1s')
        p.plot(Y1s, label='Y1s')
        p.plot(Z1s, label='Z1s')
        #p.plot(W2s, label="W2s")

        p.legend()
        p.show()

        print(time.time() - t)
        p.plot(logErrors)
        p.ylabel("Log error")
        p.xlabel("Iteration")
        p.show()    

    if TRADITIONAL_NET:
        Ws = [np.random.normal(0,1,size=(cohortSize,cohortSize+1)) for i in range(numLayers)]

        #print(len(Ws[0]))


        #Ys = [np.zeros((n, numExamples)) for _ in range(numLayers)]
        Ys = [np.random.normal(0,1,size=(layerSizes[i+1], numExamples)) for i in range(numLayers)]

        #Zs = [np.ones((n, numExamples)) for _ in range(numLayers)]
        Zs = [np.random.normal(0,1,size=(layerSizes[i+1],numExamples)) for i in range(numLayers)]
#       mus = [[np.zeros((layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]
    
        logErrors = []

        numIter = 1000
    
        for iteration in range(numIter):

            layerOutput = overallX

            for i in range(numLayers):

#               if showLayers:
#                   p.matshow(matrixF(WsDotXs(ws[i], layerOutput)))
#                   p.colorbar()
#                   p.show()            

                

                Ys[i] = np.dot(Ws[i], augmentWithBias(layerOutput))
                Zs[i] = np.vectorize(f)(Ys[i])
        #           layerOutput = splitAndBiasX(matrixF(WsDotXs(ws[i], layerOutput)))


                layerOutput = Zs[i]
#                   layerOutput = WsDotXs(ws[i], layerOutput)

#           layerOutputs.append(layerOutput)

    #           print(Zs[-1].shape)
    #           print(overallT.shape)
    #           print(Ys[-1].shape)

            outputErrors = []

            outputError = (Zs[-1] - overallT) * np.vectorize(fPrime)(Ys[-1])
    #           print(outputError.shape)

            outputErrors.append(outputError)

#           print(numLayers)

            if numLayers > 1:
                for i in range(numLayers-2, -1, -1):
#                   print(outputErrors[-1].shape)
#                   print(Ws[i+1].shape)

                    outputError = np.dot(np.transpose(diminishWithBias(Ws[i+1])), outputErrors[-1])
                    outputErrors.append(outputError)
#                   print(outputErrors[-1].shape)
            outputErrors = outputErrors[::-1]

#           print([outputError.shape for outputError in outputErrors])


            As = [overallX] + [Zs[i] for i in range(numLayers)]

#           print(As[0].shape, np.transpose(outputErrors[i]).shape)

#           print(outputErrors)

            gradients = [np.transpose(np.dot(augmentWithBias(As[i]), np.transpose(outputErrors[i]))) for i in range(numLayers)]

#           print(Zs[-1])

#           p.matshow(gradients[1])
#           p.show()

#           print([gradient.shape for gradient in gradients])
            learningRate = 3e-3





#           print(Ws[i])

            output = classicEvaluateNetwork(Ws, overallX, showLayers=False)
#           print(Zs[-1] - output)


            for i in range(numLayers):
                Ws[i] -= gradients[i] * learningRate
        #   print("output", output)
        #   print("T", overallT)
#           print(diffMat)

            diffMat = overallT - output
#           if iteration > 90:

#               p.matshow(diffMat)
#               p.show()
#           print(diffMat)
#           print(overallT)
#           print(diffMat)

        #   print(T)
#           print(diffMat)

            error = np.trace(np.dot(np.transpose(diffMat), diffMat))

            print(error)
            logErrors.append(softLog(error))

        print(time.time() - t)

        p.plot(logErrors)
        p.show()

    if HYBRID_APPROACH:

#       Ws = [np.random.normal(0,1,size=(cohortSize,cohortSize+1)) for i in range(numLayers)]

        #print(len(Ws[0]))

        movie=False

        #Ys = [np.zeros((n, numExamples)) for _ in range(numLayers)]
#       Ys = [np.random.normal(0,1,size=(layerSizes[i+1], numExamples)) for i in range(numLayers)]

        #Zs = [np.ones((n, numExamples)) for _ in range(numLayers)]
#       Zs = [np.random.normal(0,1,size=(layerSizes[i+1],numExamples)) for i in range(numLayers)]
#       mus = [[np.zeros((layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]

    
#       pickle.dump(Ws, open("ws.p", "wb"))
#       pickle.dump(Ys, open("ys.p", "wb"))
#       pickle.dump(Zs, open("zs.p", "wb"))
        
        Ws = pickle.load(open("ws.p", "rb"))
        Ys = pickle.load(open("ys.p", "rb"))
        Zs = pickle.load(open("zs.p", "rb"))


        logErrors = []

        numIterGrad = 100
        numIterMats = 1


        counter = 0

        learningRate = 3e-3

#       if numIter >= 10:
#           moniker = "startup"

#       else:
#           moniker = "vanilla"

        moniker = "SGD"

        for iteration in range(numIterGrad):

            layerOutput = overallX

            for i in range(numLayers):

#               if showLayers:
#                   p.matshow(matrixF(WsDotXs(ws[i], layerOutput)))
#                   p.colorbar()
#                   p.show()            

                

                Ys[i] = np.dot(Ws[i], augmentWithBias(layerOutput))
                Zs[i] = np.vectorize(f)(Ys[i])
        #           layerOutput = splitAndBiasX(matrixF(WsDotXs(ws[i], layerOutput)))


                layerOutput = Zs[i]
#                   layerOutput = WsDotXs(ws[i], layerOutput)

#           layerOutputs.append(layerOutput)

    #           print(Zs[-1].shape)
    #           print(overallT.shape)
    #           print(Ys[-1].shape)

#           p.matshow(np.vectorize(fPrime)(Ys[-1]))
#           p.colorbar()
#           p.show()

            outputErrors = []

            outputError = (Zs[-1] - overallT) * np.vectorize(fPrime)(Ys[-1])
#           outputError = (Zs[-1] - overallT) * np.vectorize(softSignPrime)(Ys[-1])
    #           print(outputError.shape)

            outputErrors.append(outputError)

#           print(numLayers)

            if numLayers > 1:
                for i in range(numLayers-2, -1, -1):
#                   print(outputErrors[-1].shape)
#                   print(Ws[i+1].shape)

                    outputError = np.dot(np.transpose(diminishWithBias(Ws[i+1])), outputErrors[-1]) * np.vectorize(fPrime)(Ys[i])
                    outputErrors.append(outputError)
#                   print(outputErrors[-1].shape)
            outputErrors = outputErrors[::-1]

#           print([outputError.shape for outputError in outputErrors])


            As = [overallX] + [Zs[i] for i in range(numLayers)]

#           print(As[0].shape, np.transpose(outputErrors[i]).shape)

#           print(outputErrors)

            gradients = [np.transpose(np.dot(augmentWithBias(As[i]), np.transpose(outputErrors[i]))) for i in range(numLayers)]

#           print(Zs[-1])

#           p.matshow(gradients[1])
#           p.colorbar()
#           p.show()

#           print([gradient.shape for gradient in gradients])

            learningRate *= 0.999



#           print(Ws[i])

            output = classicEvaluateNetwork(Ws, overallX, showLayers=False)
#           print(Zs[-1] - output)


            for i in range(numLayers):
                Ws[i] -= gradients[i] * learningRate

            if movie:
                p.clf()
                p.matshow(Ws[0])
                p.colorbar()
                p.savefig("nn_movie/W0_" + moniker + "_frame_" + padIntegerWithZeros(counter,3) + ".png")
                p.close()

                p.clf()
                p.matshow(Ws[1])
                p.colorbar()
                p.savefig("nn_movie/W1_" + moniker + "_frame_" + padIntegerWithZeros(counter,3) + ".png")
                p.close()

            counter += 1

        #   print("output", output)
        #   print("T", overallT)
#           print(diffMat)

            diffMat = overallT - output

#           p.matshow(diffMat)
#           p.colorbar()
#           p.show()

#           if iteration > 90:

#               p.matshow(diffMat)
#               p.show()
#           print(diffMat)
#           print(overallT)
#           print(diffMat)

        #   print(T)
#           print(diffMat)

            error = np.trace(np.dot(np.transpose(diffMat), diffMat))

            print(error)
            logErrors.append(softLog(error))



#       print(time.time() - t)

#       p.plot(logErrors)
#       p.show()        
        Ws = [[Ws[i] for _ in range(numCohorts)] for i in range(numLayers)]

        #print(len(Ws[0]))

        #Ws = [np.zeros((n, n+1)) for _ in range(numLayers)]
        #Ws = [np.array([-1/3, 4/3]), np.array([1, 0])]

        #Ys = [np.zeros((n, numExamples)) for _ in range(numLayers)]
        Ys = [[Ys[i] for i in range(numLayers)] for _ in range(numBatches)]
        lamdas = [[np.zeros((layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]

        #Ys = [np.array([[-4,3]]), np.array([[1,2]])]
        #Ys = [np.array([[1,-1]]), np.array([[-1,1]])]


        #Zs = [np.ones((n, numExamples)) for _ in range(numLayers)]
        Zs = [[Zs[i] for i in range(numLayers)] for _ in range(numBatches)]
        mus = [[np.zeros((layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]


        errors = []
#       logErrors = []

        percent = 0

        lamda0s = []
        mu0s = []
        W0s = []
        Y0s = []
        Z0s = []
        lamda1s = []
        mu1s = []
        W1s = []
        Y1s = []
        Z1s = []

        W2s = []

        batchIndex = 0

        for i in range(numIterMats):
            if i/numIter > percent/100 and (i-1)/numIter <= percent/100:
                print(i, "/", numIter)
                percent += 1

            X, T = selectSubset(overallX, overallT, batchIndex, numExamples)    
        #   X, T = selectRandomSubset(overallX, overallT, numExamples)  

    #       biasAugmentedX = augmentWithBias(X)
            splitAndBiasedXs = splitAndBiasX(X)

            batchIndex = (batchIndex + 1) % numBatches

            prevLamdas = lamdas[:]
            prevMus = mus[:]
            prevZs = Zs[:]
            prevYs = Ys[:]
            prevWs = [W[:] for W in Ws]

        #   print(W)

    #       fOfPrevYs = [[np.vectorize(f)(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]
    #       fPrimeOfPrevYs = [[np.vectorize(fPrime)(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]

            matrixF = lambda Y: np.dot(randomPerm, np.vectorize(f)(Y))
    #       matrixFprime = lambda Y: np.dot(randomPerm, np.vectorize(fPrime)(Y))

    #       matrixF = np.vectorize(f)
    #       matrixFprime = np.vectorize(fPrime)

            fOfPrevYs = [[matrixF(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]
            fPrimeOfPrevYs = [[np.vectorize(fPrime)(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]


            for layer in range(numLayers-1, -1, -1):

                splitAndBiasedPrevZs = splitAndBiasX(prevZs[batchIndex][layer-1])


        #       print(layer)

                # Lambda step
                if layer == 0:
                    lamdas[batchIndex][layer] = lamdaStep(prevLamdas[batchIndex][layer], prevYs[batchIndex][layer], prevWs[layer], splitAndBiasedXs)
                else:
    #               print(np.array(prevWs[layer]).shape)

                    lamdas[batchIndex][layer] = lamdaStep(prevLamdas[batchIndex][layer], prevYs[batchIndex][layer], \
                        prevWs[layer], splitAndBiasedPrevZs)

        #       print("lamda",layer, lamdas[layer])

                if layer == 0:
                    lamda0s.append(lamdas[batchIndex][layer][0][0])
                if layer == 1:
                    lamda1s.append(lamdas[batchIndex][layer][0][0])


                # Mu step
        #       print(layer)
        #       print(len(mus))
        #       print(len(prevMus))
        #       print(len(prevZs))
        #       print(len(fOfPrevYs))
                mus[batchIndex][layer] = muStep(prevMus[batchIndex][layer], prevZs[batchIndex][layer], fOfPrevYs[batchIndex][layer])

        #       print("mu", layer, mus[layer])

                if layer == 0:
                    mu0s.append([mus[batchIndex][layer]][0][0][0])
                if layer == 1:
                    mu1s.append([mus[batchIndex][layer]][0][0][0])


                # Z step
                if layer == numLayers - 1:
                    Zs[batchIndex][layer] = finalZStep(T, mus[batchIndex][layer], fOfPrevYs[batchIndex][layer], prevZs[batchIndex][layer])
                else:
                    Zs[batchIndex][layer] = ZStepNotLast(mus[batchIndex][layer], fOfPrevYs[batchIndex][layer], \
                        Ws[layer+1], lamdas[batchIndex][layer+1], Ys[batchIndex][layer+1], splitAndBiasX(prevZs[batchIndex][layer]))

        #       print("Y",layer, Ys[layer])
        #       print("Z",layer, Zs[layer])

                if layer == 0:
                    Z0s.append([Zs[batchIndex][layer]][0][0][0])
                if layer == 1:
                    Z1s.append([Zs[batchIndex][layer]][0][0][0])



                # Y step
                if layer == 0:
                    Ys[batchIndex][layer] = YStep(prevWs[layer], splitAndBiasedXs, mus[batchIndex][layer], fPrimeOfPrevYs[batchIndex][layer], \
                        lamdas[batchIndex][layer], Zs[batchIndex][layer], fOfPrevYs[batchIndex][layer])
                else:
                    Ys[batchIndex][layer] = YStep(prevWs[layer], splitAndBiasX(prevZs[batchIndex][layer-1]), mus[batchIndex][layer], fPrimeOfPrevYs[batchIndex][layer], \
                        lamdas[batchIndex][layer], Zs[batchIndex][layer], fOfPrevYs[batchIndex][layer])         

        #       print("Y",layer, Ys[layer])

                if layer == 0:
                    Y0s.append([Ys[batchIndex][layer]][0][0][0])
                if layer == 1:
                    Y1s.append([Ys[batchIndex][layer]][0][0][0])


                # W step
                if layer == 0:
                    Ws[layer] = jointWStepNew(lamdas[batchIndex][layer], prevWs[layer], Ys[batchIndex][layer], splitAndBiasedXs)
                else:
                    Ws[layer] = jointWStepNew(lamdas[batchIndex][layer], prevWs[layer], Ys[batchIndex][layer], splitAndBiasX(prevZs[batchIndex][layer-1]))

        #       print("W",layer, Ws[layer])

                if layer == 0:
                    W0s.append([Ws[layer]][0][0][0])
                if layer == 1:
                    W1s.append([Ws[layer]][0][0][0])
                if layer == 2:
                    W2s.append([Ws[layer]][0][0][0])



        #   mu = muStep(prevMu, prevZ, fOfPrevY)
        #   Z = ZStep(T, mu, prevZ, fOfPrevY)
        #   Y = YStep(prevW, X, mu, fPrimeOfPrevY, lamda, Z, fOfPrevY)  
        #   W = WStepPerfect(lamda, prevW, Y, X)

    #       if i % 200 == 0:
            if False:
                output = evaluateNetwork(Ws, splitAndBiasedXs, showLayers=True)
                p.matshow(T)
                p.colorbar()
                p.show()

                p.matshow(T - output)
                p.title("output - T")
                p.colorbar()
                p.show()
            else:
                output = evaluateNetwork(Ws, splitAndBiasedXs, showLayers=False)

            if movie:
                p.clf()
                p.matshow(Ws[0][0])
                p.colorbar()
                p.savefig("nn_movie/W0_" + moniker + "_frame_" + padIntegerWithZeros(counter,3) + ".png")
                p.close()

                p.clf()
                p.matshow(Ws[1][0])
                p.colorbar()
                p.savefig("nn_movie/W1_" + moniker + "_frame_" + padIntegerWithZeros(counter,3) + ".png")
                p.close()

            counter += 1

        #   print("output", output)
        #   print("T", T)

            diffMat = T - output

    #       print(diffMat)

        #   print(T)
        #   print(output)

            error = np.trace(np.dot(np.transpose(diffMat), diffMat))

            errors.append(error)
            logErrors.append(softLog(error))

        numCorrect = 0


#       p.matshow(X)
#       p.colorbar()
#       p.show()                
        output = evaluateNetwork(Ws, splitAndBiasedXs, showLayers=False)

    #   p.matshow(output)
    #   p.colorbar()
    #   p.show()

        for batchIndex in range(numBatches):
            X, T = selectSubset(overallX, overallT, batchIndex, numExamples)    
        #   X = augmentWithBias(X)
        #   X, T = selectRandomSubset(overallX, overallT, numExamples)  
            voteT = vote(T)

            splitAndBiasedXs = splitAndBiasX(X)

            output = evaluateNetwork(Ws, splitAndBiasedXs)
            voteOutput = vote(output)

    #       print(output)

            for i in range(numExamples):
            #   print(np.transpose(output)[i])
            #   print(np.argmax(np.transpose(output)[i]))

            #   print(np.transpose(T)[i])
            #   print(np.argmax(np.transpose(T)[i]))

            #   print("System's guess:", np.argmax(np.transpose(output)[i]), \
            #       "Correct:", np.argmax(np.transpose(T)[i]))

    #           print(voteOutput[i], voteT[i])

                if voteOutput[i] == voteT[i]:
                    numCorrect += 1

        accuracy = numCorrect / totalNumExamples

        print("Training Accuracy:", accuracy)

        overallXtest, overallTtest = pickle.load(open("little_mnist_one_minus_one_test.p", "rb"))

        overallTtest = repeatT(overallTtest, 5)
        overallXtest = augmentWithBias(overallXtest)

        #testOutput = evaluateNetwork(Ws, augmentWithBias(Xtest))

        numCorrect = 0
        for batchIndex in range(numBatches):
        #   X, Ttest = selectRandomSubset(Xtest, Ttest, numExamples)    
            X, Ttest = selectSubset(overallXtest, overallTtest, batchIndex, numExamples)    
        #   X = augmentWithBias(X)

            splitAndBiasedXs = splitAndBiasX(X)

            testOutput = evaluateNetwork(Ws, splitAndBiasedXs)

            voteTtest = vote(Ttest)
            voteTestOutput = vote(testOutput)

        #   print(batchIndex)
        #   print("a", testOutput.shape)
        #   print("b", Ttest.shape)


            for i in range(numExamples):
        #       print("System's guess:", np.argmax(np.transpose(testOutput)[i]), \
        #           "Correct:", np.argmax(np.transpose(Ttest)[i]))

        #       p.matshow(np.reshape(np.transpose(X)[i], (7,7)))
        #       p.show()



                if voteTestOutput[i] == voteTtest[i]:
                    numCorrect += 1

        accuracy = numCorrect / totalNumExamples
        print("Test Accuracy:", accuracy)


        #   p.matshow(np.reshape(np.transpose(X)[i], (7,7)))
        #   p.show()



        #print("teacher weights", teacherWs)
        #print("student weights", Ws)

        #print("teacher output", evaluateNetwork(teacherWs, X))
        #print("student output", evaluateNetwork(Ws, X))
        #print(T)
        #print(output)
        #print(diffMat)

        #print(Ys)

#       pickle.dump(logErrors, open("log_errors_sig_soft.p", "wb"))
#       pickle.dump(logErrors, open("log_errors_soft_sig.p", "wb"))
#       pickle.dump(logErrors, open("log_errors_grad_soft_soft.p", "wb"))
#       pickle.dump(logErrors, open("log_errors_grad_soft_sig.p", "wb"))
#       pickle.dump(logErrors, open("log_errors_grad_sig_soft.p", "wb"))
        pickle.dump(logErrors, open("log_errors_grad_sig_sig.p", "wb"))
#       logErrors2 = pickle.load(open("log_errors.p", "rb"))
        p.plot(lamda0s, label='lamda0s')
        p.plot(mu0s, label='mu0s')
        p.plot(W0s, label='W0s')
        p.plot(Y0s, label='Y0s')
        p.plot(Z0s, label='Z0s')
        p.plot(lamda1s, label='lamda1s')
        p.plot(mu1s, label='mu1s')
        p.plot(W1s, label='W1s')
        p.plot(Y1s, label='Y1s')
        p.plot(Z1s, label='Z1s')
        #p.plot(W2s, label="W2s")

        p.legend()
        p.show()

#       logErrorsSigSig = pickle.load(open("log_errors_grad_sig_sig.p", "rb"))
#       logErrorsSigSoft = pickle.load(open("log_errors_grad_sig_soft.p", "rb"))
#       logErrorsSoftSig = pickle.load(open("log_errors_grad_soft_sig.p", "rb"))
#       logErrorsSoftSoft = pickle.load(open("log_errors_grad_soft_soft.p", "rb"))

        print(time.time() - t)

#       p.plot(logErrorsSigSig, label="f=sig,f'=sig")
#       p.plot(logErrorsSoftSig, label="f=soft,f'=sig")
#       p.plot(logErrorsSigSoft, label="f=sig,f'=soft")
#       p.plot(logErrorsSoftSoft, label="f=soft,f'=soft")
#       p.legend()
        p.plot(logErrors)
#       p.plot(logErrors2)
        p.ylabel("Log error")
        p.xlabel("Iteration")
        p.show()    

    if ADAM_APPROACH:

        Ws = [np.random.normal(0,1,size=(cohortSize,cohortSize+1)) for i in range(numLayers)]

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        learningRate = 3e-2

        #print(len(Ws[0]))

        movie=False

        #Ys = [np.zeros((n, numExamples)) for _ in range(numLayers)]
        Ys = [np.random.normal(0,1,size=(layerSizes[i+1], numExamples)) for i in range(numLayers)]

        #Zs = [np.ones((n, numExamples)) for _ in range(numLayers)]
        Zs = [np.random.normal(0,1,size=(layerSizes[i+1],numExamples)) for i in range(numLayers)]
#       mus = [[np.zeros((layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]
    
        logErrors = []

        numIterGrad = 500
        numIterMats = 1


        mtMinusOne = [np.zeros((cohortSize,cohortSize+1)) for i in range(numLayers)]
        vtMinusOne = [np.zeros((cohortSize,cohortSize+1)) for i in range(numLayers)]
        mt = [np.zeros((cohortSize,cohortSize+1)) for i in range(numLayers)]
        vt = [np.zeros((cohortSize,cohortSize+1)) for i in range(numLayers)]

        mthat = [np.zeros((cohortSize,cohortSize+1)) for i in range(numLayers)]
        vthat = [np.zeros((cohortSize,cohortSize+1)) for i in range(numLayers)]
        update = [np.zeros((cohortSize,cohortSize+1)) for i in range(numLayers)]

        counter = 0


        if numIter >= 10:
            moniker = "startup"

        else:
            moniker = "vanilla"

        for iteration in range(numIterGrad):

            layerOutput = overallX

            for i in range(numLayers):

#               if showLayers:
#                   p.matshow(matrixF(WsDotXs(ws[i], layerOutput)))
#                   p.colorbar()
#                   p.show()            

                

                Ys[i] = np.dot(Ws[i], augmentWithBias(layerOutput))
                Zs[i] = np.vectorize(f)(Ys[i])
        #           layerOutput = splitAndBiasX(matrixF(WsDotXs(ws[i], layerOutput)))


                layerOutput = Zs[i]
#                   layerOutput = WsDotXs(ws[i], layerOutput)

#           layerOutputs.append(layerOutput)

    #           print(Zs[-1].shape)
    #           print(overallT.shape)
    #           print(Ys[-1].shape)

            outputErrors = []

            outputError = (Zs[-1] - overallT) * np.vectorize(fPrime)(Ys[-1])
    #           print(outputError.shape)

            outputErrors.append(outputError)

#           print(numLayers)

            if numLayers > 1:
                for i in range(numLayers-2, -1, -1):
#                   print(outputErrors[-1].shape)
#                   print(Ws[i+1].shape)

                    outputError = np.dot(np.transpose(diminishWithBias(Ws[i+1])), outputErrors[-1])
                    outputErrors.append(outputError)
#                   print(outputErrors[-1].shape)

            outputErrors = outputErrors[::-1]

#           print([outputError.shape for outputError in outputErrors])


            As = [overallX] + [Zs[i] for i in range(numLayers)]

#           print(As[0].shape, np.transpose(outputErrors[i]).shape)

#           print(outputErrors)

            gradients = [np.transpose(np.dot(augmentWithBias(As[i]), np.transpose(outputErrors[i]))) for i in range(numLayers)]

            for i in range(numLayers):
                mt[i] = beta1 * mtMinusOne[i] + (1 - beta1) * gradients[i]

                squaredGrad = np.multiply(gradients[i], gradients[i])
                vt[i] = beta2 * vtMinusOne[i] + (1 - beta2) * squaredGrad

                mthat[i] = mt[i]/(1-beta1**(iteration+1))
                vthat[i] = vt[i]/(1-beta2**(iteration+1))

                update[i] = learningRate * mthat[i] / (np.sqrt(vthat[i]) + eps)

                Ws[i] -= update[i]


#           print(Zs[-1])

#           p.matshow(gradients[1])
#           p.show()

#           print([gradient.shape for gradient in gradients])

            learningRate *= 0.999



#           print(Ws[i])

            output = classicEvaluateNetwork(Ws, overallX, showLayers=False)
#           print(Zs[-1] - output)


##          for i in range(numLayers):
#               Ws[i] -= gradients[i] * learningRate

            if movie:
                p.clf()
                p.matshow(Ws[0])
                p.colorbar()
                p.savefig("nn_movie/W0_" + moniker + "_frame_" + padIntegerWithZeros(counter,3) + ".png")
                p.close()

                p.clf()
                p.matshow(Ws[1])
                p.colorbar()
                p.savefig("nn_movie/W1_" + moniker + "_frame_" + padIntegerWithZeros(counter,3) + ".png")
                p.close()

            counter += 1

        #   print("output", output)
        #   print("T", overallT)
#           print(diffMat)

            diffMat = overallT - output
#           if iteration > 90:

#               p.matshow(diffMat)
#               p.show()
#           print(diffMat)
#           print(overallT)
#           print(diffMat)

        #   print(T)
#           print(diffMat)

            error = np.trace(np.dot(np.transpose(diffMat), diffMat))

            print(error)
            logErrors.append(softLog(error))

            mtMinusOne = mt
            vtMinusOne = vt


#       print(time.time() - t)

#       p.plot(logErrors)
#       p.show()        
        Ws = [[Ws[i] for _ in range(numCohorts)] for i in range(numLayers)]

        #print(len(Ws[0]))

        #Ws = [np.zeros((n, n+1)) for _ in range(numLayers)]
        #Ws = [np.array([-1/3, 4/3]), np.array([1, 0])]

        #Ys = [np.zeros((n, numExamples)) for _ in range(numLayers)]
        Ys = [[Ys[i] for i in range(numLayers)] for _ in range(numBatches)]
        lamdas = [[np.zeros((layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]

        #Ys = [np.array([[-4,3]]), np.array([[1,2]])]
        #Ys = [np.array([[1,-1]]), np.array([[-1,1]])]


        #Zs = [np.ones((n, numExamples)) for _ in range(numLayers)]
        Zs = [[Zs[i] for i in range(numLayers)] for _ in range(numBatches)]
        mus = [[np.zeros((layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]


        errors = []
#       logErrors = []

        percent = 0

        lamda0s = []
        mu0s = []
        W0s = []
        Y0s = []
        Z0s = []
        lamda1s = []
        mu1s = []
        W1s = []
        Y1s = []
        Z1s = []

        W2s = []

        batchIndex = 0

        for i in range(numIterMats):
            if i/numIter > percent/100 and (i-1)/numIter <= percent/100:
                print(i, "/", numIter)
                percent += 1

            X, T = selectSubset(overallX, overallT, batchIndex, numExamples)    
        #   X, T = selectRandomSubset(overallX, overallT, numExamples)  

    #       biasAugmentedX = augmentWithBias(X)
            splitAndBiasedXs = splitAndBiasX(X)

            batchIndex = (batchIndex + 1) % numBatches

            prevLamdas = lamdas[:]
            prevMus = mus[:]
            prevZs = Zs[:]
            prevYs = Ys[:]
            prevWs = [W[:] for W in Ws]

        #   print(W)

    #       fOfPrevYs = [[np.vectorize(f)(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]
    #       fPrimeOfPrevYs = [[np.vectorize(fPrime)(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]

            matrixF = lambda Y: np.dot(randomPerm, np.vectorize(f)(Y))
    #       matrixFprime = lambda Y: np.dot(randomPerm, np.vectorize(fPrime)(Y))

    #       matrixF = np.vectorize(f)
    #       matrixFprime = np.vectorize(fPrime)

            fOfPrevYs = [[matrixF(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]
            fPrimeOfPrevYs = [[np.vectorize(fPrime)(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]


            for layer in range(numLayers-1, -1, -1):

                splitAndBiasedPrevZs = splitAndBiasX(prevZs[batchIndex][layer-1])


        #       print(layer)

                # Lambda step
                if layer == 0:
                    lamdas[batchIndex][layer] = lamdaStep(prevLamdas[batchIndex][layer], prevYs[batchIndex][layer], prevWs[layer], splitAndBiasedXs)
                else:
    #               print(np.array(prevWs[layer]).shape)

                    lamdas[batchIndex][layer] = lamdaStep(prevLamdas[batchIndex][layer], prevYs[batchIndex][layer], \
                        prevWs[layer], splitAndBiasedPrevZs)

        #       print("lamda",layer, lamdas[layer])

                if layer == 0:
                    lamda0s.append(lamdas[batchIndex][layer][0][0])
                if layer == 1:
                    lamda1s.append(lamdas[batchIndex][layer][0][0])


                # Mu step
        #       print(layer)
        #       print(len(mus))
        #       print(len(prevMus))
        #       print(len(prevZs))
        #       print(len(fOfPrevYs))
                mus[batchIndex][layer] = muStep(prevMus[batchIndex][layer], prevZs[batchIndex][layer], fOfPrevYs[batchIndex][layer])

        #       print("mu", layer, mus[layer])

                if layer == 0:
                    mu0s.append([mus[batchIndex][layer]][0][0][0])
                if layer == 1:
                    mu1s.append([mus[batchIndex][layer]][0][0][0])


                # Z step
                if layer == numLayers - 1:
                    Zs[batchIndex][layer] = finalZStep(T, mus[batchIndex][layer], fOfPrevYs[batchIndex][layer], prevZs[batchIndex][layer])
                else:
                    Zs[batchIndex][layer] = ZStepNotLast(mus[batchIndex][layer], fOfPrevYs[batchIndex][layer], \
                        Ws[layer+1], lamdas[batchIndex][layer+1], Ys[batchIndex][layer+1], splitAndBiasX(prevZs[batchIndex][layer]))

        #       print("Y",layer, Ys[layer])
        #       print("Z",layer, Zs[layer])

                if layer == 0:
                    Z0s.append([Zs[batchIndex][layer]][0][0][0])
                if layer == 1:
                    Z1s.append([Zs[batchIndex][layer]][0][0][0])



                # Y step
                if layer == 0:
                    Ys[batchIndex][layer] = YStep(prevWs[layer], splitAndBiasedXs, mus[batchIndex][layer], fPrimeOfPrevYs[batchIndex][layer], \
                        lamdas[batchIndex][layer], Zs[batchIndex][layer], fOfPrevYs[batchIndex][layer])
                else:
                    Ys[batchIndex][layer] = YStep(prevWs[layer], splitAndBiasX(prevZs[batchIndex][layer-1]), mus[batchIndex][layer], fPrimeOfPrevYs[batchIndex][layer], \
                        lamdas[batchIndex][layer], Zs[batchIndex][layer], fOfPrevYs[batchIndex][layer])         

        #       print("Y",layer, Ys[layer])

                if layer == 0:
                    Y0s.append([Ys[batchIndex][layer]][0][0][0])
                if layer == 1:
                    Y1s.append([Ys[batchIndex][layer]][0][0][0])


                # W step
                if layer == 0:
                    Ws[layer] = jointWStepNew(lamdas[batchIndex][layer], prevWs[layer], Ys[batchIndex][layer], splitAndBiasedXs)
                else:
                    Ws[layer] = jointWStepNew(lamdas[batchIndex][layer], prevWs[layer], Ys[batchIndex][layer], splitAndBiasX(prevZs[batchIndex][layer-1]))

        #       print("W",layer, Ws[layer])

                if layer == 0:
                    W0s.append([Ws[layer]][0][0][0])
                if layer == 1:
                    W1s.append([Ws[layer]][0][0][0])
                if layer == 2:
                    W2s.append([Ws[layer]][0][0][0])



        #   mu = muStep(prevMu, prevZ, fOfPrevY)
        #   Z = ZStep(T, mu, prevZ, fOfPrevY)
        #   Y = YStep(prevW, X, mu, fPrimeOfPrevY, lamda, Z, fOfPrevY)  
        #   W = WStepPerfect(lamda, prevW, Y, X)

    #       if i % 200 == 0:
            if False:
                output = evaluateNetwork(Ws, splitAndBiasedXs, showLayers=True)
                p.matshow(T)
                p.colorbar()
                p.show()

                p.matshow(T - output)
                p.title("output - T")
                p.colorbar()
                p.show()
            else:
                output = evaluateNetwork(Ws, splitAndBiasedXs, showLayers=False)

            if movie:
                p.clf()
                p.matshow(Ws[0][0])
                p.colorbar()
                p.savefig("nn_movie/W0_" + moniker + "_frame_" + padIntegerWithZeros(counter,3) + ".png")
                p.close()

                p.clf()
                p.matshow(Ws[1][0])
                p.colorbar()
                p.savefig("nn_movie/W1_" + moniker + "_frame_" + padIntegerWithZeros(counter,3) + ".png")
                p.close()

            counter += 1

        #   print("output", output)
        #   print("T", T)

            diffMat = T - output

    #       print(diffMat)

        #   print(T)
        #   print(output)

            error = np.trace(np.dot(np.transpose(diffMat), diffMat))

            errors.append(error)
            logErrors.append(softLog(error))

        numCorrect = 0


#       p.matshow(X)
#       p.colorbar()
#       p.show()                
        output = evaluateNetwork(Ws, splitAndBiasedXs, showLayers=False)

    #   p.matshow(output)
    #   p.colorbar()
    #   p.show()

        for batchIndex in range(numBatches):
            X, T = selectSubset(overallX, overallT, batchIndex, numExamples)    
        #   X = augmentWithBias(X)
        #   X, T = selectRandomSubset(overallX, overallT, numExamples)  
            voteT = vote(T)

            splitAndBiasedXs = splitAndBiasX(X)

            output = evaluateNetwork(Ws, splitAndBiasedXs)
            voteOutput = vote(output)

    #       print(output)

            for i in range(numExamples):
            #   print(np.transpose(output)[i])
            #   print(np.argmax(np.transpose(output)[i]))

            #   print(np.transpose(T)[i])
            #   print(np.argmax(np.transpose(T)[i]))

            #   print("System's guess:", np.argmax(np.transpose(output)[i]), \
            #       "Correct:", np.argmax(np.transpose(T)[i]))

    #           print(voteOutput[i], voteT[i])

                if voteOutput[i] == voteT[i]:
                    numCorrect += 1

        accuracy = numCorrect / totalNumExamples

        print("Training Accuracy:", accuracy)

        overallXtest, overallTtest = pickle.load(open("little_mnist_one_minus_one_test.p", "rb"))

        overallTtest = repeatT(overallTtest, 5)
        overallXtest = augmentWithBias(overallXtest)

        #testOutput = evaluateNetwork(Ws, augmentWithBias(Xtest))

        numCorrect = 0
        for batchIndex in range(numBatches):
        #   X, Ttest = selectRandomSubset(Xtest, Ttest, numExamples)    
            X, Ttest = selectSubset(overallXtest, overallTtest, batchIndex, numExamples)    
        #   X = augmentWithBias(X)

            splitAndBiasedXs = splitAndBiasX(X)

            testOutput = evaluateNetwork(Ws, splitAndBiasedXs)

            voteTtest = vote(Ttest)
            voteTestOutput = vote(testOutput)

        #   print(batchIndex)
        #   print("a", testOutput.shape)
        #   print("b", Ttest.shape)


            for i in range(numExamples):
        #       print("System's guess:", np.argmax(np.transpose(testOutput)[i]), \
        #           "Correct:", np.argmax(np.transpose(Ttest)[i]))

        #       p.matshow(np.reshape(np.transpose(X)[i], (7,7)))
        #       p.show()



                if voteTestOutput[i] == voteTtest[i]:
                    numCorrect += 1

        accuracy = numCorrect / totalNumExamples
        print("Test Accuracy:", accuracy)


        #   p.matshow(np.reshape(np.transpose(X)[i], (7,7)))
        #   p.show()



        #print("teacher weights", teacherWs)
        #print("student weights", Ws)

        #print("teacher output", evaluateNetwork(teacherWs, X))
        #print("student output", evaluateNetwork(Ws, X))
        #print(T)
        #print(output)
        #print(diffMat)

        #print(Ys)

#       pickle.dump(logErrors, open("log_errors.p", "wb"))
        logErrors2 = pickle.load(open("log_errors.p", "rb"))
        p.plot(lamda0s, label='lamda0s')
        p.plot(mu0s, label='mu0s')
        p.plot(W0s, label='W0s')
        p.plot(Y0s, label='Y0s')
        p.plot(Z0s, label='Z0s')
        p.plot(lamda1s, label='lamda1s')
        p.plot(mu1s, label='mu1s')
        p.plot(W1s, label='W1s')
        p.plot(Y1s, label='Y1s')
        p.plot(Z1s, label='Z1s')
        #p.plot(W2s, label="W2s")

        p.legend()
        p.show()

        print(time.time() - t)
        p.plot(logErrors)
        p.plot(logErrors2)
        p.ylabel("Log error")
        p.xlabel("Iteration")
        p.show()    


if INTERESTING_PLOTS:

    xs = np.linspace(-10, 10, 100)

    p.plot(xs, [softPlusPrimeMaker(1)(x) for x in xs])
    p.plot(xs, [softPlusPrimeMaker2(1)(x) for x in xs])
    p.show()


    p.plot(xs, [softPlusMaker(0.3)(x) for x in xs])
    p.plot(xs, [softPlusMaker(1)(x) for x in xs])
    p.plot(xs, [softPlusMaker(3)(x) for x in xs])


    p.plot(xs, [softSignZeroOne3(x) for x in xs])
    p.plot(xs, [sigmoid(x) for x in xs])
    p.show()

    print(softSignZeroOnePrime(0))

    p.plot(xs, [softSignZeroOnePrime3(x) for x in xs])
    p.plot(xs, [sigmoidPrime(x) for x in xs])
    p.show()

    p.plot(xs, [softRelu(x) for x in xs])
    p.plot(xs, [relu(x) for x in xs])
    p.show()

    p.plot(xs, [softReluPrime(x) for x in xs])
    p.plot(xs, [reluPrime(x) for x in xs])
    p.show()

    p.plot(xs, [softSignAbsPrime(x) for x in xs])
    p.show()

if MULTI_SGD:

#f, fPrime, fName = sigmoid, sigmoidPrime, "sigmoid"
#f, fPrime, fName = sigmoid, softSignZeroOnePrime, "sigmoid"
#f, fPrime, fName = softSignZeroOne, sigmoid, "sigmoid"
#f, fPrime, fName = softSignZeroOne, softSignZeroOnePrime, "sigmoid"

#   f, fPrime = relu, reluPrime
#    funcs = [(sigmoid, sigmoidPrime), 
#             (sigmoid, softSignZeroOnePrime3)]
    #        (softSignZeroOne3, sigmoidPrime),
    #        (softSignZeroOne3, softSignZeroOnePrime3)]

#   funcs = [(sigmoid, sigmoid, sigmoidPrime), 
#            (sigmoid, softSignZeroOne3, softSignZeroOnePrime3),
#            (softSignZeroOne3, sigmoid, sigmoidPrime),
#            (softSignZeroOne3, softSignZeroOne3, softSignZeroOnePrime3)]

#    pickleFilenames = ["log_errors_grad_sig_sig_sm.p", 
 #                      "log_errors_grad_sig_soft_sm.p"]
#                      "log_errors_grad_soft_sig_sm.p",
#                      "log_errors_grad_soft_soft_sm.p"]


#   funcs = [(sigmoid, sigmoidPrime)]
    a = 1
    funcs = [(leakyReluMaker(0.5), leakyReluPrimeMaker(0.5))]
#    funcs = [(np.vectorize(line), np.vectorize(linePrime))]
#   pickleFilenames = ["log_errors_grad_sig_sig_sm.p"]

    for i in range(len(funcs)):
        f, fPrime = funcs[i]
#       f, dummyF, fPrime = funcs[i]

        logErrors, counter, _ = doSGD(f, fPrime, numIter=1000, learningRate=1, loadFromPickle=False, movie=False, softmaxFinalLayer=True)

#        pickle.dump(logErrors, open(pickleFilenames[i], "wb"))
    
#    for pickleFilename in pickleFilenames:
 #       p.plot(pickle.load(open(pickleFilename, "rb")), label=pickleFilename)

    p.plot(logErrors)
    p.legend()
    p.show()

if MULTI_K_TINY_SGD:
#f, fPrime, fName = sigmoid, sigmoidPrime, "sigmoid"
#f, fPrime, fName = sigmoid, softSignZeroOnePrime, "sigmoid"
#f, fPrime, fName = softSignZeroOne, sigmoid, "sigmoid"
#f, fPrime, fName = softSignZeroOne, softSignZeroOnePrime, "sigmoid"

#   f, fPrime = relu, reluPrime
#    k1s = [0.25, 0.5, 1, 2, 4, 8]
    k1 = 1
#    k2s = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    k2s = [2**logK for logK in range(-20, 3)]
    lrs = [2**logK for logK in range(10)]
#    lrs = [1]

    errorMat = []
#    for k1 in [1/64,1/32,0.0625, 0.125, 0.25, 0.5, 1, 2,4,8,16]:
    for lr in lrs:
        print(lr)
        errorMat.append([])
        for k2 in k2s:

#        for k2 in [1/64,1/32,0.0625, 0.125, 0.25, 0.5, 1, 2]:
#            plotFunc(sigmoidMaker(k1))
#            plotFunc(sigmoidPrimeMaker(k2))
#            print(k2)
#            f, fPrime = quadMaker(k1), quadPrimeMaker(k2)
            Ws = [np.array([[0., 1.]])]
            Ys = [np.array([[0.]])]
            Zs = [np.array([[0.]])]
            overallT = np.array([[0.5]])
            overallX = np.array([[0.]])



            startingVals = (Ws, Ys, Zs, overallT, overallX) 


            f, fPrime = sigmoidMaker(k1), sigmoidPrimeNormalizedMaker(k2)
#            f, fPrime = sigmoidMaker(k1), sigmoidPrimeMaker(k2)
            logErrors, updates, counter = doSGD(f, fPrime, numIter=40, learningRate=lr, 
                startingVals=None, loadFromPickle=True, movie=False, softmaxFinalLayer=False)

#            if lr == 16 and k2 < 0.001:
 #               print(updates)

            p.plot(logErrors, label=str(k1) + " " + str(k2))
#            p.plot(logErrors)
#            p.show()
            errorMat[-1].append(max(logErrors[-1], -20))
#            p.plot([softLog(abs(u)) for u in updates], label=str(k1) + " " + str(k2))
#            p.plot([u for u in updates], label=str(k1) + " " + str(k2))
#        p.legend()
#        p.show()



    p.xlabel("k2")
    p.ylabel("k1") 
    ax=p.gca()
    ax.set_xticklabels([""] + lrs)
    ax.set_yticklabels([""] + k2s)
  
    p.matshow(errorMat)

    p.colorbar()
    p.show()

if MULTI_K_LITTLE_SGD:
#f, fPrime, fName = sigmoid, sigmoidPrime, "sigmoid"
#f, fPrime, fName = sigmoid, softSignZeroOnePrime, "sigmoid"
#f, fPrime, fName = softSignZeroOne, sigmoid, "sigmoid"
#f, fPrime, fName = softSignZeroOne, softSignZeroOnePrime, "sigmoid"

#   f, fPrime = relu, reluPrime
    k1s = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]#, 32, 64, 128, 256, 512, 1024]
    k2s = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
#    k2s = [1./2**logK for logK in range(-10, 0)]
#    k2s = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    n = 3
    numExamples = 5

#    k1s = [1]
#    k2s = [1]
    
    old = False
    if old:

        errorMat = []
    #    for k1 in [1/64,1/32,0.0625, 0.125, 0.25, 0.5, 1, 2,4,8,16]:
    #    for k1 in k1s[::-1]:
    #        print(k1)
        for k2 in k2s[::-1]:
            print(k2)
            errorMat.append([])
            for k1 in k1s[::-1]:
    #        for k2 in k2s[::-1]:

    #        for k2 in [1/64,1/32,0.0625, 0.125, 0.25, 0.5, 1, 2]:
    #            plotFunc(sigmoidMaker(k1))
    #            plotFunc(sigmoidPrimeMaker(k2))
    #            print(k2)
    #            f, fPrime = quadMaker(k1), quadPrimeMaker(k2)
    #            f, fPrime = sigmoidMaker(k1), sigmoidPrimeMaker(k2)
                f, fPrime = softPlusMaker(k1), softPlusPrimeMaker(k2)
    #            f, fPrime = None, None
                logErrors, updates, counter = doSGD(f, fPrime, numIter=200, learningRate=1, loadFromPickle=True, movie=False, softmaxFinalLayer=True,
                    k1=1, k2=1)

                p.plot(logErrors, label=str(k1) + " " + str(k2))
    #            p.show()
                errorMat[-1].append(max(logErrors[-1], -10))
    #            p.plot([softLog(abs(u)) for u in updates], label=str(k1) + " " + str(k2))
    #            p.plot([u for u in updates], label=str(k1) + " " + str(k2))
            p.legend()
            p.show()




        p.matshow(errorMat)
        p.xlabel("k2")
        p.ylabel("k1")
        ax=p.gca()
        ax.set_xticklabels([""] + k1s)
        ax.set_yticklabels([""] + k2s)
        p.colorbar()
        p.show()    

    compareManySGDs(k1s, k2s, n, numExamples)

if ADMM_WITH_SOFTMAX:
    f, fPrime = leakyReluMaker(0.5), leakyReluPrimeMaker(0.5)
    doADMMLearning(numLayers, numBatches, numExamples, numCohorts, layerSizes, f, fPrime, True)


#               print(k1, k2, logErrors[-1])



#               pickle.dump(logErrors, open(pickleFilenames[i], "wb"))

#           for pickleFilename in pickleFilenames:
#               p.plot(pickle.load(open(pickleFilename, "rb")), label=pickleFilename)

#           p.legend()
#           p.show()

