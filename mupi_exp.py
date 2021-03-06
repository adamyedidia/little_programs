from __future__ import division
import sys
import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as p

n = int(sys.argv[1])
mean = 2.05
GAMES_TO_PLAY = 100000

def incrDict(dic, key):
    if key in dic:
        dic[key] += 1
    else:
        dic[key] = 1

def changeDict(dic, key, amount):
    if key in dic:
        dic[key] += amount
    else:
        dic[key] = amount

def runGame():
    # for each player, guess a number
    guessCountDict = {}
    maxGuess = 0
    for _ in range(n):
        guess = np.random.geometric(1/mean)
        incrDict(guessCountDict, guess)
        maxGuess = max(maxGuess, guess)

    print guessCountDict

    for i in range(maxGuess+1):
        if i in guessCountDict:
            if guessCountDict[i] == 1:
                return i, guessCountDict, maxGuess

    # no winner
    return 0, guessCountDict, maxGuess

meansOverTime = [mean]

scoreHistogram = {}

maxMaxGuess = 0

def weightedAverage(histogram):
    summ = 0
    total = 0
    for i in histogram:
        summ += i*histogram[i]
        total += histogram[i]

    return summ/total

def average(histogram):
    summ = 0
    for i in histogram:
        summ += i*histogram[i]

    return summ

def findBestScoreInHistogram(histogram):
    bestScore = -float("Inf")
    bestIndex = None

    for i in histogram:
        if histogram[i] > bestScore:
            bestScore = histogram[i]
            bestIndex = i

    return bestIndex

def findBestPartialSumInHistogram(histogram):
    bestSum = -float("Inf")
    bestIndex = None

    sumSoFar = 0

    for i in histogram:
        sumSoFar += histogram[i]
        if sumSoFar > bestScore:
            bestScore = sumSoFar
            bestIndex = i

    return bestIndex


for numGamesPlayed in range(GAMES_TO_PLAY):
    outcome, guessCountDict, maxGuess = runGame()
    if outcome != 0:

#        mean = 2

        maxMaxGuess = max(maxGuess, maxMaxGuess)

        for i in guessCountDict:
            if i == outcome:
                changeDict(scoreHistogram, i, 1)
            else:
                changeDict(scoreHistogram, i, -guessCountDict[i]/(n-1))

        total = 0
        for i in guessCountDict:
            if i == outcome:
                total += i
            else:
                total -= guessCountDict[i]*i/(n-1)

        print total

        mean += total/(numGamesPlayed+1)

#        if scoreHistogram[1] > 0:
#            mean -= 1/(numGamesPlayed+1)
#        else:
#            mean += 1/(numGamesPlayed+1)


        #mean -= scoreHistogram[1]/(numGamesPlayed+1)

        bestScore = findBestScoreInHistogram(scoreHistogram)


#        mean = numGamesPlayed*mean/(numGamesPlayed+1) + bestScore/(numGamesPlayed+1)

        print outcome, "wins!"
        print "new mean", mean
        meansOverTime.append(mean)
#        mean = weightedAverage(winnersHistogram)


    else:
        print "Tie game"

p.plot(meansOverTime)
p.show()

scoreList = []
for i in range(1, maxMaxGuess+1):
    if i in scoreHistogram:
        scoreList.append(scoreHistogram[i])
    else:
        scoreList.append(0)

print "weighted average", average(scoreHistogram)

print scoreHistogram
print scoreList


p.plot(range(1, maxMaxGuess+1), scoreList, "r-")
#p.plot(range(1, maxMaxGuess+1), [GAMES_TO_PLAY*(1./mean)**i for i in range(1, maxMaxGuess+1)], "b-")
p.show()
