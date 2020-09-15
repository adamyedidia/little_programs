from __future__ import division
import sys
import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as p

n = int(sys.argv[1])
#mean = sqrt(n)
MAX_GUESS_POSSIBLE = n

GAMES_PER_EPOCH = 1000

#GAMES_TO_PLAY = 10000

numEpochs = 100


def incrDict(dic, key):
    if key in dic:
        dic[key] += 1
    else:
        dic[key] = 1

def monotonize(dist):
    for i, val in enumerate(dist[1:]):
        prevVal = dist[i-1]
        if val > prevVal and i > 0:
            print(i, val, prevVal)
            print(dist)
            if (val + prevVal) % 2 == 1:
                dist[i-1] = int((val + prevVal)/2) + 1
                dist[i] = int((val + prevVal)/2)
            else:
                dist[i-1] = int((val + prevVal)/2)
                dist[i] = int((val + prevVal)/2)
            print(dist)

def indexIntoDist(index, dist):
    sumValsSoFar = 0

#    print(index, dist)

    for i, val in enumerate(dist):
        sumValsSoFar += val
        if sumValsSoFar > index:
#            print(i+1)
            return i+1

# dist is a (list, total) pair
def runGame(dist):
    l = dist[0]
    total = dist[1]

    # for each player, guess a number
    guessCountDict = {}
    maxGuess = 0
    for _ in range(n):
        guessIndex = random.randint(0, total-1)


        guess = indexIntoDist(guessIndex, l)

#        print(guessIndex, guess)

#        guess = np.random.geometric(1/mean)
        incrDict(guessCountDict, guess)
        maxGuess = max(maxGuess, guess)

    for i in range(maxGuess+1):
        if i in guessCountDict:
            if guessCountDict[i] == 1:
                return i

    # no winner
    return 0

dist = ([1]*MAX_GUESS_POSSIBLE, MAX_GUESS_POSSIBLE)



winnersHistogram = {}

for epoch in range(numEpochs):

    print(epoch)
    GAMES_TO_PLAY = (epoch + 5) * GAMES_PER_EPOCH

#    meansOverTime = [mean]

#    winnersHistogram = {}

    recentWinnersHistogram = {}

    maxOutcome = 0

    for numGamesPlayed in range(GAMES_TO_PLAY):
        outcome = runGame(dist)
        if outcome != 0:
#            mean = numGamesPlayed*mean/(numGamesPlayed+1) + outcome/(numGamesPlayed+1)

            maxOutcome = max(outcome, maxOutcome)

            incrDict(winnersHistogram, outcome)
            incrDict(recentWinnersHistogram, outcome)

#            print(outcome, "wins!")
#            print("new mean", mean)
#            meansOverTime.append(mean)

        else:
#            print("Tie game")
            pass

#    p.plot(meansOverTime)
 #   p.show()

    winnersCountList = []
    recentWinnersCountList = []

#    print(winnersHistogram)

    for i in range(1, MAX_GUESS_POSSIBLE+1):
        if i in winnersHistogram:
            winnersCountList.append(winnersHistogram[i])
        else:
            winnersCountList.append(0)

        if i in recentWinnersHistogram:
            recentWinnersCountList.append(recentWinnersHistogram[i])
        else:
            recentWinnersCountList.append(0)

    numWinners = sum(winnersCountList)
    numRecentWinners = sum(recentWinnersCountList)

    if epoch % 10 == 0:
        p.plot(range(1, MAX_GUESS_POSSIBLE+1), [i/dist[1] for i in dist[0]])
        p.plot(range(1, MAX_GUESS_POSSIBLE+1), [i/numRecentWinners for i in recentWinnersCountList])
    #    p.plot(range(1, maxOutcome+1), [GAMES_TO_PLAY*(1./mean)**i for i in range(1, maxOutcome+1)])
        p.show()

    distList = [winnersCountList[i] for i in range(MAX_GUESS_POSSIBLE)]

#    p.plot(distList)
#    p.show()

    print(winnersCountList)


#    distListSorted = sorted(distList)[::-1]
#    monotonize(distList)

#    p.plot(distListSorted)
#    p.show()

    dist = (distList, sum(distList))



#    print(dist[1], sum(dist[0]))
#    assert dist[1] == sum(dist[0])

