from __future__ import division
import sys
import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as p

n = int(sys.argv[1])
s = 1.3
GAMES_TO_PLAY = 10000

def incrDict(dic, key):
    if key in dic:
        dic[key] += 1
    else:
        dic[key] = 1

def runGame():
    # for each player, guess a number
    guessCountDict = {}
    maxGuess = 0
    for _ in range(n):
        guess = np.random.zipf(s)
        if guess > 1000:
            guess = random.randint(1001, 10000) # What a hack!!
        incrDict(guessCountDict, guess)
        maxGuess = max(maxGuess, guess)

    for i in range(maxGuess):
        if i in guessCountDict:
            if guessCountDict[i] == 1:
                return i

    # no winner
    return 0

winnersHistogram = {}

maxOutcome = 0

for numGamesPlayed in range(GAMES_TO_PLAY):
    outcome = runGame()
    if outcome != 0:
        maxOutcome = max(outcome, maxOutcome)

        incrDict(winnersHistogram, outcome)

        print outcome, "wins!"

    else:
        print "Tie game"

#p.plot(meansOverTime)
#p.show()

winnersCountList = []
for i in range(maxOutcome):
    if i in winnersHistogram:
        winnersCountList.append(winnersHistogram[i])
    else:
        winnersCountList.append(0)

p.plot(range(1, maxOutcome+1), winnersCountList)
#p.plot(range(1, maxOutcome+1), [GAMES_TO_PLAY*(1./mean)**i for i in range(1, maxOutcome+1)])
p.show()
