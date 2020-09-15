import random

numCards = 5
playsetSize = 4


numSamples = 1000

agreements = 0

for i in range(numSamples):
	print(i)
	cardsOpened = [0]*numCards
	cardOpenHistory = []

	for j in range(numCards*playsetSize):
		done = False

		while not done:

			cardToOpen = random.randint(0, numCards-1)

			if cardsOpened[cardToOpen] < playsetSize:
				done = True
				cardsOpened[cardToOpen] += 1

		cardOpened = cardToOpen
		cardOpenHistory.append(cardOpened)
#	print(cardsOpened)

#	if cardOpenHistory[-1] == cardOpenHistory[-2]:
#		agreements += 1

	if cardOpenHistory[-2] == cardOpenHistory[-3]:
		agreements += 1
#	if cardOpenHistory[0] == cardOpenHistory[1]:
#		agreements += 1

print(agreements/numSamples)

