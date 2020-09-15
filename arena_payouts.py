import numpy as np
import matplotlib.pyplot as p

MAKE_BIGFIG = True
MAKE_GAMES_PER_KILOGEM_FIG = False
MAKE_PACKS_PER_KILOGEM_FIG = False
MAKE_KILOGEM_PER_GAME_FIG = False
MAKE_PACK_PER_GAME_FIG = False
MAKE_PACK_PER_GAME_FIG_COUNTING_GEMS = False

newTraditionalDraftEntryFee = 1500
newTraditionalDraftGameType = "Bo3"
newTraditionalDraftPayouts = {
	(0, 3): (0, 1),
	(1, 2): (0, 1),
	(2, 1): (1000, 4),
	(3, 0): (3000, 6),
}
newTraditionalDraftName = "New traditional draft"
newTraditionalDraftColor = "blue"
newTraditionalDraft = (newTraditionalDraftEntryFee, 
	newTraditionalDraftGameType, 
	newTraditionalDraftPayouts,
	newTraditionalDraftName, 
	newTraditionalDraftColor)

premierDraftEntryFee = 1500
premierDraftGameType = "Bo1"
premierDraftPayouts = {
	(0, 3): (50, 1),
	(1, 3): (100, 1),
	(2, 3): (250, 2),
	(3, 3): (1000, 2),
	(4, 3): (1400, 3),
	(5, 3): (1600, 4),
	(6, 3): (1800, 5),
	(7, 2): (2200, 6),
	(7, 1): (2200, 6),
	(7, 0): (2200, 6),
}
premierDraftName = "Premier draft"
premierDraftColor = "red"
premierDraft = (premierDraftEntryFee, 
	premierDraftGameType,
	premierDraftPayouts,
	premierDraftName,
	premierDraftColor)

quickDraftEntryFee = 750
quickDraftGameType = "Bo1"
quickDraftPayouts = {
	(0, 3): (50, 1),
	(1, 3): (100, 1),
	(2, 3): (200, 1),
	(3, 3): (300, 1),
	(4, 3): (450, 1),
	(5, 3): (650, 1),
	(6, 3): (850, 1),
	(7, 2): (950, 2),
	(7, 1): (950, 2),
	(7, 0): (950, 2),
}
quickDraftName = "Quick draft"
quickDraftColor = "green"
quickDraft = (quickDraftEntryFee, 
	quickDraftGameType, 
	quickDraftPayouts,
	quickDraftName,
	quickDraftColor)

oldTraditionalDraftEntryFee = 1500
oldTraditionalDraftGameType = "Bo3"
oldTraditionalDraftPayouts = {
	(0, 2): (0, 1),
	(1, 2): (0, 2),
	(2, 2): (800, 3),
	(3, 2): (1500, 4),
	(4, 2): (1800, 5),
	(5, 1): (2100, 6),
	(5, 0): (2100, 6),
}
oldTraditionalDraftName = "Old traditional draft"
oldTraditionalDraftColor = "magenta"
oldTraditionalDraft = (oldTraditionalDraftEntryFee,
	oldTraditionalDraftGameType,
	oldTraditionalDraftPayouts,
	oldTraditionalDraftName, 
	oldTraditionalDraftColor)

def augmentedPayouts(draft, gameWinRate):
	entryFee = draft[0]
	gameType = draft[1]
	payouts = draft[2]
	gamesPerMatch = numGamesPerMatch(gameType, gameWinRate)

	augPayouts = {}

	for record in payouts:
		numWins = record[0]
		numLosses = record[1]
		numMatches = numWins + numLosses
		numGames = numMatches*gamesPerMatch

		payoutForRecord = payouts[record]
		grossGemsPayout = payoutForRecord[0]
		netGemsPayout = grossGemsPayout - entryFee

		packsPayout = payoutForRecord[1]

		augPayouts[record] = (netGemsPayout, packsPayout, numGames)

	return augPayouts	


def numGamesPerMatch(gameType, gameWinRate):
	if gameType == "Bo1":
		return 1
	elif gameType == "Bo3":
		twoGameMatchProbability = (gameWinRate**2 + (1-gameWinRate)**2)

		return twoGameMatchProbability*2 + (1-twoGameMatchProbability)*3
	else:
		print("typo alert!")
		raise

gameWinRates = np.linspace(0, 1, 101)
#gameWinRates = np.linspace(0.2, 0.8, 101)
#gameWinRates = np.linspace(0.35, 0.65, 101)


def matchifyWinRate(gameType, gameWinRate):
	if gameType == "Bo1":
		return gameWinRate
	elif gameType == "Bo3":
		probTwoOh = gameWinRate**2
		probTwoOne = gameWinRate*(1-gameWinRate)*gameWinRate + (1-gameWinRate)*gameWinRate*gameWinRate
		return probTwoOh + probTwoOne
	else:
		print("type alert!")
		raise

def expectedThing(draft, gameWinRate, thingFunc):
	augPayouts = augmentedPayouts(draft, gameWinRate)
	matchWinRate = matchifyWinRate(draft[1], gameWinRate)

	return expectedThingRecursive(augPayouts, matchWinRate, thingFunc, 0, 0)

def expectedThingRecursive(augPayouts, matchWinRate, thingFunc, numWins, numLosses):
#	print(numWins, numLosses)

	if (numWins, numLosses) in augPayouts:
		return thingFunc(augPayouts[(numWins, numLosses)])

	else:
		return matchWinRate*expectedThingRecursive(augPayouts, matchWinRate, \
			thingFunc, numWins+1, numLosses) + \
			(1-matchWinRate)*expectedThingRecursive(augPayouts, matchWinRate, \
			thingFunc, numWins, numLosses+1)

def gemFunc(payout):
	return payout[0]

def packFunc(payout):
	return payout[1]+3

def gameFunc(payout):
	return payout[2]

def kilogemFunc(payout):
	return payout[0]/1000

def posRatio(x, y):
	if y <= 0:
		return 10000
	else:
		return x/y

def plotThing(draft, gameWinRates, thing):
	thingFunc, thingName, thingLineStyle = thing

	draftName = draft[3]
	draftColor = draft[4]

#	if not ((draftName == "Premier draft" or draftName == "Old traditional draft") and thingName == "games"):
#		return

	p.plot([gameWinRate*100 for gameWinRate in gameWinRates], [expectedThing(draft, gameWinRate, thingFunc) \
		for gameWinRate in gameWinRates], color=draftColor, linestyle=thingLineStyle, 
		label=draftName + " " + thingName)

gems = (gemFunc, "gems", "dashdot")
packs = (packFunc, "packs", "dashed")
games = (gameFunc, "games", "dotted")
kilogems = (kilogemFunc, "kilogems", "solid")

def plotGamesPerKilogem(draft, gameWinRates):
	draftName = draft[3]
	draftColor = draft[4]
	p.plot([gameWinRate*100 for gameWinRate in gameWinRates], [posRatio(expectedThing(draft, gameWinRate, gameFunc), \
		-expectedThing(draft, gameWinRate, kilogemFunc)) for gameWinRate in gameWinRates], color=draftColor, 
		linestyle="solid", label=draftName)

def plotPacksPerKilogem(draft, gameWinRates):
	draftName = draft[3]
	draftColor = draft[4]
	p.plot([gameWinRate*100 for gameWinRate in gameWinRates], [posRatio(expectedThing(draft, gameWinRate, packFunc), \
		-expectedThing(draft, gameWinRate, kilogemFunc)) for gameWinRate in gameWinRates], color=draftColor, 
		linestyle="solid", label=draftName)

def plotKilogemsPerGame(draft, gameWinRates):
	draftName = draft[3]
	draftColor = draft[4]
	p.plot([gameWinRate*100 for gameWinRate in gameWinRates], [posRatio(expectedThing(draft, gameWinRate, kilogemFunc), \
		expectedThing(draft, gameWinRate, gameFunc)) for gameWinRate in gameWinRates], color=draftColor, 
		linestyle="solid", label=draftName)

def plotPacksPerGame(draft, gameWinRates):
	draftName = draft[3]
	draftColor = draft[4]
	p.plot([gameWinRate*100 for gameWinRate in gameWinRates], [posRatio(expectedThing(draft, gameWinRate, packFunc), \
		expectedThing(draft, gameWinRate, gameFunc)) for gameWinRate in gameWinRates], color=draftColor, 
		linestyle="solid", label=draftName)

def plotPacksPerGameCountingGems(draft, gameWinRates):
	draftName = draft[3]
	draftColor = draft[4]
	p.plot([gameWinRate*100 for gameWinRate in gameWinRates], [posRatio(expectedThing(draft, gameWinRate, packFunc) + \
		5*expectedThing(draft, gameWinRate, kilogemFunc), \
		expectedThing(draft, gameWinRate, gameFunc)) for gameWinRate in gameWinRates], color=draftColor, 
		linestyle="solid", label=draftName)


#	p.yrange


drafts = [newTraditionalDraft, premierDraft, quickDraft, oldTraditionalDraft]
things = [kilogems, packs, games]

if MAKE_BIGFIG:

	p.title("How much of each thing do you get?")
	p.ylabel("Amount of thing")
	p.xlabel("Your win rate (%)")

	for thing in things:
		for draft in drafts:
			plotThing(draft, gameWinRates, thing)

	p.grid()
	p.legend()
	p.show()

if MAKE_GAMES_PER_KILOGEM_FIG:
	p.title("How many games do you play per kilogem you spend?")
	p.ylabel("Games per kilogem")
	p.xlabel("Your win rate (%)")

	p.ylim([0, 30])

	for draft in drafts:
		plotGamesPerKilogem(draft, gameWinRates)

#	p.axvline(x=50, color="black")
	
	p.grid()
	p.legend()
	p.show()

if MAKE_PACKS_PER_KILOGEM_FIG:
	p.title("How many packs do you get per kilogem you spend?")
	p.ylabel("Packs per kilogem")
	p.xlabel("Your win rate (%)")

	p.ylim([0, 30])

	for draft in drafts:
		plotPacksPerKilogem(draft, gameWinRates)

#	p.axvline(x=50, color="black")
	p.plot([gameWinRate*100 for gameWinRate in gameWinRates], \
		[5 for _ in gameWinRates], color="black", label="The store")

	p.grid()
	p.legend()
	p.show()

if MAKE_KILOGEM_PER_GAME_FIG:
	p.title("How many kilogems do you get per game you play?")
	p.ylabel("Kilogems per game")
	p.xlabel("Your win rate (%)")

	for draft in drafts:
		plotKilogemsPerGame(draft, gameWinRates)

	p.grid()
	p.legend()
	p.show()

if MAKE_PACK_PER_GAME_FIG:
	p.title("How many packs do you get per game you play?")
	p.ylabel("Packs per game")
	p.xlabel("Your win rate (%)")

	for draft in drafts:
		plotPacksPerGame(draft, gameWinRates)

	p.grid()
	p.legend()
	p.show()

if MAKE_PACK_PER_GAME_FIG_COUNTING_GEMS:
	p.title("How many packs do you get per game you play\n(converting all gems to packs)?")
	p.ylabel("Packs per game")
	p.xlabel("Your win rate (%)")

	for draft in drafts:
		plotPacksPerGameCountingGems(draft, gameWinRates)

	p.grid()
	p.legend()
	p.show()	


#p.plot([numGamesPerMatch("Bo3", winRate) for winRate in winRates])
#p.show()




#print(augmentedPayouts(newTraditionalDraft, 0.6))

#def expectedThing(draft, winRate):


