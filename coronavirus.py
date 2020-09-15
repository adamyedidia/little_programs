import matplotlib.pyplot as p

STARTING_NUM_INFECTED = 10
INFECTION_RATE = 1.2
DISEASE_DURATION = 21
TOTAL_POPULATION_SIZE = 4e6
FRACTION_VULNERABLE = 0.5
QUARANTINE_FRACTION = 0.001
POPULATION_SIZE = FRACTION_VULNERABLE*TOTAL_POPULATION_SIZE
QUARANTINE_THRESHOLD = QUARANTINE_FRACTION*TOTAL_POPULATION_SIZE


numSick = STARTING_NUM_INFECTED
numHealthy = POPULATION_SIZE - numSick
numRecovered = 0

dayCount = 0
numInfectedPerDay = [STARTING_NUM_INFECTED]

listOfSickNums = [numSick]
listOfHealthyNums = [numHealthy]
listOfRecoveredNums = [numRecovered]

while numSick >= STARTING_NUM_INFECTED:

	dayCount += 1
	oldNumSick = numSick

# the disease infects people
	newSick = (INFECTION_RATE-1)*numSick*(POPULATION_SIZE-(numSick+numRecovered))/POPULATION_SIZE
	numSick += newSick
	numHealthy -= newSick
	numInfectedPerDay.append(newSick)

# sick people recover
	if dayCount >= DISEASE_DURATION:
		newRecovered = numInfectedPerDay[dayCount - DISEASE_DURATION]
		numSick -= newRecovered
		numRecovered += newRecovered

	if oldNumSick < QUARANTINE_THRESHOLD and numSick >= QUARANTINE_THRESHOLD:
		print("The quarantine begins on day", dayCount)
		beginQuarantineDay = dayCount
		p.axvline(x=dayCount, color="black")

	if oldNumSick >= QUARANTINE_THRESHOLD and numSick < QUARANTINE_THRESHOLD:
		print("The quarantine ends on day", dayCount - 1)		
		print("The quarantine lasted for", dayCount - beginQuarantineDay - 1, "days")
		p.axvline(x=dayCount-1, color="black")

	listOfSickNums.append(numSick)
	listOfHealthyNums.append(numHealthy)
	listOfRecoveredNums.append(numRecovered)

p.plot(listOfSickNums, color="red")
p.plot(listOfHealthyNums, color="green")
p.plot(listOfRecoveredNums, color="blue")
p.show()


