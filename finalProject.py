from collections import defaultdict
import matplotlib.pyplot as plt
import random
import numpy as np
import csv

"""
Kenneth Gwon
Prediction Markets in FE
Contribution Weighted Model Based on Order Data
December 23, 2021
"""


def ifps():

	#reads in the ifp id's and correct answers
	file = open("ifps.csv", errors='ignore')
	csvreader = csv.reader(file)
	header = next(csvreader)

	#sorted list of ifps according to date suspended
	ifps = []

	#dictionary that maps ifp id to correct answer
	ifpDict = {}

	#dictionary that maps the ifp id to the corresponding question
	questions = {}
	for row in csvreader:
		if (row[6] != "NULL") and (row[12]=='2') and (row[9]!=""):
			if row[0].split('-')[0] not in ifpDict:
				ifpDict[row[0].split('-')[0]] = row[9]
				ifps.append(row[0].split('-')[0])
				questions[row[0].split('-')[0]] = row[2]
	file.close()

	#shuffles the order of the prediction markets
	#this will result in randomized selection of training and test datasets
	random.shuffle(ifps)

	return ifps, ifpDict, questions

def splitData(ifpList, ifpDict):
	"""Splits the prediction markets into training data and test data"""

	#reads in the ifp id's and correct answers
	file = open("pm_transactions.lum1.yr3.csv", errors='ignore')
	csvreader = csv.reader(file)
	header = next(csvreader)

	#which prediction markets have a first price and last price differ by less than 30
	#maps ifp ids to pairs of (opening price, closing price)
	marketPrices = {}
	for row in csvreader:
		if row[4] == 'trade' and row[1] in ifpList:
			if row[1] not in marketPrices:
				marketPrices[row[1]] = [int(row[13]),int(row[13])]
			else:
				marketPrices[row[1]][1] = int(row[13])

	#delete prediction markets where no trades occur
	for i in range(len(ifpList)-1,-1,-1):
		if ifpList[i] not in marketPrices:
			ifpList.remove(ifpList[i])

	#deletes the prediction markets where opening price and closing price differ by more than 30
	for key in list(marketPrices.keys()):
		if abs(marketPrices[key][1] - marketPrices[key][0]) > 30:
			del marketPrices[key]
			ifpList.remove(key)

	#uses 80% of the data to identify positive contributors
	ifpTrain = ifpList[:round(len(ifpList)*.8)]
	#uses the remaining 20% to evaluate teh performance of the positive contributors
	ifpTest = ifpList[round(len(ifpList)*.8):]

	file.close()

	return ifpTrain, ifpTest

def calculateTrueBelief(demand,wealth,price,gamma):

	#inferred from the CRRA utility function
	x = ((demand+wealth-(price*demand))/(wealth-(price*demand)))**gamma
	return (x*price) / (1-price+(x*price))

def train(ifpTrain, ifpDict):

	#reads in the ifp id's and correct answers
	file = open("pm_transactions.lum1.yr3.csv", errors='ignore')
	csvreader = csv.reader(file)
	header = next(csvreader)

	#dictionary that tracks the amount of profit of each person
	profit = defaultdict(int)

	#tracks who bet on each event and their true belief 
	aggProb = {}
	testAggProb = {}
	#tracks who placed orders on which games
	whoBet = {}
	testWhoBet = {}

	#tracks the average of the prices of the last five trades
	prices = {}

	#dollar amount of largest trade
	largestTrade = 0
	#assumed wealth of each individual
	wealth = 0

	for row in csvreader:
		#weighing based on profit and dollar amount of largest trade
		if row[4] == 'trade' and row[1] in ifpTrain:
			#if the person bought the contract
			if ((row[6] == 'true' and row[7] == 'true') or (row[6] == 'false' and row[7] == 'false')) and row[1] in ifpDict:
				#value of contract is 1
				if ifpDict[row[1]] == row[2]:
					profit[row[3]] += (100-int(row[13]))*int(row[14])
				#value of contract is 0
				else:
					profit[row[3]] -= int(row[13])*int(row[14])
			#if the person sold the contract
			if ((row[6] == 'false' and row[7] == 'true') or (row[6] == 'true' and row[7] == 'false')) and row[1] in ifpDict:
				#value of contract is 1
				if ifpDict[row[1]] == row[2]:
					profit[row[3]] -= (100-int(row[13]))*int(row[14])
				#value of contract is 0
				else:
					profit[row[3]] += int(row[13])*int(row[14])
			#calculating dollar amount of largest trade
			largestTrade = max(largestTrade, int(row[13])*int(row[14]), (100-int(row[13]))*int(row[14]))

	#deletes the people with negative profit
	for key in list(profit.keys()):
		if profit[key] <= 0:
			del profit[key]


	#sets the wealth of each individual to 2x the largest trade amount
	wealth = 2*largestTrade

	file.close()

	return profit, wealth


def test(ifpTest, contributions, wealth, questions):

	#reads in the ifp id's and correct answers
	file = open("pm_transactions.lum1.yr3.csv", errors='ignore')
	csvreader = csv.reader(file)
	header = next(csvreader)

	#tracks the average of the prices of the last five trades
	prices = {}

	#tracks the true beliefs for each event
	trueBeliefs = {}

	#value according to Harrison
	gamma = 3.974
	for row in csvreader:
		#calculates the average price of the last five trades
		if row[4] == 'trade' and row[1] in ifpTest:
			if row[1] not in prices:
				prices[row[1]] = [row[11]]
			elif len(prices[row[1]]) == 10:
				prices[row[1]].pop(0)
				prices[row[1]].append(int(row[11]))
			else:
				prices[row[1]].append(int(row[11]))

		if row[1] in ifpTest and row[3] in contributions and row[4] == "orderCreate":
			#determines if trader wants to buy or sell
			buy = None
			if ((row[6] == 'true' and row[7] == 'true') or (row[6] == 'false' and row[7] == 'false')):
				buy = True
			if ((row[6] == 'false' and row[7] == 'true') or (row[6] == 'true' and row[7] == 'false')):
				buy = False
			if buy:
				demand = int(row[12])
			else:
				demand = -1*int(row[12])
			price = int(row[11]) / 100
			q = calculateTrueBelief(demand,wealth/100,price,gamma)
			if row[1] not in trueBeliefs:
				trueBeliefs[row[1]] = {}
			trueBeliefs[row[1]][row[3]] = q

	#takes the average of the last 5 prices
	for key in list(prices.keys()):
		prices[key] = sum(prices[key]) / len(prices[key])

	#maximum exposure for any trade is $10,000
	modelWealth = 10000

	alphas = {}
	betas = {}
	#maps market ifp id to point estimate for probability of event occuring
	predictions = {}
	demands = {}
	constant = .0001
	for market in ifpTest:
		#generates the model's prediction of the true probability
		alpha = 0
		beta = 0
		for trader in list(trueBeliefs[market].keys()):
			alpha += trueBeliefs[market][trader] * contributions[trader] * constant
			beta += contributions[trader] * constant
		alphas[market] = alpha
		betas[market] = beta
		predictions[market] = alpha / (alpha+beta)

		## graphs the models estimation of true probaiblity
		# if market == ifpTest[5]:
		# 	graph(alpha,beta,questions[market])

		#generates the demand for the contract in each market
		demands[market] = contractsDemanded(modelWealth/100, predictions[market], prices[market]/100, gamma)

	return demands, prices, predictions, alphas, betas

def modelProfit(demands, prices, eventOutcomes, predictions):

	controlProfit = 0
	controlWealth = 5000
	profit = 0
	for market in list(demands.keys()):
		#calculates the model's profit
		if demands[market] > 0:
			if eventOutcomes[market] == "a":
				profit += demands[market] * (100 - prices[market])
			else:
				profit -= demands[market] * prices[market]
		else: 
			if eventOutcomes[market] == "a":
				profit -= abs(demands[market]) * (100 - prices[market])
			else:
				profit += abs(demands[market]) * prices[market]
		#randomly chooses to buy or sell
		#calculates the profit of the random strategy
		if random.choice(["a","b"]) == "a":
			if eventOutcomes[market] == "a":
				controlProfit += controlWealth/prices[market] * (100 - prices[market])
			else:
				controlProfit -= controlWealth/prices[market] * prices[market]
		else: 
			if eventOutcomes[market] == "a":
				controlProfit -= controlWealth/(100-prices[market]) * (100 - prices[market])
			else:
				controlProfit += controlWealth/(100-prices[market]) * prices[market]

	return profit, controlProfit


def contractsDemanded(wealth, trueBelief, price, gamma):
	#uses the CRRA utility function to determine how many contracts the model will buy/sell
	#based on the model's true belief of price and its 
	y = ((trueBelief*(1-price)) / (price*(1-trueBelief)))**(1/gamma)
	x = price*(y-1)
	r = -x/(1+x)
	d = (wealth/price) * r

	return d


def graph(alpha,beta,name):
	#graphs the models aggregated beta distribution
	iterations = 10000
	y = []
	for i in range(iterations):
		y.append(np.random.beta(alpha,beta))
		# y.append(gen.generateBeta(round(alpha),round(beta)))

	plt.title(name)
	plt.hist(y, bins = [i*.01 for i in range(101)], density=True)
	plt.show()



def main():

	# so the experiment can be regerenated with the same values
	random.seed(1)

	ifpList, eventOutcomes, questions = ifps()

	totalProfit = 0
	iterations = 50
	for i in range(iterations):

		trainingMarkets, testingMarkets = splitData(ifpList, eventOutcomes)

		contributions, wealth = train(trainingMarkets, eventOutcomes)

		demands, prices, predictions, alphas, betas = test(testingMarkets, contributions, wealth, questions)

		profit, controlProfit = modelProfit(demands, prices, eventOutcomes, predictions)

		totalProfit += profit

	print("Profit Using Contribution Weighted Model:")
	print(totalProfit / iterations)
	print()
	print("Profit From Randomly Buying or Selling:")
	print(controlProfit / iterations)

main()