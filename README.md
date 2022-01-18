# Contribution Weighted Model Based on Order Data
## Purpose
I created a model to predict the answers to binary questions proposed in prediction markets based on the order data of market participants. The dataset I analyze comes from the Good Judgement Project and involves prediction markets speculating on whether an event will occur or not. Contracts are traded based on occurence of the event. If the event does occur, the value of the contract becomes 100 and if the event does not occur, the value of the contract is 0. My model first calculates the past profits of market participants. It then infers market participants' true belief of the probability of an event occuring using their market orders and a Constant Relative Risk Aversion (CRRA) utility function. The model then uses a Bayesian approach in weighing the contribution of participants with positive profits according to the amount of their profit. Based on other market paricipants' orders, the model is able to generate a probability of an event occuring and a confidience interval that can then be used to trade contracts in the prediction market.
