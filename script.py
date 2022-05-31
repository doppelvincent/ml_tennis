import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:

data = pd.read_csv("tennis_stats.csv")
# print(data.head())
print(data.columns)
# print(data.describe())



# perform exploratory analysis here:


# plt.scatter(data.BreakPointsOpportunities, data.Winnings)
# plt.title("BreakPointsOpportunities vs Winnings")
# plt.xlabel("BreakPointsOpportunities")
# plt.ylabel("Winnings")
# plt.show()
# plt.clf()

## perform single feature linear regressions here:

features = data[["FirstServeReturnPointsWon"]]
outcome = data[["Winnings"]]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8)

model = LinearRegression()
model.fit(features_train, outcome_train)

score = model.score(features_test, outcome_test)
print("The score for the model is " + str(score))
prediction = model.predict(features_test)

plt.scatter(outcome_test, prediction, alpha=0.4)
plt.title("FirstServeReturnPointsWon vs Winnings")
plt.show()
plt.clf()

##############

features = data[["BreakPointsOpportunities"]]
outcome = data[["Winnings"]]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8)

model = LinearRegression()
model.fit(features_train, outcome_train)

score = model.score(features_test, outcome_test)
print("The score for the model is " + str(score))
prediction = model.predict(features_test)

plt.scatter(outcome_test, prediction, alpha=0.4)
plt.title("BreakPointsOpportunities vs Winnings")
plt.show()
plt.clf()

######################

features = data[["Aces"]]
outcome = data[["Winnings"]]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8)

model = LinearRegression()
model.fit(features_train, outcome_train)

score = model.score(features_test, outcome_test)
print("The score for the model is " + str(score))
prediction = model.predict(features_test)

plt.scatter(outcome_test, prediction, alpha=0.4)
plt.title("Aces vs Winnings")
plt.show()
plt.clf()


#Best model: BreakPointsOpportunities, Aces, 


## perform two feature linear regressions here:

features = data[["BreakPointsOpportunities", "Aces"]]

outcome = data[["Winnings"]]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8)

model = LinearRegression()
model.fit(features_train, outcome_train)

score = model.score(features_test, outcome_test)
print("The score of the model with two features is " + str(score))

prediction = model.predict(features_test)

plt.scatter(outcome_test, prediction, alpha=0.4)
plt.title("BreakPointsOpportunities & Aces vs Winnings")
plt.show()
plt.clf

## perform multiple feature linear regressions here:
features = data[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
outcome = data[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8)

model = LinearRegression()
model.fit(features_train, outcome_train)

score = model.score(features_test, outcome_test)
print("The score of the model with multiple features is " + str(score))

prediction = model.predict(features_test)

plt.scatter(outcome_test, prediction, alpha=0.4)
plt.title("Model with multiple features vs Winnings")
plt.show()
plt.clf()