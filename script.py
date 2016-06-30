import pandas
import ipdb
# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold
# ipdb.set_trace()

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

#pandas has a method that reads a csv file
titanic = pandas.read_csv("train.csv")

#print the first 10 rows in the csv
# print(titanic.head(10))

# print(titanic.describe())

#change NAN values in Age to the median Age of the Agedata
age_median = titanic["Age"].median()
titanic["Age"] = titanic["Age"].fillna(value=age_median)

#change our Sex values to numeric
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

#change the Embarked Nan values to S and also convert our values to numeric
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S" , "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C" , "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q" , "Embarked"] = 2

alg = LinearRegression()

# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.

#titanic.shape[0] returns 891 - numbers of the rows, n_folds means divide it into 3
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

# for train, test in kf:
#     print("%s" % (train))
#http://chrisalbon.com/python/pandas_indexing_selecting.html
print(titanic[predictors])
predictions = []
for train, test in kf:
    ipdb.set_trace()e


    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # print(train_predictors)
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
