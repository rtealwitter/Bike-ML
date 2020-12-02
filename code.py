import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

def preprocess(filename):
    # Read csv
    df = pd.read_csv(filename)
    # Drop empty values (some speeds are not provided)
    df = df.dropna()
    # Actual values
    dfy = pd.DataFrame(data=np.array(df['ncoll'] > 0).astype(int), columns=['y'])
    # Drop unnecessary columns
    df = df.drop(columns=['datetime', 'date', 'ncoll', 'standarddate'])
    # Convert categorical variables to strings
    for column in ['GEOID', 'month', 'hour', 'day', 'year']:
        df[column] = df[column].astype('str')
    # Convert categories to binary variables
    df = pd.get_dummies(df)
    # Scale
    df = pd.DataFrame(preprocessing.scale(df), columns=df.columns)
    return df, dfy

def selectdf(dfx, dfy, youtcome):
    idx = dfy.index[dfy['y']==youtcome]
    return dfx.iloc[idx,:], dfy.iloc[idx,:]

def balanced(dfx, dfy):
    negdfx, negdfy = selectdf(dfx, dfy, 0)
    posdfx, posdfy = selectdf(dfx, dfy, 1)
    negidx = np.random.choice(len(negdfx), len(posdfx), replace=False)
    baldfx = pd.concat([posdfx, negdfx.iloc[negidx,:]])
    baldfy = pd.concat([posdfy, negdfy.iloc[negidx,:]])
    return baldfx, baldfy

def smalldata():
    ''' Subset data for easier loading '''
    df = pd.read_csv('data/data.csv')
    m = len(df)
    shuffledidx = np.random.permutation(list(range(m)))
    # Training and testing sets from the other half of the data
    idxtrain, idxtest = shuffledidx[:100000], shuffledidx[100000:200000]
    smalltrain, smalltest = df.iloc[idxtrain,:], df.iloc[idxtest,:]
    smalltrain.to_csv('data/smalltrain.csv', index=False)
    smalltest.to_csv('data/smalltest.csv', index=False)

def evalbalanced(trainfile, testfile, classifier, reps=10, verbose=True):    
    trainaccuracy, posaccuracy, negaccuracy = 0, 0, 0
    trainx, trainy = preprocess('data/smalltrain.csv')    
    testx, testy = preprocess('data/smalltest.csv')
    for rep in range(reps):
        balx, baly = balanced(trainx, trainy)
        model = classifier(random_state=0).fit(balx, baly['y'])
        trainaccuracy += model.score(balx, baly)
        posaccuracy += model.score(*selectdf(testx, testy,1))
        negaccuracy += model.score(*selectdf(testx, testy,0))
    if verbose:
        print('Balanced Train Error', 1-trainaccuracy/reps)
        print('Test Positive Error', 1-posaccuracy/reps)
        print('Test Negative Error', 1-negaccuracy/reps)
    return trainaccuracy/reps, posaccuracy/reps, negaccuracy/reps

print("Logistic Regression")
evalbalanced('data/smalltrain.csv', 'data/smalltest.csv', LogisticRegression)
print("Multilayer Perceptron")
evalbalanced('data/smalltrain.csv', 'data/smalltest.csv', MLPClassifier)


