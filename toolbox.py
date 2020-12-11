import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.base import clone
import cProfile

def preprocess(data, filename=True):
    ''' preprocess file '''
    if filename:
        # Read csv
        df = pd.read_csv(data)
    else:
        df = data
    # Drop empty values (some speeds are not provided)
    df = df.dropna()
    # Actual values
    dfy = pd.DataFrame(data=np.array(2*(df['ncoll'] > 0)-1).astype(int), columns=['y'])
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

def fulldata(seed=True):
    if seed: np.random.seed(1)
    df = pd.read_csv('data/data.csv')
    m = len(df)
    shuffledidx = np.random.permutation(list(range(m)))
    idxtest, idxtrain = shuffledidx[:(m//5)], shuffledidx[(m//5):]
    train, test = df.iloc[idxtrain,:], df.iloc[idxtest,:]
    trainx, trainy = preprocess(train, filename=False)
    testx, testy = preprocess(test, filename=False)
    return trainx, trainy, testx, testy

def smalldata():
    ''' Subset data for easier loading '''
    # np.random.seed(1)
    df = pd.read_csv('data/data.csv')
    m = len(df)
    shuffledidx = np.random.permutation(list(range(m)))
    # Training and testing sets from the other half of the data
    idxtrain, idxtest = shuffledidx[:100000], shuffledidx[100000:200000]
    smalltrain, smalltest = df.iloc[idxtrain,:], df.iloc[idxtest,:]
    smalltrain.to_csv('data/smalltrain.csv', index=False)
    smalltest.to_csv('data/smalltest.csv', index=False)

# Data stategies
def unsample(dfx, dfy):
    allnegidx = dfy.index[dfy['y']==-1]
    posidx = dfy.index[dfy['y']==1]
    negidx = np.random.choice(allnegidx, len(posidx), replace=False)
    newdfx = pd.concat([dfx.iloc[negidx], dfx.iloc[posidx]])
    newdfy = pd.concat([dfy.iloc[negidx], dfy.iloc[posidx]])
    return newdfx, newdfy

def resample(dfx, dfy):
    negidx = dfy.index[dfy['y']==-1]
    posidx = dfy.index[dfy['y']==1]
    posidxresampled = np.random.choice(posidx, len(negidx))
    newdfx = pd.concat([dfx.iloc[negidx], dfx.iloc[posidxresampled]])
    newdfy = pd.concat([dfy.iloc[negidx], dfy.iloc[posidxresampled]])
    return newdfx, newdfy

def reweight(dfx, dfy):
    negidx = dfy.index[dfy['y']==-1]
    posidx = dfy.index[dfy['y']==1]
    posidxreweighted = np.repeat(posidx, len(negidx)//len(posidx))
    newdfx = pd.concat([dfx.iloc[negidx], dfx.iloc[posidxreweighted]])
    newdfy = pd.concat([dfy.iloc[negidx], dfy.iloc[posidxreweighted]])
    return newdfx, newdfy

# Models
def aggregateinput(regress, neuraln, dfx, dfy):
    df = pd.DataFrame(dfy.copy())
    df['regress probability'] = regress.predict_proba(dfx)[:,0] 
    df['regress predict'] = regress.predict(dfx)
    df['regress decision'] = regress.decision_function(dfx)
    df['neuraln probability'] = neuraln.predict_proba(dfx)[:,0]
    df['neuraln predict'] = neuraln.predict(dfx)
    return df.drop(columns=['y']), df['y']

class AggregateModelSVM():
    def __init__(self, regress, neuraln, class_weight=None):
        self.regress = regress
        self.neuraln = neuraln
        self.class_weight = class_weight
    def fit(self, X, y):
        self.model = LinearSVC(random_state=0, class_weight=self.class_weight).fit(
            *aggregateinput(self.regress, self.neuraln, X, y))
        return self
    def score(self, X, y):
        return self.model.score(
            *aggregateinput(self.regress, self.neuraln, X, y))

class AggregateModelHeuristic():
    def __init__(self, regress, neuraln, class_weight=None):
        self.regress = regress
        self.neuraln = neuraln
        self.class_weight = class_weight
    def fit(self, X, y):
        return self
    def predict(self, X):
        prediction = np.sign(
            self.regress.predict_proba(X).max(axis=1) * self.regress.predict(X)
            + self.neuraln.predict_proba(X).max(axis=1) * self.neuraln.predict(X))
        return pd.DataFrame(data=np.sign(prediction).transpose(), columns=['prediction'])
    def score(self, X, y):
        return np.mean(np.array(self.predict(X)) == np.array(y))

def adaboost(X, y, T, classifier, verbose):
    evaluation = pd.DataFrame(y.copy())
    evaluation['distribution'] = 1/len(y)
    alphas, models = [], []
    for t in range(T):
        # Build weak learner
        print('Training SVM for boosting...')
        model = clone(classifier).fit(
            X, y, sample_weight=np.array(evaluation['distribution']))
        evaluation['prediction'] = model.predict(X)
        evaluation['outcome'] = evaluation['prediction'] * evaluation['y']
        error = np.sum(
            evaluation['distribution'] * (evaluation['outcome'] < 1))        
        if error > .5:
            print('Error above .5, flipping prediction...')
            error = 1-error
        assert error < .5 # Ensure error below .5 on distribution
        alpha = .5*np.log((1-error)/error)
        models.append(model)
        alphas.append(alpha)
        if verbose:
            print('Boosting round:', t)
            print('Boosting error:', error)
        if error == 0:
            print('Returning because of low error:', error)
            return alphas, models
        Z = 2*np.sqrt(error*(1-error))
        # Update distribution according to AdaBoost rule
        evaluation['distribution'] = evaluation['distribution'] * np.exp(
            -alpha*evaluation['outcome'])/Z
        evaluation['distribution'] = evaluation['distribution']/np.sum(evaluation['distribution'])
        assert np.allclose(np.sum(evaluation['distribution']), 1)
    return alphas, models

class BoostModel():
    def __init__(self, class_weight=None):
        self.classifier = LinearSVC(random_state=0, class_weight=class_weight)
    def fit(self, X, y, T, verbose=False):
        self.alphas, self.models = adaboost(X, y, T, self.classifier, verbose)
        return self
    def predict(self, X, T=None):
        prediction = np.zeros((1, len(X)))
        iterations = len(self.alphas) if T==None else T
        for i in range(iterations):
            currentprediction = np.array(self.models[i].predict(X))
            prediction = np.add(self.alphas[i]*currentprediction, prediction)
        return pd.DataFrame(data=np.sign(prediction).transpose(), columns=['prediction'])
    def score(self, X, y, T=None):
        return np.mean(np.array(self.predict(X, T)) == np.array(y))

# Testing
def selectdf(dfx, dfy, youtcome):
    ''' rows in dfx that correspond to youtcome in dfy '''
    idx = dfy.index[dfy['y']==youtcome]
    return dfx.iloc[idx,:], dfy.iloc[idx,:]

def scoremodels(models, testx, testy, names=None):
    results = {}
    for name in models:
        results[name + ' positive error'] = 1-models[name].score(
            *selectdf(testx, testy, 1))
        results[name + ' negative error'] = 1-models[name].score(
            *selectdf(testx, testy, -1))
    return results

def margin(dfy):
    poscoeff = (len(dfy.index[dfy['y']==1])**(1/4))
    negcoeff = (len(dfy.index[dfy['y']==-1])**(1/4))
    return {1: 1/poscoeff, -1: 1/negcoeff}

def nomargin(dfy):
    return None
