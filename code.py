import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

def preprocess(filename):
    ''' preprocess file '''
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
    ''' rows in dfx that correspond to youtcome in dfy '''
    idx = dfy.index[dfy['y']==youtcome]
    return dfx.iloc[idx,:], dfy.iloc[idx,:]

def balanced(dfx, dfy):
    ''' return subset of dfx with equal number of 0 and 1 rows '''
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

def evalbalancedclassifier(trainfile, testfile, classifier, reps=10, verbose=True):    
    ''' evaluate classifier on training, positive tests, and negative tests '''
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

def erroraggregate(model1, model2, dfx, dfy):
    ''' return error of aggregating model1 and model2 predictions '''
    pred = pd.DataFrame(data=dfy['y'], columns=['y'])
    pred['max1'] = model1.predict_proba(dfx).max(axis=1)
    pred['max2'] = model2.predict_proba(dfx).max(axis=1) 
    pred['greater1'] = pred['max1'] >= pred['max2']
    pred['greater2'] = pred['max1'] < pred['max2']
    pred['pred'] = pred['greater1']*model1.predict(dfx) + pred['greater2']*model2.predict(dfx)
    return np.mean(pred['pred'] != pred['y'])

def evalbalanced(trainfile, testfile, reps=10, verbose=True):
    ''' evaluate the performacne of Logistic Regression,
    Multilayer Perceptron, and aggregation of the two '''
    testx, testy = preprocess(trainfile)
    trainx, trainy = preprocess(testfile)
    errors = []
    for rep in range(reps):
        balx, baly = balanced(trainx, trainy)
        model1 = LogisticRegression(random_state=0).fit(balx, baly['y'])
        model2 = MLPClassifier(random_state=0).fit(balx, baly['y'])
        train1 = 1-model1.score(balx, baly) 
        pos1 = 1-model1.score(*selectdf(testx, testy, 1))
        neg1 = 1-model1.score(*selectdf(testx, testy, 0))
        train2 = 1-model2.score(balx, baly) 
        pos2 = 1-model2.score(*selectdf(testx, testy, 1))
        neg2 = 1-model2.score(*selectdf(testx, testy, 0))
        train = erroraggregate(model1, model2, balx, baly)
        pos = erroraggregate(model1, model2, *selectdf(testx,testy,1)) 
        neg = erroraggregate(model1, model2, *selectdf(testx,testy,0))
        errors += [[train1,pos1,neg1,train2,pos2,neg2,train,pos,neg]]

    df = pd.DataFrame(
        data=errors,
        columns=['train1','pos1','neg1','train2','pos2','neg2','train','pos','neg']
    )
    if verbose:
        print(df.mean(axis=0))
    return df.mean(axis=0)

#print("Logistic Regression")
#evalbalancedclassifier('data/smalltrain.csv', 'data/smalltest.csv', LogisticRegression)
#print("Multilayer Perceptron")
#evalbalancedclassifier('data/smalltrain.csv', 'data/smalltest.csv', MLPClassifier)
evalbalanced('data/smalltrain.csv', 'data/smalltest.csv', reps=20)
