from toolbox import *

# Models: Logistic Regresion (simple), Neural Network (complex), SVM (simple),
#         Aggregate (complex and simple), Boost SVM
# Data: Resample, Reweight, Resample + Margin, Reweight + Margin

def evaluatemodels(trainx, trainy, testx, testy, class_weight, T):
    models = {
        'Regression': LogisticRegression(random_state=0, class_weight=class_weight).fit(trainx, trainy['y']),
        'Neural Net': MLPClassifier(random_state=0).fit(trainx, trainy['y']),
        'Boost SVM': BoostModel(class_weight=class_weight).fit(trainx, trainy['y'], T=T)
    }
    models['AggregateSVM'] = AggregateModelSVM(
        models['Regression'], models['Neural Net'],
        class_weight=class_weight).fit(trainx, trainy['y'])
    models['AggregateHeuristic'] = AggregateModelHeuristic(
        models['Regression'], models['Neural Net'],
        class_weight=class_weight).fit(trainx, trainy['y'])
    return scoremodels(models, testx, testy)

def evaluateboost(trainx, trainy, testx, testy, class_weight, T):
    boostmodel = BoostModel(class_weight=class_weight).fit(trainx, trainy['y'], T=T, verbose=True)
    results = {'positive error': [], 'negative error': []}
    for t in range(min(T, len(boostmodel.alphas))):
        results['positive error'].append(
            1-boostmodel.score(*selectdf(testx, testy, 1), t))
        results['negative error'].append(
            1-boostmodel.score(*selectdf(testx, testy, -1), t))
    return results

def wrapper(pretrainx, pretrainy, testx, testy, procedure, T, seed=True):
    if seed: np.random.seed(1)
    results = {}
    datastrategies = {'Resample': unsample, 'Reweight': reweight}
    marginstrategies = {'Margin': margin, 'None': nomargin}
    for datastrategy in datastrategies:
        print('Strategizing data...')
        if seed: np.random.seed(1)
        trainx, trainy = datastrategies[datastrategy](pretrainx, pretrainy)
        for marginstrategy in marginstrategies:
            print('Current configuration:', datastrategy, marginstrategy)
            class_weight = marginstrategies[marginstrategy](trainy)
            results[datastrategy, marginstrategy] = procedure(
                trainx, trainy, testx, testy, class_weight, T)
    return results

def writeresults(filename, results):
    with open(filename, 'a') as f:
        f.write(str(results) + '\n')

def printresults(results):
    for key in results:
        print(key)
        for key2 in results[key]:
            print(key2, results[key][key2])

def experiment(usefulldata):
    print('Loading data...')
    if usefulldata:
        pretrainx, pretrainy, testx, testy = fulldata(seed=True)
    else:
        pretrainx, pretrainy = preprocess('data/smalltrain.csv')
        testx, testy = preprocess('data/smalltest.csv')

    print('Starting boosting...')
    boostresults = wrapper(pretrainx, pretrainy, testx, testy, evaluateboost, T=25)
    printresults(boostresults)
    writeresults('boostingresults.txt', boostresults)

    print('Starting modeling...')
    modelsresults = wrapper(pretrainx, pretrainy, testx, testy, evaluatemodels, T=10)
    printresults(modelsresults)
    writeresults('modelsresults.txt', modelsresults)

experiment(usefulldata=True)