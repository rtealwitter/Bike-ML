from toolbox import *

trainx, trainy = unsample(*preprocess('data/smalltrain.csv'))
class_weight=margin(trainy)

models = {
    'Regression': LogisticRegression(class_weight=class_weight).fit(trainx, trainy['y']),
    'Neural Net': MLPClassifier().fit(trainx, trainy['y']),
    'Linear SVM': LinearSVC(class_weight=class_weight).fit(trainx, trainy['y']),
    'Boost SVM': BoostModel(class_weight=class_weight).fit(trainx, trainy['y'], T=10)
}
models['Aggregate'] = AggregateModel(
    models['Regression'], models['Neural Net'], models['Linear SVM'],
    class_weight=class_weight).fit(trainx, trainy['y'])

testx, testy = preprocess('data/smalltest.csv')
results = scoremodels(models, testx, testy)

for key in results:
    print(key, results[key])