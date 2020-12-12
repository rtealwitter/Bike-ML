import matplotlib.pyplot as plt

def importfile(filename):
    with open(filename, 'r') as f:
        for line in f:
            results = eval(line.replace('\n', ''))
    return results

def plot(results, keyfilter, title, filename):
    for key in results:
        for key2 in results[key]:
            if key[0] == keyfilter:
                data = results[key][key2]
                if key[1] == 'None':
                    name = 'Balanced Margin ({})'.format(key2)
                if key[1] == 'Margin':
                    name = 'Imbalanced Margin ({})'.format(key2)
                plt.plot(list(range(1,len(data)+1)), data, label=name)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title(title)
    plt.ylim(0)
    plt.legend()
    plt.savefig(filename)
    plt.clf()

results = importfile(filename = 'boostingresults.txt')
plot(results, keyfilter='Reweight', title='AdaBoost with SVM Weak Learner and Reweighted Data', filename='graphics/reweighted.pdf')
plot(results, keyfilter='Resample', title='AdaBoost with SVM Weak Learner and Resampled Data', filename='graphics/resampled.pdf')
