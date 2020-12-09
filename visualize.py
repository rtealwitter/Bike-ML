import matplotlib.pyplot as plt

def importfile(filename):
    with open(filename, 'r') as f:
        for line in f:
            results = eval(line.replace('\n', ''))
    return results

def plot(results, keyfilter, title):
    for key in results:
        for key2 in results[key]:
            if key[1] == keyfilter:
                data = results[key][key2]
                name = "{} ({})".format(key[0], key2)
                plt.plot(list(range(1,len(data)+1)), data, label=name)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title(title)
    plt.ylim(0)
    plt.legend()
    plt.show()

results = importfile(filename = 'boostingresults.txt')
plot(results, keyfilter='Margin', title='SVM Boosting with Imbalanced Margins')
