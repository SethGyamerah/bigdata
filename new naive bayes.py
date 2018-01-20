import pandas as pd
from collections import defaultdict

def calc_y_probs(y):
    probs = defaultdict(lambda: 0)
    for i in y:
        probs[i]=0
    for k in probs.keys():
        probs[k] = len(y[y==k])/len(y)
    return probs

def calc_x_probs(X):
    probs = {}
    cols = list(X.columns.values)
    for i in range(len(cols)):
        p = calc_y_probs(X[cols[i]])
        probs[i] = p
    return probs

def conditional_probs(X,Y):
    probs = calc_y_probs(Y)
    for i in probs.keys():
        x = X[Y==i]
        probs[i] = calc_x_probs(x)
    return probs

def make_prediction(X,prob_y,prob_X,prob_Xy):
    n_cols = len(X.columns)
    result = []
    for i in range(X.values.shape[0]):
        c_pr = -1
        c_y = ''
        for j in prob_y.keys():
            pr = prob_y[j]
            for k in range(n_cols):
                pr *= (prob_Xy[j][k][X.values[i][k]] + 1e-5)
            if pr >= c_pr:
                c_y = j
                c_pr = pr
        result.append(c_y)
    return result

def main(dataset_name, target_name):

    d = pd.read_csv(dataset_name)
    targ_name = target_name
    target = d[targ_name]
    data = d.drop([targ_name], axis=1)

    prob_y = calc_y_probs(target)
    prob_X = calc_x_probs(data)
    prob_Xy = conditional_probs(data, target)

    print(list(target))
    pred = make_prediction(data,prob_y,prob_X,prob_Xy)
    print(pred)
    print(len(target[target==pred])/len(target))

main('data_weather.csv', 'play')
main('data_soybean.csv', 'class')