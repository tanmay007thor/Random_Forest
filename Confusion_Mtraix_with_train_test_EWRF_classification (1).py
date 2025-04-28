#from multiprocessing import Pool
from random import randrange, seed
from csv import reader
from math import sqrt, exp
import numpy as np
import time
import math
from pandas_ml import ConfusionMatrix
#from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Load a CSV File
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row: continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_flt(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    for row in dataset:
        row[column] = int(row[column])


# Calculate accuracy percentage
def accuracy_metric(actual, predicted,temp_conf_1):
    global temp_conf
    correct = 0
    actual_final.append(actual)
    predicted_final.append(predicted)
    for i1 in range(len(actual)):
#        print("*********", actual[i1])
        if actual[i1] == predicted[i1]: 
            correct += 1

          
    cm = confusion_matrix(actual,predicted)
    cm_n = np.float64(cm)
#    print("before",temp_conf_1)
           
    temp_conf = temp_conf + cm_n
   
#    print("Confusion Mtrix",cm)

#    predicted_conf = np.zeros(len(temp_conf))
    print("***Before Normalized Confusion Matrix***",temp_conf)
       
#    print(classification_report(actual_conf,predicted_conf))
#    cm.print_stats()
    return 100. * correct  / float(len(actual))


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, *args):
# To divide the dataset into test and training set, First test set is created 
# Then, based on test set, training set is created so that both would be different    
    test_set = subsample_test(dataset, 1 - sample_ratio)
# Training set is created during construction of each tree usin bagging 
# Hence, Random forest algorithm is called    
    predicted = algorithm(test_set, *args)
    actual = [row[-2] for row in test_set]

#    print("actual sa ", actual)
    accuracy = accuracy_metric(actual, predicted,temp_conf)

    return accuracy


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()

    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0: continue
            proportion = [row[-2] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


def gain_index(groups, classes, H_Y):
    n_instances = float(sum([len(group) for group in groups]))
    gain = 0.0
    split = 0.0
    ratio = 0.0
    score = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue

        for class_val in classes:
            p = [row[-2] for row in group].count(class_val) / float(size)
            if p == 0.0:
                continue
            score = p*math.log(p,2)
            gain += score * (size / n_instances)

        ##########################
        split = split - (size/n_instances)* math.log(size/n_instances, 2)

    gain = H_Y + gain
    if split!=0:
        ratio = gain/split
    return ratio


# Select the best split point for a dataset
def get_split(dataset, n_features):
    class_values = list(set(row[-2] for row in dataset))
    # b_index, b_value, b_score, b_groups = 999, 999, 999, None
    b_index, b_value, b_score, b_groups = 0,0,-999, None
    ###############################################3
    v = float((len(dataset[0]))-1)
    temp = 0
    for class_val in class_values:
        p = [row[-2] for row in dataset].count(class_val) / ((np.size(dataset)) / v)
        if p == 0:
            continue
        temp = temp + p * math.log(p, 2)
    H_Y = (-1) * temp
    ###############################################
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0]) - 2)
        if index not in features: features.append(index)

    for index in features:
#        rv = np.random.randint(low=0, high=(len(dataset) - 1))
        high=(len(dataset) - 1)
        rv = randrange(0, high)
        groups = test_split(index, dataset[rv][index], dataset)
        #gini = gini_index(groups, class_values)
        gain = gain_index(groups, class_values, H_Y)
        # if gini < b_score:
        if gain > b_score:
            b_index, b_value, b_score, b_groups = index, dataset[rv][index], gain, groups

    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group, n_classes):
    class_no = [0]*n_classes
    for row in group:
        no = row[-2]
        class_no[no]+=1
    return class_no


def check_terminate(arr):
    count = 0
    for i in range(len(arr)):
        if arr[i] != 0:
            count += 1

    if count == 1:
        return True

    return False


def split(node, max_depth, min_size, n_features, depth, n_classes):
    left, right = node['groups']
    del (node['groups'])
    if depth > max_depth:
        node['left'], node['right'] = to_terminal(left + right, n_classes), to_terminal(left + right, n_classes)
        return

    if len(left) >= min_size:
        temp = to_terminal(left,n_classes)
        if check_terminate(temp):
            node['left'] = temp

        else:
            node['left'] = get_split(left, n_features)
            split(node['left'], max_depth, min_size, n_features, depth + 1, n_classes)

    if len(left) < min_size:
        node['left'] = to_terminal(left, n_classes)

    if len(right) >= min_size:
        temp = to_terminal(right, n_classes)
        if check_terminate(temp):
            node['right'] = temp

        else:
            node['right'] = get_split(right, n_features)
            split(node['right'], max_depth, min_size, n_features, depth + 1, n_classes)

    if len(right) < min_size:
        node['right'] = to_terminal(right, n_classes)


# Build a decision tree
def build_tree(train1, max_depth, min_size, n_features, n_classes):
    root = get_split(train1, n_features)
    split(root, max_depth, min_size, n_features, 1, n_classes)

    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Create a random subsample from the dataset with replacement
def subsample_test(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        if dataset[index] not in sample:
            sample.append(dataset[index])
    return sample


# Create a random subsample from the dataset with replacement
def subsample_train(dataset, test_data, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        if dataset[index] not in test_data:
            sample.append(dataset[index])
    return sample


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row, n_classes,exponent_value):
    scores = list()
    tree_prob = list()
    distance = list()
    for tree in trees:
        scores, tree_prob, distance = test_knn(tree, row, scores, tree_prob, n_classes,exponent_value, distance)
    
    sum_scores = float(sum(scores))
    for i6 in range(len(scores)):
        scores[i6] = scores[i6]/sum_scores


    for i7 in range(len(scores)):
        for j in range(len(tree_prob[i7])):
            tree_prob[i7][j] = tree_prob[i7][j]*scores[i7]

    final_predict = [0]*n_classes

    for i8 in range(n_classes):
        sum_classes = 0
        for i9 in range(len(scores)):
            sum_classes += tree_prob[i9][i8]
        final_predict[i8] = sum_classes

    ind = final_predict.index(max(final_predict))
    return ind


# Random Forest Algorithm
def random_forest(test, max_depth, min_size, n_trees, n_features, sample_ratio, n_classes,exponent_value,temp_conf):
#    trees = list()
    predicted = list()
# For each tree trainig set is constructed and further build tree process is called
# All trees are constructed and appended into a Trees list.   
    for i3 in range(n_trees):
        train = subsample_train(dataset, test, sample_ratio)
        tree = build_tree(train, max_depth, min_size, n_features, n_classes)
        trees.append(tree)
# For Testing, each row from test case is passed to each constructed trees.
    for row in test:
        predict = bagging_predict(trees, row,n_classes,exponent_value)
        file.write(str(predict)+","+str(row[-1])+"\n")
        predicted.append(predict)

#    print("predicted ",predicted)
    return predicted


def depth(d, level=1):
    if not isinstance(d, dict) or not d:
        return level
    return max(depth(d[k], level + 1) for k in d)


def test_knn(tree, test_data, scores, tree_prob, n_classes, exponent_value,distance):
    root = tree
    prob = [0]*n_classes
    dist_knn = 0
    while (isinstance(root, dict)):
        ind = int(root['index'])
        dist = test_data[ind] - root['value']
        dist_knn += dist * dist
        if (dist >= 0):
            root = root['right']
        else:
            root = root['left']

    distance.append(dist_knn)
    score = exp(-sqrt(dist_knn)/exponent_value)
    scores.append(score)

    ave_sum = float(sum(root))
    if ave_sum !=0:
        for i4 in range(len(root)):
            prob[i4] = root[i4]/ave_sum

    index = prob.index(max(prob))
    for j in range(len(prob)):
        if j != index:
            prob[j] = 0.0
        else:
            prob[j] = 1.0


    tree_prob.append(prob)
    return scores, tree_prob, distance


def data_normalize(dataset):
    for i5 in range(len(dataset[0])-2):
        inz = [row[i5] for row in dataset]
        sum_mean = max(inz) - min(inz)

        if(sum_mean != 0):
            for k in range(len(dataset)):
                dataset[k][i5] = (dataset[k][i5] - min(inz)) / sum_mean
    return dataset

start = time.time()

#for f_chunk in pd.read_csv('train_10.csv',chunksize= 20):
#    print(f_chunk.shape)
# paviaU_1_to_9_class
#dataset = load_csv('hyperspectral_data.csv')
dataset = load_csv('Iris_DataSet_id.csv')
for i in range(0, (len(dataset[0])-1)):
    str_column_to_flt(dataset, i)

str_column_to_int(dataset, (len(dataset[0]) - 2))

dataset = data_normalize(dataset)

i = len(dataset[0])-2


trees = list()

max_depth = 25
min_size = 10
Max_times = 5
sample_ratio = 0.7
n_classes = 3

# Define temp_conf size as actual confusion matrix dimension
temp_conf = np.zeros((3,3))

actual_final = list()
predicted_final = list()
#temp_conf = np.zeros((n_classes-1,n_classes-1))
exponent_value = 0.75
score_array = np.array([0.0] * Max_times)
n_features = int(sqrt(len(dataset[0]) - 2)) + 1
seed(123)


for n_trees in [45]:
    for j in range(1, Max_times + 1):
        filename='Iris_predict_'+str(j)+'.csv'
        file = open(filename, 'a')        
        k = j
#        seed(k)
        scores = evaluate_algorithm(dataset, random_forest, max_depth, min_size, n_trees, n_features, sample_ratio, n_classes,exponent_value,temp_conf)
        score_array[k - 1] = scores
        print("Accuracy after",j,"Iteration:",scores)
        file.write(str(scores)+"\n")
        file.close()
    
    a_list = [y for x in actual_final for y in x]
    p_list = [y for x in predicted_final for y in x]
    conf = ConfusionMatrix(a_list,p_list)
    conf.print_stats()
    
   
    row_count = np.zeros(len(temp_conf))
    for l in range(len(temp_conf)):
        for m in range(len(temp_conf)):
            row_count[l] = row_count[l] + temp_conf[l][m]
#            predicted_conf[l] = predicted_conf[l] + temp_conf[m][l]              
        for m in range(len(temp_conf)):
            temp_conf[l][m] = float(temp_conf[l][m] / float(row_count[l]))     
    print("***Normalized Confusuin Matrix***",temp_conf)
    
    end = time.time()
    print("Total Execution Time:", end - start)
    print("No of trees:", n_trees)
    print("Max Times:", Max_times)
    print("Max depth:", max_depth)
    print("min_size:", min_size)
    print("train test ratio:", sample_ratio)
    print("exponent_value:", exponent_value)
    print('******Mean***** :', np.mean(score_array))
    print('****Standard Deviation ****:', np.std(score_array))



