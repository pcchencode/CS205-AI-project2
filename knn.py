from tqdm import tqdm
import numpy as np

def load_data(fname):
    data = np.loadtxt(fname)
    return data[:, 1:], data[:, 0]

def z_normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean)/std
    return normalized_data

def zero_one_normalize(data):
    _min = np.min(data)
    _max = np.max(data)
    normalized_data = (data - _min) / (_max - _min)
    return normalized_data

def euclidean_distance(a, b):
    # print(len(a), len(b))
    dist = 0
    for i in range(len(a)):
        dist += (a[i] - b[i])**2
    return np.sqrt(dist)

# Calculate the accuracy of nearest-neighbor under leaving-one-out cross-validation
## X -> selected features dataset X
## y -> label
def nn_leave_one_out_cv(X, y, normailze = False):
    correct_count = 0
    # for i in tqdm(range(len(X))):
    for i in range(len(X)):        
        test_data = X[i]
        train_data = np.concatenate((X[:i], X[i+1:]), axis=0)
        true_y = y[i]

        if normailze:
            norm_test_data = zero_one_normalize(test_data)
            norm_train_data = zero_one_normalize(train_data)
        else:
            norm_test_data = test_data
            norm_train_data = train_data

        shortest_dist = float('inf')
        predict_y = None
        for j in range(len(norm_train_data)):
            dist = euclidean_distance(norm_test_data, norm_train_data[j])
            if dist < shortest_dist:
                shortest_dist = dist
                # correctify the index j for true label
                if j < i:
                    predict_y = y[j]
                else:
                    predict_y = y[j+1]

        if true_y == predict_y:
            correct_count += 1

    accuracy = correct_count/len(X)
    return accuracy

if __name__ == '__main__':
    small_X, small_y = load_data("CS205_small_Data__19.txt")
    large_X, large_y = load_data("CS205_large_Data__6.txt")

    print("Testing accuracy on small dataset 19")
    selected_feature_column = [6, 11, 9]
    print(f"Selected features are {selected_feature_column}")
    selected_X = []
    for i in range(len(small_X)):
        # tmp = [X[i][5], X[i][10], X[i][8]]
        tmp = [small_X[i][idx-1] for idx in selected_feature_column]
        selected_X.append(tmp)

    selected_X = np.array(selected_X)
    normalized_X = z_normalize(selected_X)
    small_accuracy = nn_leave_one_out_cv(selected_X, small_y)
    print(f"On small dataset 19, the accuracy of using features {selected_feature_column} w/o normalization is: {small_accuracy}")

    print("====================================================")

    print("Testing accuracy on large dataset 6")
    selected_feature_column = [29, 4, 1]
    # selected_feature_column = [i for i in range(1, 51)]
    print(f"Selected features are {selected_feature_column}")
    selected_X = []
    for i in range(len(large_X)):
        # tmp = [X[i][5], X[i][10], X[i][8]]
        tmp = [large_X[i][idx-1] for idx in selected_feature_column]
        selected_X.append(tmp)

    selected_X = np.array(selected_X)
    normalized_X = z_normalize(selected_X)
    large_accuracy = nn_leave_one_out_cv(selected_X, large_y)
    print(f"On large dataset 6, the accuracy of using features {selected_feature_column} w/o normalization is: {large_accuracy}")