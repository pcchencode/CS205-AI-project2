import random
import numpy as np
from knn import load_data, z_normalize, nn_leave_one_out_cv

# Referenc from CSDN: https://blog.csdn.net/qq_40438165/article/details/107208086
# Just a simple timer
def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result
    return func_wrapper

@timer
def forward_selection(X, y):
    all_features = [i+1 for i in range(len(X[0]))]
    current_candidates = []
    
    global_best_accuracy = -float('inf')
    global_best_features = []

    print("Beginning search.")
    while all_features:
        print("")
        local_best_accuracy = -float('inf')
        local_best_features = []
        for f_idx in all_features:
            # print(f_idx)

            selected_feature_column = current_candidates.copy()
            selected_feature_column.append(f_idx)
            # print("selected_feature_column", selected_feature_column)
            selected_X = []
            for i in range(len(X)):
                tmp = [X[i][idx-1] for idx in selected_feature_column]
                selected_X.append(tmp)
            selected_X = np.array(selected_X)
            accuracy = nn_leave_one_out_cv(selected_X, y)*100
            print(f"    Using feature(s) {selected_feature_column} accuracy is {accuracy: .1f}%")
            if accuracy > local_best_accuracy:
                local_best_accuracy = accuracy
                best_feature = f_idx
                local_best_features = selected_feature_column

        print("")
        if local_best_accuracy < global_best_accuracy:
            print(f"(WARNING, Accuracy has decreased! Continuing search in case of local maximum)")
            # print("Considering improvement threshold....")
            # break
        else:
            global_best_accuracy = local_best_accuracy
            global_best_features = local_best_features
        print(f"Feature set {local_best_features} was best, accuracy is {local_best_accuracy: .1f}%")


        # best_feature = random.choice(all_features) # simulate choicing the feature with best asscuacy
        # print(f"After nn cv, the best feature is: {best_feature}")
        all_features.remove(best_feature)
        current_candidates.append(best_feature)

    print("")
    print(f"Finished search!! The best feature subset is {global_best_features}, which has accuracy of {global_best_accuracy: .1f}%")

    return

@timer
def backward_elimination(X, y):
    all_features = [i+1 for i in range(len(X[0]))]
    current_candidates = all_features.copy()
    
    print("Calculating initial global best accuracy")
    global_best_accuracy = nn_leave_one_out_cv(X, y) # accuracy using all features
    global_best_features = all_features.copy()
    print(f"Global best accuracy using all 50 features is {global_best_accuracy}")
    
    # print(global_best_accuracy)

    print("Beginning search.")
    while all_features:
        print("")
        local_best_accuracy = -float('inf')
        local_best_features = []
        for f_idx in all_features:
            selected_feature_column = current_candidates.copy()
            selected_feature_column.remove(f_idx)

            selected_X = []
            for i in range(len(X)):
                tmp = [X[i][idx-1] for idx in selected_feature_column]
                selected_X.append(tmp)
            selected_X = np.array(selected_X)
            accuracy = nn_leave_one_out_cv(selected_X, y)*100
            print(f"    Using feature(s) {selected_feature_column} accuracy is {accuracy: .1f}%")
            if accuracy > local_best_accuracy:
                local_best_accuracy = accuracy
                best_feature = f_idx
                local_best_features = selected_feature_column

        print("")
        if local_best_accuracy < global_best_accuracy:
            print(f"(WARNING, Accuracy has decreased! Continuing search in case of local maximum)")
            # print("Considering improvement threshold....")
            # break
        else:
            global_best_accuracy = local_best_accuracy
            global_best_features = local_best_features
        print(f"Feature set {local_best_features} was best, accuracy is {local_best_accuracy: .1f}%")

        all_features = local_best_features.copy()
        current_candidates = local_best_features.copy()

    print("")
    print(f"Finished search!! The best feature subset is {global_best_features}, which has accuracy of {global_best_accuracy: .1f}%")

    return


if __name__ == '__main__':
    X, y = load_data("CS205_small_Data__44.txt")
    # X, y = load_data("CS205_large_Data__21.txt")

    normalized_X = z_normalize(X)
    # print(type(X), type(normalized_X))
    forward_selection(normalized_X, y)
    # backward_elimination(normalized_X, y)