# CS205-AI-project2
project 2 of implementing Nearest-Neighbor(kNN with k=1) and Feature Selection(Forward Selection and Backward Elimination)

## File description
* `knn.py`: Main function containing normalization, leaving-one-out method and Nearest Neighbor algorithm.
* `feature_selection.py`: Main function containing Forward Selection and Backward Elimination.
* `driver.py`: Driver programm starting the selection algorithms
* `./log`: Containg the log file running the algorithms
* `.txt`: The assigned personal dataset

## How to run?
### ① If you are going to test the Nearest Neighbor algorithm on given small dataset 19 and large dataset 6, run:
```
$ python3 knn.py
```
Expected result:
```
Testing accuracy on small dataset 19
Selected features are [6, 11, 9]
On small dataset 19, the accuracy of using features [6, 11, 9] w/o normalization is: 0.946
====================================================
Testing accuracy on large dataset 6
Selected features are [29, 4, 1]
On large dataset 6, the accuracy of using features [29, 4, 1] w/o normalization is: 0.97
```

### ② If you are going to test the Feature Selection on small dataset 44, run:
```
$ python3 feature_selection.py
```
It will run the Forward Selection by default, expected result:
```
Beginning search.

    Using feature(s) [1] accuracy is  72.4%
    Using feature(s) [2] accuracy is  70.6%
    Using feature(s) [3] accuracy is  70.0%
    Using feature(s) [4] accuracy is  70.2%
    Using feature(s) [5] accuracy is  68.6%
    Using feature(s) [6] accuracy is  71.4%
    Using feature(s) [7] accuracy is  73.2%
    Using feature(s) [8] accuracy is  88.8%
    Using feature(s) [9] accuracy is  67.4%
    Using feature(s) [10] accuracy is  69.8%
    Using feature(s) [11] accuracy is  70.6%
    Using feature(s) [12] accuracy is  72.0%

Feature set [8] was best, accuracy is  88.8%

    Using feature(s) [8, 1] accuracy is  89.0%
    Using feature(s) [8, 2] accuracy is  85.2%
    Using feature(s) [8, 3] accuracy is  86.4%
    Using feature(s) [8, 4] accuracy is  86.2%
    Using feature(s) [8, 5] accuracy is  82.8%
    Using feature(s) [8, 6] accuracy is  81.4%
    Using feature(s) [8, 7] accuracy is  84.8%
    Using feature(s) [8, 9] accuracy is  84.8%
    Using feature(s) [8, 10] accuracy is  93.2%
    Using feature(s) [8, 11] accuracy is  83.2%
    Using feature(s) [8, 12] accuracy is  88.8%

(...omitted

(WARNING, Accuracy has decreased! Continuing search in case of local maximum)
Feature set [8, 10, 4, 1, 3, 12, 2, 11, 7, 6] was best, accuracy is  78.0%

    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 11, 7, 6, 5] accuracy is  76.6%
    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 11, 7, 6, 9] accuracy is  75.4%

(WARNING, Accuracy has decreased! Continuing search in case of local maximum)
Feature set [8, 10, 4, 1, 3, 12, 2, 11, 7, 6, 5] was best, accuracy is  76.6%

    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 11, 7, 6, 5, 9] accuracy is  75.2%

(WARNING, Accuracy has decreased! Continuing search in case of local maximum)
Feature set [8, 10, 4, 1, 3, 12, 2, 11, 7, 6, 5, 9] was best, accuracy is  75.2%

Finished search!! The best feature subset is [8, 10, 4], which has accuracy of  93.4%
forward_selection cost time: 24.409 s
```

### ③ Play the full process of implementation
1. Run `driver.py` and follow the instruction from terminal:
```
$ python3 driver.py
```

2. Type in the dataset(should be downloaded in this local folder) name: `CS205_small_Data__44.txt`
```
Welcome to Po-Chu Feature Selection Algorithm.
Type in the name of the file to test: CS205_small_Data__44.txt
```

3. Choose the method of Feature Selection, 1) as Forward Selection:
```
Type the number of the algorithm you want to run.
1) Forward Selection
2) Backward Elimination
1
```

4. See the selection result:
```
Welcome to Po-Chu Feature Selection Algorithm.
Type in the name of the file to test: CS205_small_Data__44.txt

Type the number of the algorithm you want to run.
1) Forward Selection
2) Backward Elimination
1

This dataset has 12 features (not including the class attribute), with 500 instances.

Running nearest neighbor with all 12 features, using "leaving-one-out" evaluation, I get an accuracy of 75.2%
Beginning search.

    Using feature(s) [1] accuracy is  72.4%
    Using feature(s) [2] accuracy is  70.6%
    Using feature(s) [3] accuracy is  70.0%
    Using feature(s) [4] accuracy is  70.2%
    Using feature(s) [5] accuracy is  68.6%
    Using feature(s) [6] accuracy is  71.4%
    Using feature(s) [7] accuracy is  73.2%
    Using feature(s) [8] accuracy is  88.8%
    Using feature(s) [9] accuracy is  67.4%
    Using feature(s) [10] accuracy is  69.8%
    Using feature(s) [11] accuracy is  70.6%
    Using feature(s) [12] accuracy is  72.0%

Feature set [8] was best, accuracy is  88.8%

    Using feature(s) [8, 1] accuracy is  89.0%
    Using feature(s) [8, 2] accuracy is  85.2%
    Using feature(s) [8, 3] accuracy is  86.4%
    Using feature(s) [8, 4] accuracy is  86.2%
    Using feature(s) [8, 5] accuracy is  82.8%
    Using feature(s) [8, 6] accuracy is  81.4%
    Using feature(s) [8, 7] accuracy is  84.8%
    Using feature(s) [8, 9] accuracy is  84.8%
    Using feature(s) [8, 10] accuracy is  93.2%
    Using feature(s) [8, 11] accuracy is  83.2%
    Using feature(s) [8, 12] accuracy is  88.8%

Feature set [8, 10] was best, accuracy is  93.2%

    Using feature(s) [8, 10, 1] accuracy is  93.2%
    Using feature(s) [8, 10, 2] accuracy is  91.4%
    Using feature(s) [8, 10, 3] accuracy is  90.0%
    Using feature(s) [8, 10, 4] accuracy is  93.4%
    Using feature(s) [8, 10, 5] accuracy is  91.4%
    Using feature(s) [8, 10, 6] accuracy is  92.2%
    Using feature(s) [8, 10, 7] accuracy is  93.4%
    Using feature(s) [8, 10, 9] accuracy is  91.8%
    Using feature(s) [8, 10, 11] accuracy is  91.4%
    Using feature(s) [8, 10, 12] accuracy is  91.6%

Feature set [8, 10, 4] was best, accuracy is  93.4%

    Using feature(s) [8, 10, 4, 1] accuracy is  92.2%
    Using feature(s) [8, 10, 4, 2] accuracy is  88.0%
    Using feature(s) [8, 10, 4, 3] accuracy is  86.4%
    Using feature(s) [8, 10, 4, 5] accuracy is  89.4%
    Using feature(s) [8, 10, 4, 6] accuracy is  90.6%
    Using feature(s) [8, 10, 4, 7] accuracy is  90.0%
    Using feature(s) [8, 10, 4, 9] accuracy is  88.6%
    Using feature(s) [8, 10, 4, 11] accuracy is  89.8%
    Using feature(s) [8, 10, 4, 12] accuracy is  89.4%

(WARNING, Accuracy has decreased! Continuing search in case of local maximum)
Feature set [8, 10, 4, 1] was best, accuracy is  92.2%

    Using feature(s) [8, 10, 4, 1, 2] accuracy is  87.2%
    Using feature(s) [8, 10, 4, 1, 3] accuracy is  89.0%
    Using feature(s) [8, 10, 4, 1, 5] accuracy is  86.8%
    Using feature(s) [8, 10, 4, 1, 6] accuracy is  87.4%
    Using feature(s) [8, 10, 4, 1, 7] accuracy is  87.2%
    Using feature(s) [8, 10, 4, 1, 9] accuracy is  87.6%
    Using feature(s) [8, 10, 4, 1, 11] accuracy is  88.0%
    Using feature(s) [8, 10, 4, 1, 12] accuracy is  89.0%

(WARNING, Accuracy has decreased! Continuing search in case of local maximum)
Feature set [8, 10, 4, 1, 3] was best, accuracy is  89.0%

    Using feature(s) [8, 10, 4, 1, 3, 2] accuracy is  84.0%
    Using feature(s) [8, 10, 4, 1, 3, 5] accuracy is  83.4%
    Using feature(s) [8, 10, 4, 1, 3, 6] accuracy is  82.8%
    Using feature(s) [8, 10, 4, 1, 3, 7] accuracy is  84.6%
    Using feature(s) [8, 10, 4, 1, 3, 9] accuracy is  85.2%
    Using feature(s) [8, 10, 4, 1, 3, 11] accuracy is  84.8%
    Using feature(s) [8, 10, 4, 1, 3, 12] accuracy is  85.4%

(WARNING, Accuracy has decreased! Continuing search in case of local maximum)
Feature set [8, 10, 4, 1, 3, 12] was best, accuracy is  85.4%

    Using feature(s) [8, 10, 4, 1, 3, 12, 2] accuracy is  85.0%
    Using feature(s) [8, 10, 4, 1, 3, 12, 5] accuracy is  84.6%
    Using feature(s) [8, 10, 4, 1, 3, 12, 6] accuracy is  81.6%
    Using feature(s) [8, 10, 4, 1, 3, 12, 7] accuracy is  81.4%
    Using feature(s) [8, 10, 4, 1, 3, 12, 9] accuracy is  83.8%
    Using feature(s) [8, 10, 4, 1, 3, 12, 11] accuracy is  83.2%

(WARNING, Accuracy has decreased! Continuing search in case of local maximum)
Feature set [8, 10, 4, 1, 3, 12, 2] was best, accuracy is  85.0%

    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 5] accuracy is  80.8%
    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 6] accuracy is  79.8%
    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 7] accuracy is  80.2%
    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 9] accuracy is  79.4%
    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 11] accuracy is  81.6%

(WARNING, Accuracy has decreased! Continuing search in case of local maximum)
Feature set [8, 10, 4, 1, 3, 12, 2, 11] was best, accuracy is  81.6%

    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 11, 5] accuracy is  77.4%
    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 11, 6] accuracy is  78.2%
    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 11, 7] accuracy is  80.6%
    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 11, 9] accuracy is  78.8%

(WARNING, Accuracy has decreased! Continuing search in case of local maximum)
Feature set [8, 10, 4, 1, 3, 12, 2, 11, 7] was best, accuracy is  80.6%

    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 11, 7, 5] accuracy is  76.0%
    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 11, 7, 6] accuracy is  78.0%
    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 11, 7, 9] accuracy is  76.8%

(WARNING, Accuracy has decreased! Continuing search in case of local maximum)
Feature set [8, 10, 4, 1, 3, 12, 2, 11, 7, 6] was best, accuracy is  78.0%

    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 11, 7, 6, 5] accuracy is  76.6%
    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 11, 7, 6, 9] accuracy is  75.4%

(WARNING, Accuracy has decreased! Continuing search in case of local maximum)
Feature set [8, 10, 4, 1, 3, 12, 2, 11, 7, 6, 5] was best, accuracy is  76.6%

    Using feature(s) [8, 10, 4, 1, 3, 12, 2, 11, 7, 6, 5, 9] accuracy is  75.2%

(WARNING, Accuracy has decreased! Continuing search in case of local maximum)
Feature set [8, 10, 4, 1, 3, 12, 2, 11, 7, 6, 5, 9] was best, accuracy is  75.2%

Finished search!! The best feature subset is [8, 10, 4], which has accuracy of  93.4%
forward_selection cost time: 24.011 s
```

# 🥳 Hope you have fun:)