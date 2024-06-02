import sys
import os
from knn import load_data, z_normalize, nn_leave_one_out_cv
from feature_selection import forward_selection, backward_elimination

def main():
    print("Welcome to Po-Chu Feature Selection Algorithm.")
    file_name = input("Type in the name of the file to test: ")

    # Check if file exists
    if not os.path.isfile(file_name):
        print(f"Error: The file '{file_name}' does not exist. Please check the file name and try again.")
        return

    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    algorithm_choice = input()

    X, y = load_data(file_name)

    n_features = len(X[0])
    n_instances = len(X)

    print(f"\nThis dataset has {n_features} features (not including the class attribute), with {n_instances} instances.")

    normalized_X = z_normalize(X)
    initial_accuracy = nn_leave_one_out_cv(normalized_X, y) * 100
    print(f"\nRunning nearest neighbor with all {n_features} features, using \"leaving-one-out\" evaluation, I get an accuracy of {initial_accuracy:.1f}%")

    if algorithm_choice == '1':
        forward_selection(normalized_X, y)
        algorithm_name = "Forward Selection"
    elif algorithm_choice == '2':
        backward_elimination(normalized_X, y)
        algorithm_name = "Backward Elimination"
    else:
        print("Invalid choice.")
        return

if __name__ == "__main__":
    main()