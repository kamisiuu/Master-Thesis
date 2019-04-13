import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
# Importing C-Support Vector Classification from scikit-learn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from time import time



# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-1, 1], 'C': [1, 10, 100]},
                    {'kernel': ['linear'], 'C': [1, 10, 100]},
                    {'kernel': ['poly'], 'degree': [2, 3, 4], 'coef0': [0, 1], 'gamma': [1e-2, 1e-1, 1],
                     'C': [1, 10, 100]}]

class grid_search_svc:
    def __init__(self,train_x,train_y):
        # run rbf randomized search
        start = time()
        rbf_grid_search = GridSearchCV(SVC(), tuned_parameters[0], cv=5)

        rbf_grid_search.fit(train_x, train_y)
        print("RBF RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), len(rbf_grid_search.cv_results_['params'])))
        report(rbf_grid_search.cv_results_)

        # run linear randomized
        start = time()
        linear_grid_search = GridSearchCV(SVC(), tuned_parameters[1], cv=5)

        linear_grid_search.fit(train_x, train_y)
        print("LINEAR RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), len(linear_grid_search.cv_results_['params'])))
        report(rbf_grid_search.cv_results_)

        # run poly randomized search
        start = time()
        n_iter_search = 20
        poly_random_search = RandomizedSearchCV(SVC(), tuned_parameters[2], cv=5,
                                                n_iter=n_iter_search)

        poly_random_search.fit(train_x, train_y)
        print("POLY RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        report(poly_random_search.cv_results_)

# class grid_search_knn:
