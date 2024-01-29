import numpy as np
from scipy.optimize import curve_fit
import time
from scipy.stats import norm
import scipy.integrate
from scipy.optimize import fmin
import random
from matplotlib.dates import num2date, date2num
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform, loguniform
import math
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sktime.split import (
    CutoffSplitter,
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
def find_rul(failure_time, last_prediction_time):
    try:
        rul = (num2date(failure_time) - num2date(last_prediction_time)).days
        # print(f"failure_time: {failure_time}")
        # print(f"last prediction time: {last_prediction_time}")
        # print(f"rul is {rul}")
        if rul < 0:
            return np.nan
        return rul
    except:
        return np.nan
class SupportVectorRegression:
    def __init__(self):
        pass

    def extrapolation(self, x, y, threshold, options:dict):
        for k in range(len(x)): 
            if k < 2:
                result = {
                    'iteration': k,
                    'info': {
                        'model':'svr'
                    },
                    'x_prediction': np.nan,
                    'y_prediction': np.nan,
                    'failure_time':np.nan,
                    'threshold': np.nan,
                    'rul': np.nan
                } 
                yield result
                continue
            x_train = x[0: k + 1]   
            y_train = y[0: k + 1]
            pipeline = Pipeline([
                ('scaler', MinMaxScaler()),
                ('svr', SVR(**options))
            ])

            param_grid = {
                'svr__C': [100],
                'svr__epsilon': [0.001, 0.01, 0.1, 0.2],
                'svr__gamma': [0.0001, 0.001, 0.01, 0.1],
            }
            tscv = TimeSeriesSplit(n_splits=max(2, len(x_train)-1))
            mdl = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error')
            mdl.fit(x_train.reshape(-1, 1), y_train.ravel())
            t_end = self.find_threshold_passing(x_train, threshold, mdl)
            if t_end is None:
                x_prediction = np.nan
                y_prediction = np.nan
            else:
                x_prediction = np.linspace(x_train[0], t_end, 1000)
                y_prediction = mdl.predict(x_prediction.reshape(-1, 1))
            rul = find_rul(t_end, x_train[-1])

            result = {
                'iteration': k,
                'info': {
                    'model':'svr'
                },
                'x_prediction': x_prediction,
                'y_prediction': y_prediction,
                'failure_time':t_end,
                'threshold': threshold,
                'rul': rul
            } 
            yield result
    def customized_extrapolation(self, x, y, threshold, options:dict):
        for k in range(len(x)): 
            if k < 2:
                result = {
                    'iteration': k,
                    'info': {
                        'model':'svr'
                    },
                    'x_prediction': np.nan,
                    'y_prediction': np.nan,
                    'failure_time':np.nan,
                    'threshold': np.nan,
                    'rul': np.nan
                } 
                yield result
                continue
            x_train = x[0: k + 1]   
            y_train = y[0: k + 1]

            self.x_train = x_train
            self.y_train = y_train

            self.x_scaler = MinMaxScaler()
            self.y_scaler = MinMaxScaler()

            self.xn_train = self.x_scaler.fit_transform(self.x_train.reshape(-1, 1))
            self.yn_train = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1))

            mdl = SVR(**options)
            param_distributions = {
                'epsilon': loguniform(1e-2, 5e-1),
            }
            n_iter = 50
            
            random_cv = RandomizedSearchCV(
                mdl,
                scoring=self.custom_scorer, 
                param_distributions=param_distributions,
                cv=2,
                n_iter=n_iter,
                random_state=0
            )

            random_cv.fit(self.xn_train.reshape(-1, 1), self.yn_train.ravel())
            mdl = random_cv.best_estimator_

            threshold_normalized = self.y_scaler.transform(np.array([[threshold]]))[0, 0]
            tn_end = self.find_threshold_passing(self.xn_train, threshold_normalized, mdl)

            if tn_end is None:
                x_prediction = np.nan
                y_prediction = np.nan
                t_end = np.nan
            else:
                t_end = self.x_scaler.inverse_transform(tn_end.reshape(-1, 1))[0, 0]
                x_prediction = np.linspace(x_train[0], t_end, 1000)
                xn_prediction = self.x_scaler.transform(x_prediction.reshape(-1, 1))
                yn_prediction = mdl.predict(xn_prediction)
                y_prediction = self.y_scaler.inverse_transform(yn_prediction.reshape(-1, 1))

            rul = find_rul(t_end, x_train[-1])

            result = {
                'iteration': k,
                'info': {
                    'model':'svr'
                },
                'x_prediction': x_prediction,
                'y_prediction': y_prediction,
                'failure_time':t_end,
                'threshold': threshold,
                'rul': rul
            } 
            yield result

            

    
    def custom_scorer(self, mdl, X_test, y_test):
        X, y = self.x_train.reshape(-1, 1), self.y_train.ravel()
        if X_test[-1] == X[-1]:
            mdl.fit(X, y)
            epsilon = mdl.get_params()['epsilon']
            s, u = 0, 0
            for i in mdl.support_:
                if ((abs(y[i] - (mdl.predict(X[i].reshape(-1, 1)) + epsilon))) <
                        (abs(y[i] - (mdl.predict(X[i].reshape(-1, 1)) - epsilon)))):
                    u += (y[i] - (mdl.predict(X[i].reshape(-1, 1)) + epsilon) >= -0.001)
                    s += (-0.001 < (y[i] - (mdl.predict(X[i].reshape(-1, 1)) + epsilon)) < 0.02)

            C0, SV_score = 10, -1 * abs(u-s)
            C1, epsilon_score = 0.7, - math.log(epsilon)/math.log(10)
            C2, SV_score2 = 3, s/len(X)
            C3, slope_score = 100, (mdl.predict((X[-1]+1).reshape(-1, 1)) - mdl.predict(X[-1].reshape(-1, 1)) > 0) and \
                (mdl.predict((X[-1] + 0.1).reshape(-1, 1)) - mdl.predict(X[-1].reshape(-1, 1)) > 0) and \
                (mdl.predict((X[-1] + 0.5).reshape(-1, 1)) - mdl.predict(X[-1].reshape(-1, 1)) > 0) and \
                (mdl.predict((X[-1] + 0.7).reshape(-1, 1)) - mdl.predict(X[-1].reshape(-1, 1)) > 0)

            score = C0 * SV_score + C1 * epsilon_score + C2 * SV_score2 + C3 * slope_score
            # print(mdl.get_params(), s, u, SV_score, SV_score2, epsilon_score, slope_score, score)

            # n = -1 * mdl.n_support_[0]
            # score = n + C1 * epsilon_score
            # print('number of SVs:', n, 'epsilon: ', epsilon, '\t epsilon_score:', epsilon_score, '\t C1 * epsilon score:',
            #       score - n, '\n score:', score)
            return score
        else:
            return -1e10
    def find_threshold_passing(self, x, threshold, mdl):
        x_future = 2 * x[-1] - x[0]
        x_pred = np.linspace(x[0], x_future, 1000)
        for _ in range(10):
            y_pred = mdl.predict(x_pred.reshape(-1, 1))
            exceeding_indices = np.where(y_pred > threshold)[0]
            if len(exceeding_indices) > 0:
                return x_pred[exceeding_indices[0]]
            x_min, x_max = x_pred[-1], 2 * x_pred[-1] - x_pred[0]
            x_pred = np.linspace(x_min, x_max, 1000)
        return None