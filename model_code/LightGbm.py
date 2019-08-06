#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import lightgbm as lgb
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

def model_lgb(X_train,X_test,y_train,y_test,cat_features):

    print("load data")
    # create dataset for lightgbm
    #lgb_train = lgb.Dataset(X_train, label=y_train)
    #lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


    def lgb_f1_score(labels, preds):
        score = f1_score(labels, np.round(preds))
        return 'f1',score, True

    # specify your configurations as a dict
    gbm = LGBMClassifier(
        random_seed=2019,
        n_jobs=-1,
        objective='binary',
        learning_rate=0.1,
        n_estimators=4000,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=20,
        min_child_weight=9,
        subsample_freq=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=5
    )

    evals_result = {}

    print('Start training...')
    # train
    gbm.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_names=['train', 'val'],
        eval_metric=lgb_f1_score,
        early_stopping_rounds=100,
        verbose=10,
        categorical_feature=cat_features
    )
    print('best score', gbm.best_score_)

    #lgb.plot_metric(evals_result, metric='f1')
    #print('Save model...')

    # save model to file
    #gbm.save_model('lightgbm/model.txt')
    return gbm


    # eval
    '''print(y_pred)
    print('The F1-macro of prediction is:', precision_score(y_test, y_pred))
    print('Dump model to JSON...')

    # dump model to json (and save to file)
    model_json = gbm.dump_model()
    with open('lightgbm/model.json', 'w+') as f:
        json.dump(model_json, f, indent=4)
    print('Feature names:', gbm.feature_name())
    print('Calculate feature importances...')

    # feature importances
    print('Feature importances:', list(gbm.feature_importance()))'''
