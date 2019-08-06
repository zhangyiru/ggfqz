from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.metrics import precision_score

def model_gbdt(X_train,y_train):
    gbdt_model = GradientBoostingClassifier(
        learning_rate=0.1,
        n_estimators=500,
        max_depth=6
    )

    gbdt_model = gbdt_model.fit(X_train, y_train)
    return gbdt_model
