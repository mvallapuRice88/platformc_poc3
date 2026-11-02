from xgboost import XGBClassifier

def train(x,y,seed):
    model=XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,eval_metric="logloss",random_state=seed,n_jobs=-1)
    return model.fit(x,y)

def predict(model,x):return model.predict(x)
