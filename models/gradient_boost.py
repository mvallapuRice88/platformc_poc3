from sklearn.ensemble import GradientBoostingClassifier

def train(x,y,seed):
    model=GradientBoostingClassifier(n_estimators=200,learning_rate=.05,max_depth=3,random_state=seed)
    return model.fit(x,y)

def predict(model,x):return model.predict(x)
