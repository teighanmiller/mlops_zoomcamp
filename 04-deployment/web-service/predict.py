import pickle

with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds