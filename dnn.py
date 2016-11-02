import pandas
import numpy

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
cases3 = pandas.read_csv('echr_dataset/Article3/cases_a3.csv', header=None)
circumstances3 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_circumstances.csv', header=None)
full3 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_full.csv', header=None)
law3 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_law.csv', header=None)
procedure3 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_procedure.csv', header=None)
relevantLaw3 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_relevantLaw.csv', header=None)

cases6 = pandas.read_csv('echr_dataset/Article3/cases_a3.csv', header=None)
circumstances6 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_circumstances.csv', header=None)
full6 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_full.csv', header=None)
law6 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_law.csv', header=None)
procedure6 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_procedure.csv', header=None)
relevantLaw6 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_relevantLaw.csv', header=None)

cases8 = pandas.read_csv('echr_dataset/Article3/cases_a3.csv', header=None)
circumstances8 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_circumstances.csv', header=None)
full8 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_full.csv', header=None)
law8 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_law.csv', header=None)
procedure8 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_procedure.csv', header=None)
relevantLaw8 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_relevantLaw.csv', header=None)

cases = pandas.concat([cases3, cases6, cases8])
circumstances = pandas.concat([circumstances3, circumstances6, circumstances8])
full = pandas.concat([full3, full6, full8])
law = pandas.concat([law3, law6, law8])
procedure = pandas.concat([procedure3, procedure6, procedure8])
relevantLaw = pandas.concat([relevantLaw3, relevantLaw6, relevantLaw8])

x = pandas.concat([circumstances, full, law, procedure, relevantLaw], axis=1).as_matrix()
y = cases[1].values

# encode verdicts as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

# Model architecture
def DNN():
    model = Sequential()
    model.add(Dense(2048, input_dim=x.shape[1], init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('mlp', KerasClassifier(build_fn=DNN, nb_epoch=16, batch_size=64, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, x, encoded_y, cv=kfold, verbose=1)
print("Result: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

