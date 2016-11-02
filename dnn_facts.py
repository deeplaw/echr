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
relevantLaw3 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_relevantLaw.csv', header=None)

cases6 = pandas.read_csv('echr_dataset/Article3/cases_a3.csv', header=None)
circumstances6 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_circumstances.csv', header=None)
relevantLaw6 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_relevantLaw.csv', header=None)

cases8 = pandas.read_csv('echr_dataset/Article3/cases_a3.csv', header=None)
circumstances8 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_circumstances.csv', header=None)
relevantLaw8 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_relevantLaw.csv', header=None)

cases = pandas.concat([cases3, cases6, cases8])
circumstances = pandas.concat([circumstances3, circumstances6, circumstances8])
relevantLaw = pandas.concat([relevantLaw3, relevantLaw6, relevantLaw8])

x = circumstances+relevantLaw
y = cases[1]

# shuffle and convert to numpy arrays
perm = numpy.random.permutation(numpy.arange(len(x)))
x_shuffled = x.iloc[perm].as_matrix()
y_shuffled = y.iloc[perm].values

# encode verdicts as integers
encoder = LabelEncoder()
encoder.fit(y_shuffled)
encoded_shuffled_y = encoder.transform(y_shuffled)

# model architecture
def DNN():
    model = Sequential()
    model.add(Dense(256, input_dim=x.shape[1], init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('mlp', KerasClassifier(build_fn=DNN, nb_epoch=64, batch_size=32, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, x_shuffled, encoded_shuffled_y, cv=kfold, verbose=1)
print("Result: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

