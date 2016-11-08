import pandas
import numpy

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, TensorBoard

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder


# load dataset
cases3 = pandas.read_csv('echr_dataset/Article3/cases_a3.csv', header=None)
circumstances3 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_circumstances.csv', header=None)
relevantLaw3 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_relevantLaw.csv', header=None)
full3 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_full.csv', header=None)
law3 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_law.csv', header=None)
procedure3 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_procedure.csv', header=None)
relevantLaw3 = pandas.read_csv('echr_dataset/Article3/ngrams_a3_relevantLaw.csv', header=None)

cases6 = pandas.read_csv('echr_dataset/Article6/cases_a6.csv', header=None)
circumstances6 = pandas.read_csv('echr_dataset/Article6/ngrams_a6_circumstances.csv', header=None)
relevantLaw6 = pandas.read_csv('echr_dataset/Article6/ngrams_a6_relevantLaw.csv', header=None)

cases8 = pandas.read_csv('echr_dataset/Article8/cases_a8.csv', header=None)
circumstances8 = pandas.read_csv('echr_dataset/Article8/ngrams_a8_circumstances.csv', header=None)
relevantLaw8 = pandas.read_csv('echr_dataset/Article8/ngrams_a8_relevantLaw.csv', header=None)

X = pandas.concat((circumstances3, relevantLaw3), axis=1)
y = cases3[1]

X = X.as_matrix()
y = y.values

# encode verdicts as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)


# neural network builder
def build_fn(n_hidden_units, init, W_regularization_weight, b_regularization_weight, activation, dropout, optimizer):
    model = Sequential()

    # input layer
    model.add(Dense(n_hidden_units[0], init=init, input_dim=X.shape[1], W_regularizer=l2(W_regularization_weight), b_regularizer=l2(b_regularization_weight)))
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # hidden layers
    for n_hidden_unit in n_hidden_units[1:]:
        model.add(Dense(n_hidden_unit, init=init, W_regularizer=l2(W_regularization_weight), b_regularizer=l2(b_regularization_weight)))
        model.add(Activation(activation))
        model.add(Dropout(dropout))

    # output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # build network
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# neural network parameters
param_grid = dict(
    nb_epoch=numpy.array([64]),
    batch_size=numpy.array([8]),
    n_hidden_units=[[64, 64, 64], ],  # [64, 64, 64]
    init=['uniform'],
    W_regularization_weight=[0.000003, 0.0002],
    b_regularization_weight=[0.00003, ],
    activation=['relu'],
    dropout=[0.5],
    optimizer=['adam'])

# Tensor Board
tensor_board = TensorBoard(log_dir='./TensorBoard')

# Early stopping
early_stopping = EarlyStopping(monitor='acc', patience=8, verbose=1)

# search parameter space for best neural network architecture, fix internal random state for reproducibility
classifier = KerasClassifier(build_fn=build_fn, verbose=1)
skfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=8)
grid_search = GridSearchCV(classifier, param_grid, cv=skfold, verbose=1, fit_params={'callbacks': [tensor_board, early_stopping]})
grid_result = grid_search.fit(X, encoded_y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

