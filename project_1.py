#!/usr/bin/env python
# coding: utf-8

# # Step 1.

# ### Importing libraries
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ### Initializing the random number 

seed = 7
numpy.random.seed(seed)

# ### Loading Dataset

dataframe = pandas.read_csv("C:/Users/FC/Documents/Deep_Learning_Project_One/sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]


# ### Label Encoding

encoded_Y = LabelEncoder().fit_transform(Y)

# ## Step 2. Baseline Neural Network Model Performance

def create_baseline():
    # create model, write code below
    model = Sequential()
    model.add(Dense(60, activation='relu', input_shape=(60,)))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model, write code below
    model.compile(optimizer='Adam',loss= 'binary_crossentropy', metrics=["accuracy"])
    return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5,verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## **Step 3. Re-Run The Baseline Model With Data Preparation**

def create_baseline():
    # create model, write code below
    model = Sequential()
    model.add(Dense(60, activation='relu', input_shape=(60,)))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model, write code below
    model.compile(optimizer='Adam',loss= 'binary_crossentropy', metrics=["accuracy"])
    return model


# evaluate baseline model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## Step 4. Tuning Layers and Number of Neurons in The Model

# ### Evaluating a Smaller Network

# smaller model
def create_smaller():
    # create model
    model = Sequential()
    model.add(Dense(30, activation='relu', input_shape=(60,)))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(optimizer='Adam',loss= 'binary_crossentropy', metrics=["accuracy"])
    return model
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ### Evaluating a Larger Network

# larger model
def create_larger():
    # create model
    model = Sequential()
    model.add(Dense(60, activation='relu', input_shape=(60,)))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(optimizer='Adam',loss= 'binary_crossentropy', metrics=["accuracy"])
    return model
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## Step 5: Really Scaling up: developing a model that overfits

# overfit model
def create_larger():
    # create model
    model = Sequential()
    model.add(Dense(600, activation='relu', input_shape=(60,)))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(optimizer='Adam',loss= 'binary_crossentropy', metrics=["accuracy"])
    return model
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=150, batch_size=5,verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## Step 6: Tuning the Model

# tuned model
def tuned_model():
    model = Sequential()
    model.add(Dense(30, activation='relu', input_shape=(60,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model, write code below
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

    return model

# evaluate baseline model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=tuned_model, epochs=50, batch_size=5,verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## Step 7: Rewriting the code using the Keras (Functional API)

import keras
from keras import layers
def funtional_model():
    # create model, write code below
    inputs = keras.Input(shape=(60,))
    x = layers.Dense(60,activation='relu')(inputs)
    outputs = layers.Dense(1,activation='sigmoid')(x)
    model = keras.Model(inputs,outputs)
    # Compile model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=funtional_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Functional Api: Accuracy %.2f%%, Error (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## Step 7: Rewriting the code using the Keras (Model Subclassing)

import keras
from keras import layers
class Mymodel(keras.Model):
    def__init__(self):
        super(Mymodel).__init__()
        self.dense1 = Dense(60, activation = 'relu')
        self.dense2 = Dense(1, activation = 'sigmoid')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    

def subclass_model():
    inputs = keras.Input(shape=(60,))
    model = MyModel()
    outputs = model.call(inputs)
    
    model = keras.Model(inputs, outputs)
    
    # Compile model, write code below
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    
    return model
  numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=subclass_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Model Subclassing: Accuracy %.2f%%, Error (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## Step 9: Rewriting the code without using scikit-learn

data = pandas.read_csv("C:/Users/FC/Documents/Deep_Learning_Project_One/sonar.csv", header=None)

data[60][data[60]=='M']=1
data[60][data[60]=='R']=0

data_values = data.values
numpy.random.seed(7)
numpy.random.shuffle(data_values)

train_data = data_values[:167,0:60].astype(float)
train_labels =data_values[:167,60]
test_data= data_values[167:,0:60].astype(float)
test_labels= data_values[167:,60]

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

def keras_model():
    # create model, write code below
    model = models.Sequential()
    model.add(layers.Dense(60, activation='relu', input_shape=(60,)))
    model.add(layers.Dense(1,activation='sigmoid'))
    # Compile model, write code below
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

import numpy as np
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_labels[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([train_labels[:i * num_val_samples],train_labels[(i + 1) * num_val_samples:]],axis=0)
    model = keras_model()
    model.fit(partial_train_data, partial_train_targets,epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

result = model.model.evaluate(test_data,test_labels)
print(result)

