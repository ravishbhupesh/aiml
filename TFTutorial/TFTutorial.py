import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import math
import datetime

min_max_scaler = preprocessing.MinMaxScaler()

class TFTutorial(object):
    def main(self):
        print(">>> main start...")
        input = self.load_data()
        print(input.tail())
        #self.plot_initial_data(input)
        input_nrm = self.normalize_data(input)
        print(input_nrm.head())
        seq_len = 22
        X_train, y_train, X_test, y_test = self.prepare(input_nrm)
        print(X_train, X_test)
        shape = [4, seq_len, 1]  # feature, window, output
        neurons = [128, 128, 32, 1]
        d = 0.2
        model = self.create_model(shape, neurons, d)
        model.fit(X_train, y_train, batch_size=512, epochs=300, validation_split=0.1, verbose=1)
        self.model_score(model, X_train, y_train, X_test, y_test)

        p = self.percentage_difference(model, X_test, y_test)
        print("percentage_difference:", p)

        model.save('model.h5')


        print("<<< main end!")

    def percentage_difference(model, X_test, y_test):
        print(">>> percentage_difference start...")
        percentage_diff = []

        p = model.predict(X_test)
        for u in range(len(y_test)):  # for each data index in test data
            pr = p[u][0]  # pr = prediction on day u

            percentage_diff.append((pr - y_test[u] / pr) * 100)
        print(">>> percentage_difference end!")
        return p

    def model_score(model, X_train, y_train, X_test, y_test):
        print(">>> model_score start...")
        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
        print(">>> model_score end!")
        return trainScore[0], testScore[0]

    def create_model(self, layers, neurons, d = 0.2):
        print(">>> create_model start.....")
        model = Sequential()

        model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))

        model.add(LSTM(neurons[1], input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))

        model.add(Dense(neurons[2], kernel_initializer="uniform", activation='relu'))
        model.add(Dense(neurons[3], kernel_initializer="uniform", activation='linear'))
        # model = load_model('my_LSTM_stock_model1000.h5')
        # adam = keras.optimizers.Adam(decay=0.2)
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.summary()
        print("<<< create_model end!")
        return model

    def prepare(self, input_nrm, seq_len = 22):
        print(">>> prepare start...")
        no_of_features = len(input_nrm.columns)
        print(no_of_features)
        data = input_nrm.as_matrix()
        sequence_length = seq_len + 1  # index starting from 0
        result = []
        for index in range(len(data) - sequence_length):  # maxmimum date = lastest date - sequence length
            result.append(data[index: index + sequence_length])  # index : index + 22days
        result = np.array(result)
        row = round(0.9 * result.shape[0])  # 90% split
        train = result[:int(row), :]  # 90% date
        X_train = train[:, :-1]  # all data until day m
        y_train = train[:, -1][:, -1]  # day m + 1 adjusted close price
        X_test = result[int(row):, :-1]
        y_test = result[int(row):, -1][:, -1]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], no_of_features))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], no_of_features))
        print("<<< prepare end!")
        return X_train, y_train, X_test, y_test

    def normalize_data(self, input):
        print(">>> normalize data start...")
        df = pd.DataFrame(data = input, columns=['Date', 'Open Price', 'High Price', 'Low Price'])
        df['Open Price'] = min_max_scaler.fit_transform(df['Open Price'].values.reshape(-1,1))
        df['High Price'] = min_max_scaler.fit_transform(df['High Price'].values.reshape(-1, 1))
        df['Low Price'] = min_max_scaler.fit_transform(df['Low Price'].values.reshape(-1, 1))
        print("<<< normalize data end!")
        return df

    def load_data(self):
        print(">>> load_data start...")
        input = pd.read_csv("data/TCS_01012017_30062017.csv", parse_dates=['Date'])
        print(">>> load_data end!")
        return input

    def plot_initial_data(self, input):
        print(">>> plot initial data start...")
        # plt.plot(input['High Price']) #plots only y axis
        plt.plot(input['Date'], input['High Price'])  # plots x and y axis
        plt.plot(input['Date'], input['Low Price'])

        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.title("TCS Share Price")
        plt.show()
        print(">>> plot initial data end!")


if __name__ == "__main__":
    TFTutorial().main()
