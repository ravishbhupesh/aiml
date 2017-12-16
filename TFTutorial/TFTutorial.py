import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


min_max_scaler = preprocessing.MinMaxScaler()

class TFTutorial(object):
    def main(self):
        print(">>> main start...")
        input = self.load_data()
        print(input.tail())
        #self.plot_initial_data(input)
        input_nrm = self.normalize_data(input)
        print(input_nrm.head())

        no_of_features = len(input_nrm.columns)
        print(no_of_features)
        input_nrm_as_mtrx = input_nrm.as_matrix()

        print("<<< main end!")

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
