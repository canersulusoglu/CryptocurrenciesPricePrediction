from DatasetLoader import DatasetLoader
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class Model():
    def __init__(self, dataset_loader:DatasetLoader, lookback=60, forecast=10, load_model=False):
        self.dataset_loader = dataset_loader
        self.load_model = load_model
        self.lookback = lookback
        self.forecast = forecast
        self.unchanged_data = dataset_loader.getDataset()
        self.training_data = self.unchanged_data.drop(['Date', 'Adj Close', 'Open', 'High', 'Low', 'Volume'], axis = 1)
        self.training_data['Close'] = self.training_data['Close'].fillna(0)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.training_data = self.scaler.fit_transform(self.training_data)
        self.model_checkpoint_dir = os.path.dirname(os.path.realpath(__file__)) + "/checkpoint/" + self.dataset_loader.currency + "/" + str(lookback) + "-" + str(forecast)
        self.results_dir = os.path.dirname(os.path.realpath(__file__)) + "/results/" + self.dataset_loader.currency + "/" + str(lookback) + "-" + str(forecast)
        if(not os.path.exists(self.model_checkpoint_dir)):
            os.makedirs(self.model_checkpoint_dir)
        if(not os.path.exists(self.results_dir)):
            os.makedirs(self.results_dir)
        self.model_checkpoint = ModelCheckpoint(
            filepath=self.model_checkpoint_dir + "/model.h5",
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=True
        )
        self.model = self.__createModel()

    def __createTrainSet(self):
        X_train = []
        y_train = []

        for i in range(self.lookback, len(self.training_data)):
            X_train.append(self.training_data[i - self.lookback : i])
            y_train.append(self.training_data[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        return X_train, y_train

    def __createModel(self):
        model = None
        if self.load_model:
            model = load_model(self.model_checkpoint_dir + "/model.h5")
        else:
            model = Sequential()
            model.add(LSTM(units=50, input_shape=(self.lookback , 1), return_sequences=True, activation="tanh"))
            model.add(LSTM(units=60, return_sequences=True, activation="tanh"))
            model.add(LSTM(units=80, return_sequences=True, activation="tanh"))
            model.add(LSTM(units=120, activation="tanh"))
            model.add(Dense(units=self.forecast))
            model.compile(
                optimizer='adam',
                loss='mean_squared_error'
            )
        model.summary()
        return model
        

    def trainModel(self, batch_size=50, epochs=10, validation_split=0.1):
        X_train, y_train = self.__createTrainSet()
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[self.model_checkpoint]
        )
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title("Training and Validation Loss")
        # Save figure as image
        plt.savefig(self.results_dir + "/lossPlot.png")
        # Show figure in window
        plt.show()

    def predictModel(self, plot=True):
        # Past Predict
        X_test_past = []
        y_test_past = []
        for i in range (self.lookback, self.training_data.shape[0]):
            X_test_past.append(self.training_data[i-self.lookback:i]) 
            y_test_past.append(self.training_data[i, 0])
        X_test_past, y_test_past = np.array(X_test_past), np.array(y_test_past)
        y_pred_past = self.model.predict(X_test_past)

        y_test_past = 1 / self.scaler.scale_ * y_test_past
        y_pred_past = 1 / self.scaler.scale_ * y_pred_past

        # Future Predict
        X_test_future = self.training_data[-self.lookback:]
        X_test_future = X_test_future.reshape(1, self.lookback, 1)

        y_pred_future = self.model.predict(X_test_future).reshape(-1, 1)
        y_pred_future = self.scaler.inverse_transform(y_pred_future)

        if plot:
            # Past
            df_past = pd.DataFrame(columns=['Date', 'Actual', 'Predict'])
            df_past['Actual'] = y_test_past
            df_past['Date'] = pd.to_datetime(self.unchanged_data['Date']) + pd.Timedelta(self.lookback, unit='d')
            df_past['Predict'] = y_pred_past

            # Future
            df_future = pd.DataFrame(columns=['Date', 'Actual', 'Predict'])
            df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=self.forecast)
            df_future['Predict'] = y_pred_future.flatten()

            # Merge Past And Future
            results = df_past.append(df_future).set_index('Date')

            # plot the results
            plot = results.plot(
                figsize=(14,5),
                title='Bitcoin Price Prediction using RNN-LSTM'
            )

            # Save figure as image
            fig = plot.get_figure()
            fig.savefig(self.results_dir + "/predictPlot.png")
            # Show figure in window
            plt.show()