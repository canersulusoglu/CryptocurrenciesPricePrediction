import requests
import pytz
import datetime
import pandas as pd
import numpy as np
import io
import os

class DatasetLoader():
    def __init__(self, currency='BTC-USD', dataset_path='./dataset', download_dataset=False, use_downloaded_dataset=False):
        self.dataset_path = os.path.dirname(os.path.realpath(__file__)) + dataset_path + "/" + currency + ".csv"
        self.download_dataset = download_dataset
        self.use_downloaded_dataset = use_downloaded_dataset
        self.currency = currency

    def getDataset(self):
        if self.use_downloaded_dataset and os.path.exists(self.dataset_path):
            return pd.read_csv(self.dataset_path)
        else:
            base = 'https://query1.finance.yahoo.com/v7/finance/download/'
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
                'Accept': 'text/csv;charset=utf-8'
            }
            fromDate = 0 # Beginning of currency
            toDate = int(datetime.datetime.now(pytz.timezone('UTC')).timestamp()) # Current timestamp according to UTC timezone.
            params = { 
                'period1': str(fromDate),
                'period2': str(toDate),
                'interval': '1d',
                'events': 'history',
                'includeAdjustedClose': 'true'
            }
            response = requests.request("GET", base + self.currency, headers=headers, params=params)
            if response.status_code == 200:
                if self.download_dataset:
                    open(self.dataset_path , 'wb').write(response.content)
                return pd.read_csv(io.StringIO(response.content.decode('utf-8')))
