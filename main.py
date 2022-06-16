from DatasetLoader import DatasetLoader
from Model import Model

if __name__ == '__main__':
    dataset_loader = DatasetLoader(
        currency='BTC-USD', 
        dataset_path='./dataset', 
        download_dataset=True, 
        use_downloaded_dataset=True
    )
    model = Model(
        dataset_loader=dataset_loader, 
        lookback=60, 
        forecast=30,
        load_model=True
    )
    model.trainModel(epochs=100)
    model.saveModel()
    predicted_values = model.predictModel(plot=True)