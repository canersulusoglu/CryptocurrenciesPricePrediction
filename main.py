import argparse
from DatasetLoader import DatasetLoader
from Model import Model

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Neural network model training for cryptocurrencies price prediction.')
    subparsers = main_parser.add_subparsers(required=True)
    
    # Train Args
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('-c', '--currency', nargs='?', type=str, default="BTC-USD")
    train_parser.add_argument('-d', '--download_dataset', nargs='?', type=bool, default=False, const=True)
    train_parser.add_argument('-udd', '--use_downloaded_dataset', nargs='?', type=bool, default=False, const=True)
    train_parser.add_argument('-lm', '--load_model', nargs='?', type=bool, default=False, const=True)
    train_parser.add_argument('-e', '--epoch', nargs='?', type=int, default=200)
    train_parser.add_argument('-l', '--lookback', nargs='?', type=int, default=60)
    train_parser.add_argument('-f', '--forecast', nargs='?', type=int, default=30)
    train_parser.set_defaults(subparser='train')

    # Test Args
    test_parser = subparsers.add_parser('test')
    test_parser.set_defaults(subparser='test')
    test_parser.add_argument('-c', '--currency', nargs='?', type=str, default="BTC-USD")
    test_parser.add_argument('-l', '--lookback', nargs='?', type=int, default=60)
    test_parser.add_argument('-f', '--forecast', nargs='?', type=int, default=30)
    test_parser.set_defaults(subparser='test')

    args = main_parser.parse_args()

    try:
        if(args.subparser == 'train'):
            currency = args.currency
            download_dataset = args.download_dataset
            use_downloaded_dataset = args.use_downloaded_dataset
            load_model = args.load_model
            epoch = args.epoch
            lookback = args.lookback
            forecast = args.forecast

            dataset_loader = DatasetLoader(
                currency=currency, 
                dataset_path='./dataset', 
                download_dataset=download_dataset, 
                use_downloaded_dataset=use_downloaded_dataset 
            )
            model = Model(
                dataset_loader=dataset_loader, 
                lookback=lookback, 
                forecast=forecast,
                load_model=load_model
            )
            model.trainModel(epochs=epoch)

        elif(args.subparser == 'test'):
            currency = args.currency
            lookback = args.lookback
            forecast = args.forecast

            dataset_loader = DatasetLoader(
                currency=currency, 
                dataset_path='./dataset', 
                download_dataset=False, 
                use_downloaded_dataset=True 
            )
            model = Model(
                dataset_loader=dataset_loader, 
                lookback=lookback, 
                forecast=forecast,
                load_model=True
            )
            model.predictModel(plot=True)
        else:
            raise Exception("Command not found!")
    except Exception as error:
        print(error)