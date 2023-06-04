import pandas as pd

class CollectData:

    def __init__(self) -> None:
        self.TYPE_LABEL = {'clicks':0, 'carts':1, 'orders':2}

    # def collect_validataion_data(self):
    #     '''validation用のchunckファイル収集データ
    #     '''
    #     train_val_paths = glob.glob(os.path.join(DATA_DIR, 'validation', 'train_parquet', '*'))
    #     test_val_paths = glob.glob(os.path.join(DATA_DIR, 'validation', 'test_parquet', '*'))
    #     test_label = os.path.join(DATA_DIR, 'validation', 'test_labels.parquet')
    #     validation_train = {}
    #     # validation_test = {}
    #     for p in train_val_paths: validation_train[p] = read_file_to_cache(p)
    #     # for p in test_val_paths: validation_test[p] = read_file_to_cache(p)
    #     # validation_label = pd.read_parquet(test_label)
        



    def _read_file_to_cache(self, f):
        df = pd.read_parquet(f)
        df.ts = (df.ts/1000).astype('int32')
        df['type'] = df['type'].map(self.TYPE_LABEL).astype('int8')
        return df

