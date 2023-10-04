import tensorflow_datasets as tfds

def load_data():
    (train_data, test_data), ds_info = tfds.load('stanford_dogs', split=['train', 'test'], with_info=True, as_supervised=True)
    return train_data, test_data, ds_info



if __name__ == "__main__":
    train_data, test_data, ds_info= load_data()
