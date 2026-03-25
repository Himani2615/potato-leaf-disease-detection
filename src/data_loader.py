import tensorflow as tf

IMAGE_SIZE = 256
BATCH_SIZE = 32

def load_data(data_dir):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        seed=123,
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )
    return dataset


def split_data(dataset):
    def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1):
        ds_size = len(ds)

        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)

        return train_ds, val_ds, test_ds

    return get_dataset_partitions_tf(dataset)