import random
import tensorflow as tf
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import pandas
import cv2 as cv

from concurrent.futures import ThreadPoolExecutor


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras with an imgaug sequence"""

    def __init__(self,
                 dataframe,
                 x_col,
                 y_col,
                 aug_sequence,
                 reduction=0.0,
                 batch_size=32,
                 image_size=224,
                 shuffle=True,
                 balance_dataset=True):
        if reduction != 0.0 and balance_dataset is False:
            raise Exception("Cannot set reduction without balancing dataset.")

        self.df = dataframe if not balance_dataset else self.balance_dataset(dataframe, y_col, x_col, reduction)
        self.x_col = x_col
        self.y_col = y_col
        self.df_index = self.df.index.tolist()
        self.labels = self.extract_labels()
        self.indexes = np.arange(len(self.df_index))
        self.size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug_sequence = aug_sequence
        self.future_data_provider = None
        self.current_index = 0
        self.prefetch = False

        print("Generator initialized")
        print(f"Got {len(self.df)} images in {len(self.labels)} classes.")
        print(f"Samples per class: {self.df.groupby('label').count().to_string()}")

        self.on_epoch_end()

    def __getitem__(self, index):
        """Generate one batch of data"""
        if self.prefetch:
            data_batch = self.future_data_provider.result()
            self.current_index = (self.current_index + 1) % self.__len__()
            self.__prefetch_data(self.current_index)

            return data_batch

        return self.__get_data(index)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.df_index) // self.batch_size

    def reduce(self, dataframe, reduction):
        if reduction > 0.0:
            reduction_count = int(len(dataframe) * reduction)
            drop_indices = np.random.choice(dataframe.index, reduction_count, replace=False)
            return dataframe.drop(drop_indices).reset_index()
        return dataframe

    def balance_dataset(self, dataframe, label_column, path_column, reduction):
        count_per_label = \
            dataframe.value_counts(subset=[label_column]).reset_index(name='count').set_index(label_column)[
                'count'].to_dict()
        upper_limit = int((1.0 - reduction) * max(count_per_label.values()))
        grouped_by_label = dataframe.groupby(label_column)[path_column].apply(list).reset_index(name='paths')

        for index, row in grouped_by_label.iterrows():
            label_name = row[label_column]
            current_count = count_per_label[label_name]
            missing = upper_limit - current_count

            if missing < 0:
                indices = dataframe.index[dataframe[label_column] == label_name]
                drop_indices = np.random.choice(indices, abs(missing), replace=False)
                dataframe = dataframe.drop(drop_indices).reset_index(drop=True)
            elif missing > 0:
                repeated = random.choices(row['paths'], k=missing)
                extension = pandas.DataFrame([[label_name, path] for path in repeated],
                                             columns=[label_column, path_column])
                dataframe = pandas.concat([dataframe, extension])

        return dataframe

    def extract_labels(self):
        labels = self.df[self.y_col].unique().tolist()
        labels = [str(x) for x in labels]
        labels.sort()
        return labels

    def get_all_classes(self, one_hot=False):
        y_true_list = []
        for label in self.df['label']:
            y_true = self.__to_one_hot(label) if one_hot else str(label)
            y_true_list.append(y_true)

        return np.array(y_true_list)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.df_index))
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indexes)
        if self.prefetch:
            self.__prefetch_data(0)

    def __prefetch_data(self, index):
        with ThreadPoolExecutor(max_workers=1) as executor:
            self.future_data_provider = executor.submit(self.__get_data, index)

    def __to_one_hot(self, label):
        encoding = np.zeros((len(self.labels)))
        encoding[self.labels.index(label)] = 1.0
        return encoding

    def one_hot_to_label(self, one_hot):
        index = np.argmax(one_hot)
        return self.labels[index]

    def __get_data(self, index):
        # X.shape : (batch_size, *dim)
        index = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch = [self.df_index[k] for k in index]

        labels = []
        for k in batch:
            label = self.df.iloc[k][self.y_col]
            label_encoding = self.__to_one_hot(label)
            labels.append(label_encoding)

        labels = np.array(labels)
        images = self.resize_with_pad([cv.imread(self.df.iloc[k][self.x_col]) for k in batch], image_size=self.size)

        if self.aug_sequence is not None:
            images = self.aug_sequence.augment_images(images)

        return np.stack(images), labels

    @staticmethod
    def resize_with_pad(images, image_size=224):
        return iaa.Sequential([
            iaa.Resize({"longer-side": image_size, "shorter-side": "keep-aspect-ratio"}),
            iaa.PadToFixedSize(width=image_size, height=image_size)])(images=images)