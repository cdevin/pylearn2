from pylearn2.datasets.mnist import MNIST


class IndexedZCA_Dataset(ZCA_Dataset):
    def __init__(self,
            preprocessed_dataset,
            preprocessor,
            indexes,
            convert_to_one_hot = True,
            start = None,
            stop = None,
            axes = ['b', 0, 1, 'c']):

        super(IndexedZCA_Dataset, self.__init__(preprocessed_dataset=preprocessed_dataset,
                                        preprocessor=preprocessor,
                                        convert_to_one_hot=convert_to_one_hot,
                                        start=start,
                                        stop=stop,
                                        axes=axes))

        if self.X is not None:
            self.X = self.X[indexes]
            self.y=self.y[indexes]




class ZCA_Dataset_BIN(ZCA_Dataset):
    def __init__(self,
            preprocessed_dataset,
            preprocessor,
            labels,
            convert_to_one_hot = True,
            start = None,
            stop = None,
            axes = ['b', 0, 1, 'c']):

        super(IndexedZCA_Dataset, self.__init__(preprocessed_dataset=preprocessed_dataset,
                                        preprocessor=preprocessor,
                                        convert_to_one_hot=convert_to_one_hot,
                                        start=start,
                                        stop=stop,
                                        axes=axes))

        if self.y is not None:
            new_y = np.zeros((self.y.shape[0], 2))
            for item in self.y:
                new_y[labels[np.argmax(item)]] = 1



        import ipdb
        ipdb.set_trace()
