import glob

class DatasetIterator(object):

    def __init__(self, data_path, which):
        self.files = glob.glob(data_path, "*{}*.pkl".format(which))
        self.current_index = 0
        self.num_files = len(self.files)

    def init_shared(self):
        data = serial.load(self.files[0])
        data_x = data.X
        data_y = data.y
        self.x, self.y = shared_dataset(self.x, self.y, cast_int = False)
        return self.x, self.y

    def __iter__(self):
        return self

    def next(self):
        if self.current_index < self.num_files:
            data = serial.load(self.files[self.current_index])
            data_y = data.y
            self.x.set_value(data.x, borrow = True)
            self.y.set_value(data.y, borrow = True)
            self.current_index += 1
            return data.x.shape[0]
        else:
            raise StopIteration



