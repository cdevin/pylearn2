

class CIFAR10_Bin(CIFAR10)
     def __init__(self, which_set, center = False, rescale = False, gcn = None,
             one_hot = False, start = None, stop = None, axes=('b', 0, 1, 'c'),
            toronto_prepro = False, preprocessor = None):


         super(CIFAR10_Bin, self).__init__(which_set = which_set,
                                            center = center,
                                            rescale = rescale,
                                            gcn = gcn,
                                            one_hot = on)
