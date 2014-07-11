import numpy as np
cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t


def make_dataset(voc, files):

    """
    Construct the dataset

    Paramters
    ---------
        voc: dict
            Vocabulary dictionary
        files:
            Dataset text files
    """

    cdef int DATA_SIZE = 829250940
    cdef np.ndarray data = np.zeros([DATA_SIZE], dtype=DTYPE)
    #cdef np.ndarray sent_ends = np.zeros([DATA_SIZE], dtype=DTYPE)
    cdef ind = 0
    cdef end_ind = 0

    for file in files:
        print "Processing {}".format(file)
        with open(file, 'r') as file:
            for line in file.readlines():
                words = line.rstrip('/n').split(' ')
                for item in words:
                    try:
                        key = voc[item]
                    except KeyError:
                        key = voc['<UNK>']
                    data[ind] = key
                    ind += 1
                # end of sentence
                data[ind] = voc['</S>']
                #sent_ends[end_ind] = ind
                #end_ind += 1

    #return data, sent_ends
    return data



