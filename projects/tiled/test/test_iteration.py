

def test_format_setence():
    for k in xrange(20):
        data = [np.random.randint(-1,10) for i in range(12)]
        #data = [2, 6, 1, 8, 8, 7, 7, -1, 7, 6, 4, 7]
        #data = [3, 7, 6, 0, 7, 4, 4, -1, -1, 1, 0, 4]
        #data = [1, 5, -1, 3, 7, 8, 0, 2, 4, 3, 9, 3]
        print data
        #import ipdb
        #ipdb.set_trace()
        ind = np.random.randint(0,12)
        #ind = 5
        print ind
        print seq_len(data , ind, 5)

