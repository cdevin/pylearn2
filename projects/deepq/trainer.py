from pylearn.config import yaml_parse


class DividedDataset(object):

    def __init__(self, base, num_division):
        self.base = yaml_parse.load(base)

        ind = range(self.base.num_examples)
        self.rng.shuffle(ind)
        self.division_indexs = []


    def get_random_split(self, ind):

        if len(self.division_indexs):
            raise ValueError("Split first")

        ds = self.base.copy()
        ds.X = ds.X[self.division_indexs[ind]]
        ds.y = ds.y[self.division_indexs[ind]]
        return ds

    def split_randomly(self):
        num_examples = self.num_examples
        ind = range(num_examples)
        self.rng.shuffle(ind)

        ds = []
        for i in xrange(self.num_division - 1):
            stride = num_examples / self.num_division
            self.division_indexs.append(ind[i * string : (i+1) * stride])


class TrainHME(obejct):
    """
    Train hard mixture of experts
    """


    def __init__(self, dataset, expert, num_experts, exprt_algo gater):
        """

        Parameters
        ----------
        expert: yaml file
            expert yaml file with its own training alg
        num_experts: int
            number of experts
        gater: object
            gater
        """



        # devide data
        self.dataset = DividedDataset(dataset, num_experts)
        self.dataset.split_randomly()

        # construct experts
        self.experts = []
        for i in xrange(num_experts):
            # assign dataset

            expert_ = expert % ({'expert_index' : i})
            expert_ = yaml_parse.load(expert_)
            expert_.data_index = = self.data_assignment

        self.gater = yaml_parse(gater)
        self.gater.experts = self.experts
        self.epoch = 0

    def main_loop(self, dataset):

        while True:
            # train each expert
            # TODO use mpi to parallerize this for loop
            for i in xrange(len(self.experts)):
                self.experts[i] = train_expert(self.experts[i])

            # train gater
            gater.main_loop()

            # devide data


            self.epoch += 1
            if not self.conitune_learning():
                break



    def conitune_learning(self):
        if self.termination_criterion is None:
            return True
        else:
            return self.termination_criterion.conitune_learning(None)


    def train_expert(self, expert):

        # load the model
        assert isinstance(exper, string)
        if expert[-3:] == 'pkl':
            expert = serial.load(expert)
        elif expert[-3:] == 'yaml'
            expert = yaml_parse.load(expert)
        else:
            raise ValueError("Unsupported file format")

        # load the data


        # load the trainer
        expert_trainer = serial.load(self.expert_trainer)
        expert_trainer.model = expert

        # train
        expert.main_loop()

        # update expert with the new trained one
        return epxerTODO

    def divide_data(self, random = False):

