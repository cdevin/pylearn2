from pylearn.config import yaml_parse

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
        self.data_assignment = self.divide_data()

        # construct experts
        self.experts = []
        for i in xrange(num_experts):
            # assign dataset

            expert_ = expert % ({'expert_index' : i})
            expert_ = yaml_parse.load(expert_)
            expert_.data_index = = self.data_assignment

        self.gater = yaml_parse(gater)
        self.gater.experts = self.experts

    def main_loop(self, dataset):

        while True:
            # train each expert
            # TODO use mpi to parallerize this for loop
            for expert in self.experts:
                expert.main_loop()

            # train gater
            gater.main_loop()

            # devide data



            if not self.conitune_learning():
                break



    def conitune_learning(self):
        if self.termination_criterion is None:
            return True
        else:
            return self.termination_criterion.conitune_learning(None)

