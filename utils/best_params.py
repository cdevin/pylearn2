from collections import OrderedDict
import numpy
np = numpy
from pylearn2.train_extensions import TrainExtension
import theano
import theano.tensor as T
from pylearn2.utils import serial


class MonitorBasedBest(TrainExtension):
    """
    A callback that saves a copy of the model every time it achieves
    a new minimal value of a monitoring channel.
    """
    def __init__(self, channel_name, save_channel_names):
        """
        Parameters
        ----------
        channel_name: the name of the channel we want to minimize
        save_path: the path to save the best model to
        """

        self.__dict__.update(locals())
        del self.self
        self.best_cost = np.inf
        self.best_params = OrderedDict()

    def on_monitor(self, model, dataset, algorithm):
        """
        Looks whether the model performs better than earlier. If it's the
        case, saves the model.

        Parameters
        ----------
        model : pylearn2.models.model.Model
                model.monitor must contain a channel with name given by self.channel_name
        dataset : pylearn2.datasets.dataset.Dataset
            not used
        algorithm : TrainingAlgorithm
            not used
        """

        monitor = model.monitor
        channels = monitor.channels
        channel = channels[self.channel_name]
        val_record = channel.val_record
        new_cost = val_record[-1]

        if new_cost < self.best_cost:
            self.best_cost = new_cost
            for channel_name in self.save_channel_names:
                channel = channels[channel_name]
                self.best_params[channel_name] = channel.val_record[-1]

