from pylearn2.config import yaml_parse

train_mnist = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.mnist.MNIST {
        which_set: 'train',
        one_hot: 1,
        start: 0,
        stop: 50000
    },
    model: !obj:pylearn2.models.mlp.MLP {
        batch_size: 100,
        input_space: !obj:pylearn2.space.Conv2DSpace {
            shape: [28, 28],
            num_channels: 1
        },
        layers: [ !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                     layer_name: 'h2',
                     output_channels: 64,
                     irange: .05,
                     kernel_shape: [5, 5],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     max_kernel_norm: 1.9365
                 }, !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                     layer_name: 'h3',
                     output_channels: 64,
                     irange: .05,
                     kernel_shape: [5, 5],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     max_kernel_norm: 1.9365
                 }, !obj:pylearn2.models.mlp.Softmax {
                     max_col_norm: 1.9365,
                     layer_name: 'y',
                     n_classes: 10,
                     istdev: .05
                 }
                ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 100,
        learning_rate: .01,
        init_momentum: .5,
        monitoring_dataset:
            {
                'valid' : !obj:pylearn2.datasets.mnist.MNIST {
                              which_set: 'train',
                              one_hot: 1,
                              start: 50000,
                              stop:  60000
                          },
                'test'  : !obj:pylearn2.datasets.mnist.MNIST {
                              which_set: 'test',
                              one_hot: 1,
                          }
            },
        cost: !obj:pylearn2.costs.cost.SumOfCosts { costs: [
            !obj:pylearn2.costs.cost.MethodCost {
                method: 'cost_from_X',
                supervised: 1
            }, !obj:pylearn2.models.mlp.WeightDecay {
                coeffs: [ .00005, .00005, .00005 ]
            }
            ]
        },
        termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs : 0
        }
    },
    extensions:
        [ !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
            start: 1,
            saturate: 10,
            final_momentum: .99
        }
    ]
}

"""

train_samll = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.svhn.SVHN {
        which_set: 'train',
    },
    model: !obj:pylearn2.models.mlp.MLP {
        batch_size: 1,
        input_space: !obj:pylearn2.space.Conv2DSpace {
            shape: [32, 32],
            num_channels: 3
        },
        layers: [ !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                     layer_name: 'h2',
                     output_channels: 64,
                     irange: .05,
                     kernel_shape: [5, 5],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     max_kernel_norm: 1.9365
                 }, !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                     layer_name: 'h3',
                     output_channels: 64,
                     irange: .05,
                     kernel_shape: [5, 5],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     max_kernel_norm: 1.9365
                 }, !obj:pylearn2.models.mlp.Softmax {
                     max_col_norm: 1.9365,
                     layer_name: 'y',
                     n_classes: 10,
                     istdev: .05
                 }
                ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 1,
        learning_rate: .01,
        init_momentum: .5,
        monitoring_dataset:
            {
                'test'  : !obj:pylearn2.datasets.svhn.SVHN {
                              which_set: 'test',
                          }
            },
        cost: !obj:pylearn2.costs.cost.SumOfCosts { costs: [
            !obj:pylearn2.costs.cost.MethodCost {
                method: 'cost_from_X',
                supervised: 1
            }, !obj:pylearn2.models.mlp.WeightDecay {
                coeffs: [ .00005, .00005, .00005 ]
            }
            ]
        },
        termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs : 0
        }
    },
    extensions:
        [ !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
            start: 1,
            saturate: 10,
            final_momentum: .99
        }
    ]
}

"""

train_large = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.svhn.SVHN {
        which_set: 'extra',
        start: 0,
        stop: 5000,
        path: '/data/lisatmp2/mirzamom/data/SVHN/'
    },
    model: !obj:pylearn2.models.mlp.MLP {
        batch_size: 1,
        input_space: !obj:pylearn2.space.Conv2DSpace {
            shape: [32, 32],
            num_channels: 3
        },
        layers: [ !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                     layer_name: 'h2',
                     output_channels: 64,
                     irange: .05,
                     kernel_shape: [5, 5],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     max_kernel_norm: 1.9365
                 }, !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                     layer_name: 'h3',
                     output_channels: 64,
                     irange: .05,
                     kernel_shape: [5, 5],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     max_kernel_norm: 1.9365
                 }, !obj:pylearn2.models.mlp.Softmax {
                     max_col_norm: 1.9365,
                     layer_name: 'y',
                     n_classes: 10,
                     istdev: .05
                 }
                ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 1,
        learning_rate: .01,
        init_momentum: .5,
        monitoring_dataset:
            {
                'test'  : !obj:pylearn2.datasets.svhn.SVHN {
                              which_set: 'test',
                          }
            },
        cost: !obj:pylearn2.costs.cost.SumOfCosts { costs: [
            !obj:pylearn2.costs.cost.MethodCost {
                method: 'cost_from_X',
                supervised: 1
            }, !obj:pylearn2.models.mlp.WeightDecay {
                coeffs: [ .00005, .00005, .00005 ]
            }
            ]
        },
        termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs : 0
        }
    },
    extensions:
        [ !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
            start: 1,
            saturate: 10,
            final_momentum: .99
        }
    ]
}

"""



@profile
def run(yaml_f):
    train = yaml_parse.load(yaml_f)
    train.main_loop()

if __name__ == "__main__":
    #run(train_samll)
    run(train_large)
    #run(train_mnist)
