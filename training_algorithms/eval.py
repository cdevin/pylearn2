import numpy

def Eval(model, datasets, batch_size):

    """ Evauate performance of a model without any learning """

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # get the training, validation and testing function for the model
    print '... getting the evaluation functions'
    train_fn, validate_model, test_model = model.build_finetune_functions(
                datasets=datasets,
                batch_size=batch_size,
                w_l1_ratio = 0,
                act_l1_ratio = 0,
                enable_momentum = False)

    validation_losses = validate_model()
    validation_score = numpy.mean(validation_losses)


    test_losses = test_model()
    test_score = numpy.mean(test_losses)

    return test_score * 100., validation_score * 100.


