from pylearn2.config import yaml_parse
import argparse


def experiment(yaml_string):

    # now run yaml file with default train.py script
    train_obj = yaml_parse.load(yaml_string)
    train_obj.main_loop()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'conv trainer')
    parser.add_argument('-f', '--file')
    args = parser.parse_args()
    experiment(args.file)

