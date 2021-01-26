import random

from cross_valid import cross_valid


def train(train_datas, valid_datas, hp1, hp2, hp3):
    print(f'train with hp1:{hp1}, hp2:{hp2}, hp3:{hp3}, ')
    print(f'train_datas:{train_datas},valid_datas:{valid_datas}')
    model = None
    return model


def evaluate(datasets, model, hp1, hp2, hp3):
    print(f'eval with hp1:{hp1}, hp2:{hp2}, hp3:{hp3}, ')
    print(f'eval_datas:{datasets}')
    return random.random()


if __name__ == '__main__':
    datasets = [11, 22, 33, 44, 55, 66, 77, 88, 99]
    hyper_params = {'hp1': [0.1, 0.2, 0.3], 'hp2': [4, 5, 6, 7], 'hp3': ['bert', 'elmo']}
    cross_valid(datasets, hyper_params, train, evaluate)
