import random
import itertools
import numpy as np


def grid_search(hyper_params: dict):
    hyper_param_names = list(hyper_params.keys())
    hyper_param_lists = [hyper_params[key] for key in hyper_param_names]
    for params in itertools.product(*hyper_param_lists):
        yield dict((hyper_param_names[i], params[i]) for i in range(len(hyper_param_names)))


def split(datas: list, is_random=False, train_pro=0.9):
    indexes = list(range(len(datas)))
    if is_random:
        random.shuffle(indexes)
    train_indexes, test_indexes = indexes[:int(train_pro * len(datas))], indexes[int(train_pro * len(datas)):]
    return [datas[i] for i in train_indexes], [datas[i] for i in test_indexes]



'''
各变量命名参考文档
参数：
    D:list， 数据集（如每条数据的id）
    hyper_params:dict，超参数(str)以及超参数对应的搜索空间(list)（如{'learning_rate':[1e-5,1e-4,1e-3],'emb':['bert','elmo','robert'],'topk':[1,2,3,4,5]}）
    train:func，训练函数，参数列表为（train_datas, valid_datas, **hyper_params），返回值为模型（或模型地址，依赖train具体实现）
    evaluate:func，测试函数，参数列表为（eval_datas, model, **hyper_params），返回值为测试分数
    k:int，折数
    R:int，内折数
    random_seed:int，随机数种子
返回值：
    result:float，k折平均结果
    std:float，k折结果标准差
'''
def cross_valid(D: list, hyper_params: dict, train, evaluate, k=5, R=3, random_seed=1):
    random.seed(random_seed)
    results = []
    for i in range(k):
        print(f'start fold {i}...')
        TEST_i = D[i * len(D) // k:(i + 1) * len(D) // k]
        TV_i = D[:i * len(D) // k] + D[(i + 1) * len(D) // k:]
        TRAIN_i, VALID_i = split(TV_i, is_random=False)
        THETA = grid_search(hyper_params)
        score_best = 0.0
        theta_i = None
        for theta in THETA:
            model = train(TRAIN_i, VALID_i, **theta)
            score = evaluate(VALID_i, model, **theta)
            if score > score_best:
                score_best = score
                theta_i = theta
        assert theta_i
        print(f'best hyper-params in fold {i}: {theta_i}')
        result_i = []
        for j in range(R):
            TRAIN_ij, VALID_ij = split(TV_i, is_random=True)
            model = train(TRAIN_ij, VALID_ij, **theta_i)
            result_ij = evaluate(TEST_i, model, **theta_i)
            result_i.append(result_ij)
        result_i = sum(result_i) / len(result_i)
        results.append(result_i)
        print(f'score in fold {i}: {result_i}\n\n')
    result = np.mean(results)
    std = np.std(results)
    print(f'final score/std over all folds: {result}/{std}')
    return result, std
