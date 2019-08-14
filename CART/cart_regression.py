import numpy as np

def load_data_set():
    dataSet = []
    f = open('regData.txt')
    fr = f.readlines()
    for line in fr:
        line = line.strip().split('\t')
        linef = [float(li) for li in line]
        dataSet.append(linef)
    dataSetMat = np.mat(dataSet)
    return dataSetMat

def calc_err(data_set: np.array) -> float:
    error = np.var(data_set[:, -1]) * np.shape(data_set)[0]
    return error


def choose_best_feature_val2split(data_set_mat: np.array):
    """
    :param data_set:
    :return: best_feature, best_val( default None, None)
    """
    if(len(set(data_set_mat[:,-1].T.tolist()[0])) == 1): return None, None
    best_feature = 0
    best_val = 0
    lowest_err = 1000000
    total_err = calc_err(data_set_mat)

    for feature in range(np.shape(data_set_mat)[1] - 1):
        all_values = [d[feature] for d in data_set_mat.tolist()]
        values = set(all_values)
        # 选取一个分界值
        for value in values:
            left_child, right_child = split_by_fature_val(feature, value, data_set_mat)
            if(np.shape(left_child)[0] == 0 or np.shape(right_child)[0] == 0): continue
            cur_err = calc_err(left_child) + calc_err(right_child)
            if(cur_err < lowest_err):
                best_feature = feature
                best_val = value
                lowest_err = cur_err
    if(total_err - lowest_err < 1): return None, None
    return best_feature, best_val


def split_by_fature_val(feature, value, data_set_mat):
    """

    :param feature:
    :param value:
    :param data_set:
    :return: np.array
    """
    # np.nonzero(a) 返回的索引值数组是一个2维tuple数组，该tuple数组中包含一维的array数组。其中，一维array向量的个数与a的维数是一致的。
    left_child = data_set_mat[np.nonzero(data_set_mat[:, feature] > value)[0], :]
    right_child = data_set_mat[np.nonzero(data_set_mat[:, feature] <= value)[0], :]
    return left_child, right_child



def create_classfication_tree(data_set_mat: np.mat):
    feature, value = choose_best_feature_val2split(data_set_mat)
    if feature == None: return np.mean(data_set_mat[:, -1])
    regression_tree = {}
    regression_tree['feature_index'] = feature
    regression_tree['value'] = value
    left_child, right_child = split_by_fature_val(feature, value, data_set_mat)
    regression_tree['left_child'] = create_classfication_tree(left_child)
    regression_tree['right_child'] = create_classfication_tree(right_child)
    return regression_tree

if __name__ == '__main__':
    data_set_mat = load_data_set()
    regression_tree = create_classfication_tree(data_set_mat)
    print(regression_tree)



