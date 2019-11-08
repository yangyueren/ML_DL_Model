import numpy as np

def load_data_set():
    # dataSet = [[0, 0, 0],
    #            [0, 1, 1],
    #            [0, 2, 0],
    #            [1, 0, 1],
    #            [1, 1, 1],
    #            [1, 2, 1],
    #            [2, 0, 0],
    #            [2, 1, 1],
    #            [2, 2, 0]]
    dataSet = [
        [1,1,1],
        [1,0,0],
        [0,1,0],
        [0,0,0],
        [1,0,0]
    ]
    labels = ['color','shape']
    return dataSet, labels

def calc_gini(data_set: np.array) -> float:
    total_num = np.shape(data_set)[0]
    label_num = {}
    gini = 1.0
    for data in data_set:
        label = data[-1]
        if label in label_num:
            label_num[label] += 1
        else:
            label_num[label] = 1
    for key in label_num:
        p = label_num[key] / total_num
        gini -= p*p
    return gini


def choose_best_feature_val2split(data_set: np.array):
    """
    :param data_set:
    :return: best_feature, best_val( default None, None)
    """
    if(len(data_set[0]) == 1): return None, None
    if(len(set(d[-1] for d in data_set)) == 1): return None, None
    best_feature = 0
    best_val = 0
    lowest_guni = 1000000
    total_gini = calc_gini(data_set)
    total_num = np.shape(data_set)[0]

    # 选择颜色这个feature
    for feature in range(np.shape(data_set)[1] - 1):
        all_values = [d[feature] for d in data_set]
        values = set(all_values)
        # 选取一个分界值
        for value in values:
            left_child, right_child = split_by_fature_val(feature, value, data_set)
            if(np.shape(left_child)[0] == 0 or np.shape(right_child)[0] == 0): continue
            left_num = np.shape(left_child)[0]
            right_num = np.shape(right_child)[0]
            cur_guni = left_num / total_num * calc_gini(left_child) + right_num / total_num * calc_gini(right_child)
            if(cur_guni < lowest_guni):
                best_feature = feature
                best_val = value
                lowest_guni = cur_guni
    if(total_gini - lowest_guni < 0.00001): return None, None
    return best_feature, best_val


def split_by_fature_val(feature, value, data_set):
    """

    :param feature:
    :param value:
    :param data_set:
    :return: np.array
    """
    data_set = np.mat(data_set)
    # np.nonzero(a) 返回的索引值数组是一个2维tuple数组，该tuple数组中包含一维的array数组。其中，一维array向量的个数与a的维数是一致的。
    left_child = data_set[np.nonzero(data_set[:, feature] <= value)[0], :].tolist()
    right_child = data_set[np.nonzero(data_set[:, feature] > value)[0], :].tolist()
    return left_child, right_child


def check_is_one_category(new_data_set: np.array):
    """

    :param new_data_set:
    :return: flag
    """
    flag = False
    category_list = [data[-1] for data in new_data_set]
    category = set(category_list)
    if(len(category) == 1):
        flag = True
    return flag

def majority_category(new_data_set: np.array):
    category_count = {}
    category_list = [data[-1] for data in new_data_set]
    for c in category_list:
        if c not in category_count:
            category_count[c] = 1
        else:
            category_count[c] += 1
    # sorted() items()是key value对，x[1]是选择value
    sorted_category = sorted(category_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_category[0][0]

def create_classfication_tree(data_set: np.array):
    feature, value = choose_best_feature_val2split(data_set)
    if feature == None and check_is_one_category(data_set):
        return data_set[0][-1]
    if feature == None and not check_is_one_category(data_set):
        return majority_category(data_set)
    classfication_tree = {}
    classfication_tree['feature_index'] = feature
    classfication_tree['value'] = value
    left_child, right_child = split_by_fature_val(feature, value, data_set)
    classfication_tree['left_child'] = create_classfication_tree(left_child)
    classfication_tree['right_child'] = create_classfication_tree(right_child)
    return classfication_tree

def prune(classfication_tree, test_data):
    """
    剪枝待添加
    :param classfication_tree:
    :param test_data:
    :return:
    """
    pass
if __name__ == '__main__':
    data_set, labels = load_data_set()
    classification_tree = create_classfication_tree(data_set)
    print(classification_tree)



