# !/usr/local/bin/python

class HMM(object):
    def __init__(self):
        import os
        self.model_file = './data/hmm_model.pkl'
        # 初始状态分布
        self.state_list = ['B', 'M', 'E', 'S']
        self.load_para = False

    def try_load_model(self, trained):
        if trained:
            import pickle
            with open(self.model_file, 'rb') as f:
                self.A_dcit = pickle.load(f)
                self.B_dict = pickle.load(f)
                self.Pi_dict = pickle.load(f)
                self.load_para = True
        else:
            # 隐藏状态转移矩阵
            self.A_dcit = {}
            # 发射概率：隐藏状态->词语（观测）
            # 观测状态概率矩阵：隐藏qj生成观测状态vk的概率bj（k）
            self.B_dict = {}
            # 初始的隐藏状态概率分布
            self.Pi_dict = {}
            self.load_para = False

    def train(self, path):
        self.try_load_model(False)
        # 统计状态出现的次数，求p（o）
        Count_dict = {}

        def init_parameters():
            for state in self.state_list:
                self.A_dcit[state] = {s: 0.0 for s in self.state_list}
                self.B_dict[state] = {}
                self.Pi_dict[state] = 0.0

                Count_dict[state] = 0

        def make_label(text: str) -> list:
            out_text = []
            if len(text) == 1:
                out_text.append('S')
            else:
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']
            return out_text

        init_parameters()
        line_num = -1
        # 观察者集合
        words = set()
        with open(path, encoding='utf-8') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                word_list = [i for i in line if i != '']
                words = words | set(word_list)

                line_list = line.split()
                line_state = []
                for w in line_list:
                    line_state.extend(make_label(w))

                assert len(word_list) == len(line_state)

                for k, v in enumerate(line_state):
                    Count_dict[v] += 1
                    if k == 0:
                        self.Pi_dict[v] += 1
                    else:
                        self.A_dcit[line_state[k - 1]][v] += 1
                        self.B_dict[line_state[k]][word_list[k]] = self.B_dict[line_state[k]].get(word_list[k], 0) + 1.0

        self.Pi_dict = {k: v * 1.0 / line_num for k, v in self.Pi_dict.items()}
        self.A_dcit = {k: {k1: v1 / Count_dict[k] for k1, v1 in v.items()} for k, v in self.A_dcit.items()}
        self.B_dict = {k: {k1: (v1 + 1) / Count_dict[k] for k1, v1 in v.items()} for k, v in self.B_dict.items()}

        import pickle
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.A_dcit, f)
            pickle.dump(self.B_dict, f)
            pickle.dump(self.Pi_dict, f)

        return self

    def viterbi(self, text, states, start_p, trans_p, emit_p):
        '''

        :param text: 要切割的文本
        :param states: 隐藏状态list
        :param start_p: Pi_dict
        :param trans_p: A_dict 隐藏状态的转移概率（E B M这些状态）
        :param emit_p: 发射概率（标注 -> 词语）
       :return: prob, path[state]
        '''
        # 各个隐藏状态对应的最大probability
        v = [{}]
        path = {}
        for y in states:
            v[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]

        for t in range(1, len(text)):
            v.append({})
            new_path = {}

            never_seen = text[t] not in emit_p['S'].keys() and \
                         text[t] not in emit_p['M'].keys() and \
                         text[t] not in emit_p['E'].keys() and \
                         text[t] not in emit_p['B'].keys()
            for y in states:
                emitP = emit_p[y].get(text[t], 0) if not never_seen else 1.0
                (prob, state) = max([(v[t - 1][y0] * trans_p[y0].get(y, 0) * emitP, y0)
                                     for y0 in states if v[t - 1][y0] > 0])
                v[t][y] = prob
                new_path[y] = path[state] + [y]
            path = new_path

        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            (prob, state) = max([(v[len(text) - 1][y], y) for y in ('E', 'M')])
        else:
            (prob, state) = max([(v[len(text) - 1][y], y) for y in states])
        #     a = [(1,9),(2,6),(4,5)]
        #     b, c = max(a) 其中b = 4, c = 5, 说明max函数返回的是一个tuple

        return prob, path[state]

    def cut(self, text_cut):
        import os
        if not self.load_para:
            self.try_load_model(os.path.exists(self.model_file))

        (prob, pos_list) = self.viterbi(text_cut, self.state_list, self.Pi_dict, self.A_dcit, self.B_dict)

        begin, next_index = 0, 0
        for i, char in enumerate(text_cut):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text_cut[begin: i + 1]
                next_index = i + 1
            elif pos == 'S':
                yield char
                next_index = i + 1
        if next_index < len(text_cut):
            yield text_cut[next:]


if __name__ == "__main__":
    hmm = HMM()
    # hmm.train('data/trainCorpus.txt_utf8.txt')
    text = "这件事情他做的不对"
    # res = hmm.cut(text)
    # print(str(list(res)))
    f = hmm.cut(text)
    print(list(f))
