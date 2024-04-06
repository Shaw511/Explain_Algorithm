import networkx as nx
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
import configs
from scipy.special import softmax
# from pgmpy.estimators import ConstraintBasedEstimator  # 没用到，但是会报错
from globalBalancing_baseline import *
from pgmpy.estimators.CITests import chi_square
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from scipy import stats
from collections import deque


def shortest_path(adjacency_matrix, start_node, end_node):
    if start_node == end_node:
        return 0  # 起点和终点相同，距离为0

    n = len(adjacency_matrix)
    visited = [False] * n
    queue = deque([(start_node, 0)])

    while queue:
        current_node, distance = queue.popleft()
        visited[current_node] = True

        for neighbor, is_connected in enumerate(adjacency_matrix[current_node]):
            if is_connected and not visited[neighbor]:
                if neighbor == end_node:
                    return distance + 1  # 找到终点，返回距离
                queue.append((neighbor, distance + 1))

    return -1  # 无法到达终点


def get_ground_truth(node, args):
    gt = []
    if args.dataset == 'syn1':
        gt = get_ground_truth_syn1(node)  # correct
    elif args.dataset == 'syn2':
        gt = get_ground_truth_syn1(node)  # correct
    elif args.dataset == 'syn3':
        gt = get_ground_truth_syn3(node)  # correct
    elif args.dataset == 'syn4':
        gt = get_ground_truth_syn4(node)  # correct
    elif args.dataset == 'syn5':
        gt = get_ground_truth_syn5(node)  # correct
    elif args.dataset == 'syn6':
        gt = get_ground_truth_syn1(node)  # correct
    return gt


def get_ground_truth_syn1(node):
    base = [0, 1, 2, 3, 4]
    ground_truth = []
    offset = node % 5
    ground_truth = [node - offset + val for val in base]
    return ground_truth


def get_ground_truth_syn3(node):
    base = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    buff = node - 3
    ground_truth = []
    offset = buff % 9
    ground_truth = [buff - offset + val + 3 for val in base]
    return ground_truth


def get_ground_truth_syn4(node):
    buff = node - 1
    base = [0, 1, 2, 3, 4, 5]
    ground_truth = []
    offset = buff % 6
    ground_truth = [buff - offset + val + 1 for val in base]
    return ground_truth


def get_ground_truth_syn5(node):
    base = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    buff = node - 7
    ground_truth = []
    offset = buff % 9
    ground_truth = [buff - offset + val + 7 for val in base]
    return ground_truth


class Node_Explainer:
    def __init__(self, model, A, X, ori_pred, num_layers, mode=0, print_result=1):
        self.model = model
        self.model.eval()
        self.A = A
        self.X = X
        self.ori_pred = ori_pred
        self.num_layers = num_layers
        self.mode = mode
        self.print_result = print_result
        print("Explainer settings")
        print("\\ A dim: ", self.A.shape)
        print("\\ X dim: ", self.X.shape)
        print("\\ Number of layers: ", self.num_layers)
        print("\\ Perturbation mode: ", self.mode)
        print("\\ Print result: ", self.print_result)

    def n_hops_A(self, n_hops):
        # Compute the n-hops adjacency matrix
        # n-hops 邻接矩阵表示从一个节点出发，经过不超过 n 个跳跃（或者说 "hops"）可以到达的所有其他节点
        adj = torch.tensor(self.A, dtype=torch.float)
        hop_adj = power_adj = adj
        for i in range(n_hops - 1):
            power_adj = power_adj @ adj
            prev_hop_adj = hop_adj
            hop_adj = hop_adj + power_adj
            hop_adj = (hop_adj > 0).float()
        return hop_adj.numpy().astype(int)

    def extract_n_hops_neighbors(self, nA, node_idx):
        # Return the n-hops neighbors of a node
        node_nA_row = nA[node_idx]
        # nA中的第node_idx行已经包含了当前矩阵通过n_hop能到达的所有节点，下面的neighbors为n-hop邻居节点索引
        neighbors = np.nonzero(node_nA_row)[0]
        node_idx_new = sum(node_nA_row[:node_idx])  # 对目标节点进行重新编号
        # 这里sub_A的维度会发生变化，新的维度是根据n-hop邻居节点数量决定的
        sub_A = self.A[neighbors][:, neighbors]
        sub_X = self.X[neighbors]
        return node_idx_new, sub_A, sub_X, neighbors

    def perturb_features_on_node(self, feature_matrix, node_idx, random=0, mode=0):
        # return a random perturbed feature matrix
        # random = 0 for nothing, 1 for random.
        # mode = 0 for random 0-1, 1 for scaling with original feature
        # 这里传进来的node_idx不是待解释节点的idx，而是待解释节点的邻居节点的idx
        # mode 代表不同的扰动方式
        X_perturb = feature_matrix
        if mode == 0:
            if random == 0:
                perturb_array = X_perturb[node_idx]
            elif random == 1:
                perturb_array = np.random.randint(2, size=X_perturb[node_idx].shape[0])  # mode等于0的话表示扰动值是0或者1,这是原始的扰动方式
                # perturb_array = np.random.randint(1, size=X_perturb[node_idx].shape[0])  # mode等于0的话表示扰动值是0或者1，这是自己修改的扰动方式
                # perturb_array = np.zeros(X_perturb[node_idx].shape[0]) + 0.1
            X_perturb[node_idx] = perturb_array
        elif mode == 1:
            if random == 0:
                perturb_array = X_perturb[node_idx]
            elif random == 1:
                perturb_array = np.multiply(X_perturb[node_idx],
                                            np.random.uniform(low=0.0, high=2.0, size=X_perturb[node_idx].shape[0]))
            X_perturb[node_idx] = perturb_array
        return X_perturb

    def explain(self, node_idx, num_samples=100, top_node=None, p_threshold=0.05, pred_threshold=0.1):

        print("Explaining node: " + str(node_idx))
        nA = self.n_hops_A(self.num_layers)
        node_idx_new, sub_A, sub_X, neighbors = self.extract_n_hops_neighbors(nA, node_idx)

        if (node_idx not in neighbors):
            neighbors = np.append(neighbors, node_idx)

        X_torch = torch.tensor([self.X], dtype=torch.float)
        A_torch = torch.tensor([self.A], dtype=torch.float)

        pred_torch, _ = self.model.forward(X_torch, A_torch)
        # 将模型的原始预测转换为概率分布
        soft_pred = np.asarray([softmax(np.asarray(pred_torch[0][node_].data)) for node_ in range(self.X.shape[0])])
        pred_node = np.asarray(pred_torch[0][node_idx].data)  # 某个节点的预测结果
        label_node = np.argmax(pred_node)
        soft_pred_node = softmax(pred_node)

        Samples = []
        Pred_Samples = []
        # num_samples是对原图进行扰动的次数
        for iteration in range(num_samples):

            X_perturb = self.X.copy()
            sample = []
            # 通过循环的方式对待解释节点的所有n-hop邻居节点进行扰动
            for node in neighbors:
                seed = np.random.randint(2)  # 生成一个随机整数0或者1，表示对某个节点的特征是扰动或者不扰动
                if seed == 1:
                    latent = 1
                    X_perturb = self.perturb_features_on_node(X_perturb, node, random=seed)
                else:
                    latent = 0
                sample.append(latent)
            # 一次采样后得到的数据
            X_perturb_torch = torch.tensor([X_perturb], dtype=torch.float)
            pred_perturb_torch, _ = self.model.forward(X_perturb_torch, A_torch)
            # 将模型在干扰数据上的预测结果转换为概率分布
            soft_pred_perturb = np.asarray(
                [softmax(np.asarray(pred_perturb_torch[0][node_].data)) for node_ in range(self.X.shape[0])])

            sample_bool = []
            # 判断哪些邻居节点对于模型的预测结果有显著影响，如果差异大于pre_threshold，表示节点对于预测结果有显著影响，将对应的标志设为1，否则设为0
            for node in neighbors:
                if (soft_pred_perturb[node, np.argmax(soft_pred[node])] + pred_threshold) < np.max(soft_pred[node]):
                    sample_bool.append(1)
                else:
                    sample_bool.append(0)

            Samples.append(sample)
            Pred_Samples.append(sample_bool)

        Samples = np.asarray(Samples)  # 标记每一次扰动时，各个邻居节点的特征是否被扰动, Samples的维度大小为num_perturb_samples * num_nodes
        Pred_Samples = np.asarray(Pred_Samples)  # 标记每一次扰动时，邻居节点自身的预测值是否有明显差异
        Combine_Samples = Samples - Samples

        # 该代码将`Samples`数组中的每个元素乘以10，然后将`Pred_Samples`数组中对应元素的值加到乘以10的结果中。最后，对于每一行，它还在每个结果上加1。
        for s in range(Samples.shape[0]):
            Combine_Samples[s] = np.asarray(
                [Samples[s, i] * 10 + Pred_Samples[s, i] + 1 for i in range(Samples.shape[1])])

        data = pd.DataFrame(Combine_Samples)
        ind_sub_to_ori = dict(zip(list(data.columns), neighbors))
        data = data.rename(columns={0: "A", 1: "B"})  # Trick to use chi_square test on first two data columns
        ind_ori_to_sub = dict(zip(neighbors, list(data.columns)))

        p_values = []
        dependent_neighbors = []
        dependent_neighbors_p_values = []
        for node in neighbors:

            chi2, p = chi_square(ind_ori_to_sub[node], ind_ori_to_sub[node_idx], [], data)
            p_values.append(p)
            if p < p_threshold:
                dependent_neighbors.append(node)
                dependent_neighbors_p_values.append(p)

        pgm_stats = dict(zip(neighbors, p_values))

        pgm_nodes = []
        if top_node == None:
            pgm_nodes = dependent_neighbors
        else:
            top_p = np.min((top_node, len(neighbors) - 1))
            ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
            pgm_nodes = [ind_sub_to_ori[node] for node in ind_top_p]

        data = data.rename(columns={"A": 0, "B": 1})
        data = data.rename(columns=ind_sub_to_ori)
        return pgm_nodes, data, pgm_stats

    def explain_range(self, node_list, num_samples=1000, top_node=None, p_threshold=0.05, pred_threshold=0.1):
        print("当前的图数据为：", "\n", self.A, "\n", self.X)
        # 解释一段范围内节点的预测结果，首先找到每个节点通过n-hop能够到达的节点
        nA = self.n_hops_A(self.num_layers)
        neighbors_list = {}
        all_neighbors = []
        for node in node_list:
            _, _, _, neighbors = self.extract_n_hops_neighbors(nA, node)
            if (node not in neighbors):
                neighbors = np.append(neighbors, node)
            # 当前节点通过n-hop能够到达的邻居节点列表
            neighbors_list[node] = neighbors
            # 将all_neighbors中的元素和每一个node对应的neighbors进行合并得到新的数组, 这里是去过重的
            all_neighbors = list(set(all_neighbors) | set(np.append(neighbors, node)))
        N1 = self.A.shape[0]
        N2 = len(all_neighbors)
        print("当前的图中节点数量为：", N1, ", 待解释节点的n-hop邻居数量为：", N2)
        X_torch = torch.tensor([self.X], dtype=torch.float)
        A_torch = torch.tensor([self.A], dtype=torch.float)
        # 查看模型的原始预测值，并将原始预测转换为概率分布
        pred_torch, _ = self.model.forward(X_torch, A_torch)
        soft_pred = np.asarray([softmax(np.asarray(pred_torch[0][node_].data)) for node_ in range(self.X.shape[0])])

        # 新的数据生成方式加上重加权
        prog_args = configs.arg_parse()
        explanations = {}
        distance_list = []
        for target in node_list:
            print("*"*100)
            print(f"为当前节点 {target} 生成可解释节点")
            print("当前节点的n-hop邻居数量为：", len(neighbors_list[target]))

            # 尝试展示n-hop节点图
            print('尝试展示n-hop节点图：')
            # 创建一个有向图
            G = nx.DiGraph()
            # 添加节点
            nodes = neighbors_list.keys()
            G.add_nodes_from(nodes)
            # 添加边
            for node in nodes:
                neighbors = neighbors_list[node]
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)
            # 画图
            pos = nx.spring_layout(G)  # 定义布局
            nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_color='black',
                    edge_color='gray')
            # 在节点周围标注n-hop邻居数量
            for node in nodes:
                x, y = pos[node]
                neighbors_count = len(neighbors_list[node])
                plt.text(x, y, f'n-hop neighbors: {neighbors_count}', fontsize=8, ha='center', va='bottom')
            plt.show()





            # 首先对n-hop邻居进行扰动，生成每个待解释节点对应的数据
            nhops_sample = []
            target_sample = []
            for iteration in range(num_samples):
                X_perturb = self.X.copy()
                # 通过循环的方式对当前target的目标节点, p为对n-hop邻居进行扰动的概率
                p = 0.4
                n_sample = []
                target_neighbors = neighbors_list[target]
                for node in target_neighbors:
                    flag = np.random.binomial(1, p)
                    if flag == 1:
                        latent = 1
                        X_perturb = self.perturb_features_on_node(X_perturb, node, random=flag, mode=self.mode)
                    else:
                        latent = 0
                    n_sample.append(latent)
                # 一次扰动之后得到数据，下面是根据扰动数据得到的预测结果和对应的概率分布
                X_perturb_torch = torch.tensor([X_perturb], dtype=torch.float)
                pred_perturb_torch, _ = self.model.forward(X_perturb_torch, A_torch)
                soft_pred_perturb = np.asarray([softmax(np.asarray(pred_perturb_torch[0][node_].data)) for node_ in range(self.X.shape[0])])
                # 判断此时目标节点的预测结果是否会发生变化
                t_sample = []
                if (soft_pred_perturb[target, np.argmax(soft_pred[target])] + pred_threshold) < np.max(soft_pred[target]):  # 这里有个问题是能直接用target当索引吗
                    t_sample.append(1)
                else:
                    t_sample.append(0)
                nhops_sample.append(n_sample)
                target_sample.append(t_sample)
            # 格式转换
            nhops_sample = np.asarray(nhops_sample)
            target_sample = np.asarray(target_sample)
            total_sample = np.column_stack((nhops_sample, target_sample))
            data = pd.DataFrame(total_sample)
            # 为n-hop邻居中的节点进行编号映射
            target_neighbors = target_neighbors.tolist()
            target_neighbors.append("Y")
            ind_ori_to_sub = {}
            for index, value in enumerate(target_neighbors):
                ind_ori_to_sub[value] = index

            # 开始做可解释性分析
            print("Generating explanation for node: ", target)
            p_values = []
            dependent_neighbors = []
            dependent_neighbors_p_values = []
            for node in neighbors_list[target]:
                p = 0
                if node ==target:
                    p = 0
                    p_values.append(p)
                else:
#                     print(chi_square(ind_ori_to_sub[node], ind_ori_to_sub["Y"], [], data)) # False
                    chi2, p = chi_square(ind_ori_to_sub[node], ind_ori_to_sub["Y"], [], data) # 调用CITests.py中的卡方分布
                    p_values.append(p)
                if p < 0.05:
                    dependent_neighbors.append(node)
                    dependent_neighbors_p_values.append(p)
            pgm_nodes = []
            if top_node == None:
                pgm_nodes = dependent_neighbors
            else:
                ind_subnei_to_ori = dict(zip(range(len(neighbors_list[target])), neighbors_list[target]))
                if top_node < len(neighbors_list[target]):
                    ind_top = np.argpartition(p_values, top_node)[0:top_node]
                    pgm_nodes = [ind_subnei_to_ori[node] for node in ind_top]
                else:
                    pgm_nodes = neighbors_list[target]
                explanations[target] = pgm_nodes
                if self.print_result == 1:
                    print("Current pgm_nodes are: ", pgm_nodes)
                ground_truth_nodes = get_ground_truth(target, prog_args)
                print("Current ground_truth_nodes为：", ground_truth_nodes)
                # 遍历得到的可解释结果,查看它是当前节点的几跳邻居
                for node in pgm_nodes:
                    if node not in ground_truth_nodes:
                        distance = shortest_path(self.A, node, target)
                        if distance != -1:
                            print(f"节点 {node} 和节点 {target} 之间的最短跳数为 {distance}")
                            distance_list.append(distance)
                        else:
                            print(f"节点 {node} 和节点 {target} 之间没有直接连接")
        print("当前模型得到得错误节点都是：", distance_list, "邻居")
        return explanations

    def search_MK(self, data, target, nodes):
        target = str(int(target))
        data.columns = data.columns.astype(str)
        nodes = [str(int(node)) for node in nodes]

        MB = nodes
        while True:
            count = 0
            for node in nodes:
                evidences = MB.copy()
                evidences.remove(node)
                _, p = self.chi_square(target, node, evidences, data[nodes + [target]])
                if p > 0.05:
                    MB.remove(node)
                    count = 0
                else:
                    count = count + 1
                    if count == len(MB):
                        return MB

    def pgm_generate(self, target, data, pgm_stats, subnodes, child=None):

        subnodes = [str(int(node)) for node in subnodes]
        target = str(int(target))
        subnodes_no_target = [node for node in subnodes if node != target]
        data.columns = data.columns.astype(str)

        MK_blanket = self.search_MK(data, target, subnodes_no_target.copy())

        if child == None:
            est = HillClimbSearch(data[subnodes_no_target], scoring_method=BicScore(data))
            pgm_no_target = est.estimate()
            for node in MK_blanket:
                if node != target:
                    pgm_no_target.add_edge(node, target)

            #   Create the pgm
            pgm_explanation = BayesianModel()
            for node in pgm_no_target.nodes():
                pgm_explanation.add_node(node)
            for edge in pgm_no_target.edges():
                pgm_explanation.add_edge(edge[0], edge[1])

            #   Fit the pgm
            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self.generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self.generalize_others)
            pgm_explanation.fit(data_ex)
        else:
            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self.generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self.generalize_others)

            est = HillClimbSearch(data_ex, scoring_method=BicScore(data_ex))
            pgm_w_target_explanation = est.estimate()

            #   Create the pgm    
            pgm_explanation = BayesianModel()
            for node in pgm_w_target_explanation.nodes():
                pgm_explanation.add_node(node)
            for edge in pgm_w_target_explanation.edges():
                pgm_explanation.add_edge(edge[0], edge[1])

            #   Fit the pgm
            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self.generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self.generalize_others)
            pgm_explanation.fit(data_ex)

        return pgm_explanation

    def pgm_conditional_prob(self, target, pgm_explanation, evidence_list):
        pgm_infer = VariableElimination(pgm_explanation)
        for node in evidence_list:
            if node not in list(pgm_infer.variables):
                print("Not valid evidence list.")
                return None
        evidences = self.generate_evidence(evidence_list)
        elimination_order = [node for node in list(pgm_infer.variables) if node not in evidence_list]
        elimination_order = [node for node in elimination_order if node != target]
        q = pgm_infer.query([target], evidence=evidences,
                            elimination_order=elimination_order, show_progress=False)
        return q.values[0]

    def generalize_target(self, x):
        if x > 10:
            return x - 10
        else:
            return x

    def generalize_others(self, x):
        if x == 2:
            return 1
        elif x == 12:
            return 11
        else:
            return x

    def generate_evidence(self, evidence_list):
        return dict(zip(evidence_list, [1 for node in evidence_list]))

    def chi_square(self, X, Y, Z, data):
        """
        Modification of Chi-square conditional independence test from pgmpy
        Tests the null hypothesis that X is independent from Y given Zs.

        Parameters
        ----------
        X: int, string, hashable object
            A variable name contained in the data set
        Y: int, string, hashable object
            A variable name contained in the data set, different from X
        Zs: list of variable names
            A list of variable names contained in the data set, different from X and Y.
            This is the separating set that (potentially) makes X and Y independent.
            Default: []
        Returns
        -------
        chi2: float
            The chi2 test statistic.
        p_value: float
            The p_value, i.e. the probability of observing the computed chi2
            statistic (or an even higher value), given the null hypothesis
            that X _|_ Y | Zs.
        sufficient_data: bool
            A flag that indicates if the sample size is considered sufficient.
            As in [4], require at least 5 samples per parameter (on average).
            That is, the size of the data set must be greater than
            `5 * (c(X) - 1) * (c(Y) - 1) * prod([c(Z) for Z in Zs])`
            (c() denotes the variable cardinality).
        References
        ----------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.2.2.3 (page 789)
        [2] Neapolitan, Learning Bayesian Networks, Section 10.3 (page 600ff)
            http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        [3] Chi-square test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Test_of_independence
        [4] Tsamardinos et al., The max-min hill-climbing BN structure learning algorithm, 2005, Section 4
        """
        X = str(int(X))
        Y = str(int(Y))
        if isinstance(Z, (frozenset, list, set, tuple)):
            Z = list(Z)
        Z = [str(int(z)) for z in Z]

        state_names = {
            var_name: data.loc[:, var_name].unique() for var_name in data.columns
        }

        row_index = state_names[X]
        column_index = pd.MultiIndex.from_product(
            [state_names[Y]] + [state_names[z] for z in Z], names=[Y] + Z
        )

        XYZ_state_counts = pd.crosstab(
            index=data[X], columns=[data[Y]] + [data[z] for z in Z],
            rownames=[X], colnames=[Y] + Z
        )

        if not isinstance(XYZ_state_counts.columns, pd.MultiIndex):
            XYZ_state_counts.columns = pd.MultiIndex.from_arrays([XYZ_state_counts.columns])
        XYZ_state_counts = XYZ_state_counts.reindex(
            index=row_index, columns=column_index
        ).fillna(0)

        if Z:
            XZ_state_counts = XYZ_state_counts.sum(axis=1, level=list(range(1, len(Z) + 1)))  # marginalize out Y
            YZ_state_counts = XYZ_state_counts.sum().unstack(Z)  # marginalize out X
        else:
            XZ_state_counts = XYZ_state_counts.sum(axis=1)
            YZ_state_counts = XYZ_state_counts.sum()
        Z_state_counts = YZ_state_counts.sum()  # marginalize out both

        XYZ_expected = np.zeros(XYZ_state_counts.shape)

        r_index = 0
        for X_val in XYZ_state_counts.index:
            X_val_array = []
            if Z:
                for Y_val in XYZ_state_counts.columns.levels[0]:
                    temp = XZ_state_counts.loc[X_val] * YZ_state_counts.loc[Y_val] / Z_state_counts
                    X_val_array = X_val_array + list(temp.to_numpy())
                XYZ_expected[r_index] = np.asarray(X_val_array)
                r_index = +1
            else:
                for Y_val in XYZ_state_counts.columns:
                    temp = XZ_state_counts.loc[X_val] * YZ_state_counts.loc[Y_val] / Z_state_counts
                    X_val_array = X_val_array + [temp]
                XYZ_expected[r_index] = np.asarray(X_val_array)
                r_index = +1

        observed = XYZ_state_counts.to_numpy().reshape(1, -1)
        expected = XYZ_expected.reshape(1, -1)
        observed, expected = zip(*((o, e) for o, e in zip(observed[0], expected[0]) if not (e == 0 or math.isnan(e))))
        chi2, significance_level = stats.chisquare(observed, expected)

        return chi2, significance_level
