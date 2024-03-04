import networkx as nx
import math
import torch
import configs_server
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
# from pgmpy.estimators import ConstraintBasedEstimator  # 没用到，但是会报错
from globalBalancing_baseline_server import *
from pgmpy.estimators.CITests import chi_square
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from scipy import stats

# 100个节点卡方测试选五个=top-5/top-10/改成学100个节点的因果结构/MB学习/MHC/PC 学习父子节点--曹大元
# ---和
# 关联关系 对基础的卡方测试来说 改变 应用 不同 因果结构学习方法
# 数据集不变 合成数据集 挑几个
# 可解释节点
# mhc文件 行是样本 列是节点 学列指间的关系
# 翻译两篇 翻译多少字 几千字 稳定学习论文
# ground-truth 看论文github地址 例子


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
    def __init__(self, model, A, X, ori_pred, num_layers, mode=0, print_result=1 ):
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
                perturb_array = np.multiply(X_perturb[node_idx], np.random.uniform(low=0.0, high=2.0, size=X_perturb[node_idx].shape[0]))
            X_perturb[node_idx] = perturb_array
        return X_perturb

    def explain(self, node_idx, num_samples=100, top_node=None, p_threshold=0.05, pred_threshold=0.1):
        # 解释单个节点和解释多个节点的代码是不一样的
        # top-node是选择的前M个相关性比较大的节点
        print("Explaining node: " + str(node_idx))
        nA = self.n_hops_A(self.num_layers)
        node_idx_new, sub_A, sub_X, neighbors = self.extract_n_hops_neighbors(nA, node_idx)

        if (node_idx not in neighbors):
            neighbors = np.append(neighbors, node_idx)

        X_torch = torch.tensor([self.X], dtype=torch.float)
        A_torch = torch.tensor([self.A], dtype=torch.float)

        # 这里有一个问题，explainer没有读取事先保存下来的预测结果，而是把A和X输入到模型中得到了一个未经反向传播的预测结果
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
            soft_pred_perturb = np.asarray([softmax(np.asarray(pred_perturb_torch[0][node_].data)) for node_ in range(self.X.shape[0])])

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
            Combine_Samples[s] = np.asarray([Samples[s, i] * 10 + Pred_Samples[s, i] + 1 for i in range(Samples.shape[1])])

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

        pred_torch, _ = self.model.forward(X_torch, A_torch)
        # 将模型的原始预测转换为概率分布, soft_pred的每一行数值加起来等于1，其中有一个概率最大的项，例如 [0.03438661 0.00180207 0.08030216 0.88350904]
        soft_pred = np.asarray([softmax(np.asarray(pred_torch[0][node_].data)) for node_ in range(self.X.shape[0])])

        # 新的数据生成方式加上重加权
        prog_args = configs_server.arg_parse()
        explanations = {}
        for target in node_list:
            print("Generating explanation for node: ", target)
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
            # print("为当前目标节点生成的数据大小是：\n", nhops_sample.shape, target_sample.shape)
            # 为n-hop邻居中的节点进行编号映射
            target_list = [i for i in range(len(target_neighbors))]  # target_list中的元素是从0开始的, 比如[0, 283]
            ind_ori_to_sub = dict(zip(target_neighbors, target_list))
            ind_sub_to_ori = dict(zip(target_list, target_neighbors))
            # print("当前节点" + str(target) + "的邻居节点有：", target_neighbors)
            # print("ind_sub_to_ori是：", ind_sub_to_ori)

            # 开始利用重加权方法为待解释的节点生成可解释结果
            Xol = 1e-8
            RMSE_gb, F1_gb, W = f_baseline_globalBalancing(1, X_in, Y_in, learning_rate, num_steps, tol)
            # 寻找因果效应值为top-k的节点并转换索引
            sorted_W = np.argsort(np.squeeze(W))[::-1]
            top_k_indices = sorted_W[:prog_args.top_node]
            explaination_nodes = [ind_sub_to_ori[item] for item in top_k_indices]
            p_in = nhops_sample
            Y_in = target_sample
            learning_rate = 0.01
            num_steps = 3000
            trint("当前节点的explaination_nodes为：", explaination_nodes)
            ground_truth_nodes = get_ground_truth(target, prog_args)
            print("当前节点的ground_truth_nodes为：", ground_truth_nodes)
            explanations[target] = explaination_nodes


        # Samples = []
        # Pred_Samples = []
        #
        # for iteration in range(num_samples):
        #
        #     X_perturb = self.X.copy()
        #     sample = []
        #     # 通过循环的方式对待解释节点的所有n-hop邻居节点进行扰动, 但是all_neighbors会包含图中的所有节点，所以最后得到的结果是对所有的节点都进行一次可能的扰动操作
        #     for node in all_neighbors:
        #         seed = np.random.randint(2)  # 生成一个随机整数0或者1，表示对某个节点的特征是扰动或者不扰动
        #         if seed == 1:
        #             latent = 1
        #             X_perturb = self.perturb_features_on_node(X_perturb, node, random=seed, mode=self.mode)
        #         else:
        #             latent = 0
        #         sample.append(latent)
        #     # 一次扰动之后得到数据，下面是根据扰动数据得到的预测结果和对应的概率分布
        #     X_perturb_torch = torch.tensor([X_perturb], dtype=torch.float)
        #     pred_perturb_torch, _ = self.model.forward(X_perturb_torch, A_torch)
        #     soft_pred_perturb = np.asarray([softmax(np.asarray(pred_perturb_torch[0][node_].data)) for node_ in range(self.X.shape[0])])
        #
        #     sample_bool = []
        #     for node in all_neighbors:
        #         # 判断哪些邻居节点对于模型的预测结果有显著影响，如果差异大于pre_threshold，表示节点对于预测结果有显著影响，将对应的标志设为1，否则设为0
        #         if (soft_pred_perturb[node, np.argmax(soft_pred[node])] + pred_threshold) < np.max(soft_pred[node]):
        #             sample_bool.append(1)
        #         else:
        #             sample_bool.append(0)
        #
        #     Samples.append(sample)
        #     Pred_Samples.append(sample_bool)
        #
        # # Samples的维度大小为num_perturb_samples * num_all_neighbors
        # Samples = np.asarray(Samples)
        # Pred_Samples = np.asarray(Pred_Samples)
        # # Combine_Samples = Samples-Samples
        # Combine_Samples = np.zeros_like(Samples, dtype=float)
        # for s in range(Samples.shape[0]):
        #     # Combine_Samples[s] = np.asarray([Samples[s, i] * 10 + Pred_Samples[s, i] + 1 for i in range(Samples.shape[1])])
        #     # Combine_Samples[s] = np.asarray([1 if Samples[s, i] * 0.75 + Pred_Samples[s, i] * 0.25 > 0.5 else 0 for i in range(Samples.shape[1])])
        #     Combine_Samples[s] = np.asarray([Samples[s, i] and Pred_Samples[s, i] for i in range(Samples.shape[1])])  # 自己改的数据生成方式
        # data = pd.DataFrame(Combine_Samples)
        # data = data.rename(columns={0: "A", 1: "B"})  # Trick to use chi_square test on first two data columns
        # # 将原始节点的索引映射到扰动后的子节点的列索引, all_neighbors中的节点编号是按照原始的图节点编号来的，而data.columns中的节点编号是按照新的顺序编号
        # ind_ori_to_sub = dict(zip(all_neighbors, list(data.columns)))
        # ind_sub_to_ori = dict(zip(list(data.columns), all_neighbors))  # 子数据集的每一列对应的是neighbors中的哪个邻居节点

        '''
        # 原始的PGM算法
        explanations = {}
        for target in node_list:
            print("Generating explanation for node: ", target)
            print("当前节点的n-hop邻居数量为：", len(neighbors_list[target]))
            p_values = []
            dependent_neighbors = []
            dependent_neighbors_p_values = []
            for node in neighbors_list[target]:
                p = 0
                if node == target:
                    p = 0
                    p_values.append(p)
                else:
                    # 卡方检验，用于确定两个离散随机变量之间是否存在条件独立性关系。
                    chi2, p = chi_square(ind_ori_to_sub[node], ind_ori_to_sub[target], [], data)
                    p_values.append(p)
                if p < 0.05:
                    dependent_neighbors.append(node)
                    dependent_neighbors_p_values.append(p)
            # 如果有无孩约束的话就不需要去找MB节点集合了，因为肯定只有父节点
            pgm_nodes = []
            if top_node == None:
                pgm_nodes = dependent_neighbors
            else:
                ind_subnei_to_ori = dict(zip(range(len(neighbors_list[target])), neighbors_list[target]))
                print(ind_subnei_to_ori)
                if top_node < len(neighbors_list[target]):
                    # 查找一个NumPy数组（或类似的数据结构）p_values中的最小的top_node个元素的索引，并将这些索引存储在名为ind_top的新数组中
                    ind_top = np.argpartition(p_values, top_node)[0:top_node]
                    pgm_nodes = [ind_subnei_to_ori[node] for node in ind_top]
                else:
                    pgm_nodes = neighbors_list[target]

            explanations[target] = pgm_nodes
            if self.print_result == 1:
                print("Current pgm_nodes are: ", pgm_nodes)
        '''

        '''
        # 绘制ground-truth的结构图
        prog_args = configs_server.arg_parse()
        count = 0
        for target in node_list:
            print("dataset is:", prog_args.dataset)
            print("Generating explanation for node: ", target)
            # 输出目标节点n-hop邻居图并保存到结果中
            A = self.A
            G = nx.Graph()
            # 遍历邻接矩阵的每一个元素，将非零元素对应的边添加到图中
            for i in range(A.shape[0]):
                for j in range(i + 1, A.shape[0]):  # 因为是无向图，只添加上三角部分
                    if A[i, j] == 1:
                        G.add_edge(i, j)
            # 找到指定节点的3-hop邻居
            target_neighbors = neighbors_list[target]
            print("node " + str(target) + "的n-hop邻居数量为：", len(target_neighbors))
            three_hop_neighbors = set(nx.single_source_shortest_path_length(G, target, cutoff=3).keys())
            print(len(three_hop_neighbors))
            count += len(three_hop_neighbors)
            # 绘制子图，包括指定节点和其3-hop邻居
            subgraph = G.subgraph(three_hop_neighbors)
            # 绘制图形
            pos = nx.spring_layout(subgraph)  # 为子图指定布局
            nx.draw(subgraph, pos, with_labels=True, node_color='b', node_size=500, font_size=10)
            fig_name = "/home/user/文档/wyj/PGMExplainer-master/PGM_Node/Explain_GNN/result/" + str(
                prog_args.dataset) + "/nhops_graph/" + str(target) + "_subgraph.png"
            plt.savefig(fig_name)
            # 清除当前绘图以准备下一个节点
            plt.clf()
        print(count / 50)
        '''

        '''
        explanations = {}
        prog_args = configs_server.arg_parse()
        for target in node_list:
            print("Generating explanation for node: ", target)
            # 首先找到目标节点在子数据中对应的位置, 并提取数据Y_in
            target_sub = ind_ori_to_sub[target]
            Y_in = data[target_sub]
            # 然后找到目标节点的邻居节点在子数据中对应的位置，并提取数据X_in
            target_neighbors = neighbors_list[target]
            print("node " + str(target) + "的n-hop邻居数量为：", len(target_neighbors))
            target_neighbors_sub = []
            for tn in target_neighbors:
                target_neighbors_sub.append(ind_ori_to_sub[tn])
            X_in = data[target_neighbors_sub]

            # 设置映射
            col_to_sub = {}
            for i in range(X_in.shape[1]):
                col_to_sub[i+1] = X_in.columns[i]
            # 转换数据，设定重加权参数
            X_in = np.array(X_in)
            Y_in = np.array(Y_in).reshape(-1, 1)
            learning_rate = 0.01
            num_steps = 3000
            tol = 1e-8
            RMSE_gb, F1_gb, W = f_baseline_globalBalancing(1, X_in, Y_in, learning_rate, num_steps, tol)
            # 寻找因果效应值为top-k的节点并转换索引
            sorted_W = np.argsort(np.squeeze(W))[::-1]
            top_k_indices = sorted_W[:prog_args.top_node] + 1
            pgm_nodes = []
            for item in top_k_indices:
                temp = col_to_sub[item]
                node = ind_sub_to_ori[temp]
                pgm_nodes.append(node)
            print("当前节点的pgm_nodes为：", pgm_nodes)
            ground_truth = get_ground_truth(target, prog_args)
            print("当前节点的ground_truth_nodes为：", ground_truth)
            explanations[target] = pgm_nodes
        '''

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
