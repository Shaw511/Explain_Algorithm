import torch
import numpy as np


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