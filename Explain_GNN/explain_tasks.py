import utils
import models
import numpy as np
import torch
import pgm_explainer as pe


def task_syn(args):
    # 加载输入图数据集
    A, X = utils.load_XA(args.dataset, datadir="../Generate_XA_Data/XAL") # 邻接矩阵，特征矩阵
    L = utils.load_labels(args.dataset, datadir="../Generate_XA_Data/XAL") # 标签
    num_classes = max(L) + 1    # 类别数
    input_dim = X.shape[1]      # 输入特征维度 (特征矩阵列数)
    print("input dim: ", input_dim, "; num classes: ", num_classes)

    ckpt = utils.load_ckpt(args) # 加载模型checkpoint文件
    # 初始化gcn编码节点模型
    model = models.GcnEncoderNode(input_dim=input_dim, hidden_dim=args.hidden_dim, embedding_dim=args.output_dim,
                                  label_dim=num_classes, num_layers=args.num_gc_layers, bn=args.bn, args=args,)
    model.load_state_dict(ckpt["model_state"]) # torch方法 state_dict=模型参数字典 本句即加载checkpoint文件的模型参数
    pred = ckpt["save_data"]["pred"] # 获取checkpoint模型预测结果

    # 解释器 输入：模型，节点邻接矩阵，节点特征矩阵，模型预测结果，GcnEncoderNode模型中的图卷积层数
    explainer = pe.Node_Explainer(model, A, X, pred, args.num_gc_layers)

    explanations = {}
    if args.explain_node == None: # 需要解释整个数据集一定范围内节点
#         '''
#         pgm_explainer.py explain_range 方法输入参数：
#         【num_samples】 对原始图进行扰动的次数，即生成解释的样本数。
#         【top_node】    要返回的可解释节点的数量，=None，则返回所有可解释节点；否则，返回根据p-value从低到高排序的前top_node个可解释节点
#         【pred_threshold】用于判断邻居节点对于预测结果影响是否显著的阈值。如果某个邻居节点的预测结果差异大于pred_threshold，则认为该节点对预测结果有显著影响。默认值为0.1。
#         【p_threshold】 用于判断两个节点之间是否存在条件独立关系的p值阈值。如果两个节点之间的卡方检验的p值小于p_threshold，则认为两个节点之间存在条件独立关系。默认值为0.05。
#         '''
        if args.dataset == 'syn1':
            explanations = explainer.explain_range(list(range(300, 350)), num_samples=args.num_perturb_samples, top_node=args.top_node,
                                                   pred_threshold=0.1)  # 原始范围是300-700
        elif args.dataset == 'syn2':
            explanations = explainer.explain_range(list(range(300, 350)), num_samples=args.num_perturb_samples, top_node=args.top_node,
                                                   pred_threshold=0.1)  # 原始范围是300-700 + list(range(1000, 1400)
        elif args.dataset == 'syn3':
            explanations = explainer.explain_range(list(range(300, 350)), num_samples=args.num_perturb_samples, top_node=args.top_node,  # 原始范围是300-1020
                                                   pred_threshold=0.05)  # pred_threshold默认值是0.05
        elif args.dataset == 'syn4':
            explanations = explainer.explain_range(list(range(511, 561)), num_samples=args.num_perturb_samples, top_node=args.top_node,  # 原始范围是511-871
                                                   pred_threshold=0.1)
        elif args.dataset == 'syn5':
            explanations = explainer.explain_range(list(range(511, 561)), num_samples=args.num_perturb_samples, top_node=args.top_node, pred_threshold=0.05)  # 原始范围是511-1231
        elif args.dataset == 'syn6':
            explanations = explainer.explain_range(list(range(300, 350)), num_samples=args.num_perturb_samples,
                                                   top_node=args.top_node) # 原始范围是300-700
    else:   # 解释数据集中指定节点
#     '''pgm_explainer.py explain 方法：
#     【num_samples】 对原始图进行扰动的次数，即生成解释的样本数。
#     【top_node】    要返回的可解释节点的数量，=None，则返回所有可解释节点；否则，返回根据p-value从低到高排序的前top_node个可解释节点
#     '''
        explanation = explainer.explain(args.explain_node, num_samples=args.num_perturb_samples, top_node=args.top_node)
        print("Explaination for current node is :", explanation)
        explanations[args.explain_node] = explanation
    print("Explainations for current node set are :", explanations)

    # explanations保存
    savename = utils.gen_filesave(args)
    np.save(savename, explanations) # PGM_Node/Explain_GNN/result/syn2/explanations_syn2_top_5_800_samples.npy


def bitcoin(args):
    A, X = utils.load_XA(args.dataset, datadir="../Generate_XA_Data/XAL")
    L = utils.load_labels(args.dataset, datadir="../Generate_XA_Data/XAL")
    num_classes = max(L) + 1
    input_dim = X.shape[1]
    num_nodes = X.shape[0]
    ckpt = utils.load_ckpt(args)
    print("input dim: ", input_dim, "; num classes: ", num_classes)

    model = models.GcnEncoderNode(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.output_dim,
        label_dim=num_classes,
        num_layers=args.num_gc_layers,
        bn=args.bn,
        args=args,
    )

    model.load_state_dict(ckpt["model_state"])
    pred = ckpt["save_data"]["pred"]
    explainer = pe.Node_Explainer(model, A, X, pred, 1)
    node_to_explain = [i for [i] in np.argwhere(np.sum(A, axis=0) > 2)]
    explanations = explainer.explain_range(node_to_explain, num_samples=args.num_perturb_samples,
                                           top_node=args.top_node)

    print(explanations)
    savename = utils.gen_filesave(args)
    np.save(savename, explanations)
