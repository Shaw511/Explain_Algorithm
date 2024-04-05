import models
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
import time
import sklearn.metrics as metrics
import scipy.sparse


def evaluate_node(ypred, labels, train_idx, test_idx):
    _, pred_labels = torch.max(ypred, 2)  # 得到每一行
    pred_labels = pred_labels.numpy()

    pred_train = np.ravel(pred_labels[:, train_idx])  # 将多维数组转换为1维数组
    pred_test = np.ravel(pred_labels[:, test_idx])
    labels_train = np.ravel(labels[:, train_idx])
    labels_test = np.ravel(labels[:, test_idx])

    result_train = {
        "prec": metrics.precision_score(labels_train, pred_train, average="macro"),  # 精确度是指模型在预测为正例的样本中，实际上确实为正例的比例。
        "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
        "acc": metrics.accuracy_score(labels_train, pred_train),  # 分类准确度
        "conf_mat": metrics.confusion_matrix(labels_train, pred_train),  # 混淆矩阵的行表示实际标签的类别，列表示模型预测的类别。矩阵中的每个元素表示实际标签属于某个类别并且模型预测为该类别的样本数量。混淆矩阵如下所示：
    }
    result_test = {
        "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
        "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
        "acc": metrics.accuracy_score(labels_test, pred_test),
        "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
    }
    return result_train, result_test


def train(model, A, X, L, args, normalize_adjacency=False):
    np.set_printoptions(threshold=np.inf)
    A= A.reshape(1,-1)[0][0]
    # <700x700 sparse matrix of type '<class 'numpy.int32'>' with 4110 stored elements in Compressed Sparse Row format>
    print('type of A:',type(A)) # <class 'scipy.sparse._csr.csr_matrix'>
    non_zero_elements = A.nnz
    print('non_zero_elements:',non_zero_elements)# non_zero_elements: 4110
    # for row in A:
    #     print(row.toarray())
    # print(A)
    # 统计稀疏矩阵中非零元素的个数
    num_nodes = A.shape[0]  # A是稀疏矩阵 这里输出的A.shape[0]即为节点的总数
    print('num_nodes:', num_nodes)
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    if normalize_adjacency == True:
        A_ = normalize_A(A)
    else:
        A_ = A

    # add batch dim，对维度进行扩展
    A_ = np.expand_dims(A_, axis=0)
    X_ = np.expand_dims(X, axis=0)
    L_ = np.expand_dims(L, axis=0)

    labels_train = torch.tensor(L_[:, train_idx], dtype=torch.long)
    #将A_从numpy数组转换为float64
    # A_ = A_.toarray()
    print(A_)

    adj = torch.tensor(A_, dtype=torch.float32)
    x = torch.tensor(X_, requires_grad=True, dtype=torch.float)
    # scheduler是用来调整优化器的学习率的
    scheduler, optimizer = utils.build_optimizer(args, model.parameters(), weight_decay=args.weight_decay)
    model.train()

    ypred = None
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()
        ypred, adj_att = model(x, adj)
        # 因为前面对数据的维度进行了扩充，所以下面找训练数据的预测标签时有3个维度
        ypred_train = ypred[:, train_idx, :]
        loss = model.loss(ypred_train, labels_train)
        loss.backward()
        # PyTorch的 torch.nn.utils.clip_grad_norm 函数的使用方式发生了变化，现在使用torch.nn.utils.clip_grad_norm_ 函数来代替
        # 对梯度的值进行修剪，防止梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        elapsed = time.time() - begin_time
        result_train, result_test = evaluate_node(ypred.cpu(), L_, train_idx, test_idx)
        if epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; train_acc: ",
                result_train["acc"],
                "; test_acc: ",
                result_test["acc"],
                "; train_prec: ",
                result_train["prec"],
                "; test_prec: ",
                result_test["prec"],
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )

        if scheduler is not None:
            scheduler.step()

    print("训练集上的混淆矩阵为：\n", result_train["conf_mat"])
    print("测试集上的混淆矩阵为：\n", result_test["conf_mat"])

    model.eval()
    ypred, _ = model(x, adj)

    save_data = {
        "adj": A_,
        "feat": X_,
        "label": L_,
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": train_idx,
    }
    # ypred pred即为预测结果 研究一下怎么输出这个结果


    utils.save_checkpoint(model, optimizer, args, num_epochs=-1, save_data=save_data)


# import models
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import utils
# import time
# import sklearn.metrics as metrics
#
#
# def evaluate_node(ypred, labels, train_idx, test_idx):
#     _, pred_labels = torch.max(ypred, 2)
#     pred_labels = pred_labels.numpy()
#
#     pred_train = np.ravel(pred_labels[:, train_idx])
#     pred_test = np.ravel(pred_labels[:, test_idx])
#     labels_train = np.ravel(labels[:, train_idx])
#     labels_test = np.ravel(labels[:, test_idx])
#
#     result_train = {
#         "prec": metrics.precision_score(labels_train, pred_train, average="macro"),
#         "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
#         "acc": metrics.accuracy_score(labels_train, pred_train),
#         "conf_mat": metrics.confusion_matrix(labels_train, pred_train),
#     }
#     result_test = {
#         "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
#         "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
#         "acc": metrics.accuracy_score(labels_test, pred_test),
#         "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
#     }
#     return result_train, result_test
#
#
# def train(model, A, X, L, args, normalize_adjacency=False):
#     num_nodes = A.shape[0]
#     num_train = int(num_nodes * args.train_ratio)
#     idx = [i for i in range(num_nodes)]
#
#     np.random.shuffle(idx)
#     train_idx = idx[:num_train]
#     test_idx = idx[num_train:]
#
#     if normalize_adjacency == True:
#         A_ = normalize_A(A)
#     else:
#         A_ = A
#
#     # add batch dim
#     A_ = np.expand_dims(A_, axis=0)
#     X_ = np.expand_dims(X, axis=0)
#     L_ = np.expand_dims(L, axis=0)
#
#     labels_train = torch.tensor(L_[:, train_idx], dtype=torch.long)
#     adj = torch.tensor(A_, dtype=torch.float)
#     x = torch.tensor(X_, requires_grad=True, dtype=torch.float)
#     scheduler, optimizer = utils.build_optimizer(
#         args, model.parameters(), weight_decay=args.weight_decay
#     )
#     model.train()
#
#     ypred = None
#     for epoch in range(args.num_epochs):
#         begin_time = time.time()
#         model.zero_grad()
#         ypred, adj_att = model(x, adj)
#         ypred_train = ypred[:, train_idx, :]
#         loss = model.loss(ypred_train, labels_train)
#         loss.backward()
#         nn.utils.clip_grad_norm(model.parameters(), args.clip)
#         optimizer.step()
#         elapsed = time.time() - begin_time
#         result_train, result_test = evaluate_node(
#             ypred.cpu(), L_, train_idx, test_idx
#         )
#         if epoch % 10 == 0:
#             print(
#                 "epoch: ",
#                 epoch,
#                 "; loss: ",
#                 loss.item(),
#                 "; train_acc: ",
#                 result_train["acc"],
#                 "; test_acc: ",
#                 result_test["acc"],
#                 "; train_prec: ",
#                 result_train["prec"],
#                 "; test_prec: ",
#                 result_test["prec"],
#                 "; epoch time: ",
#                 "{0:0.2f}".format(elapsed),
#             )
#
#         if scheduler is not None:
#             scheduler.step()
#
#     print(result_train["conf_mat"])
#     print(result_test["conf_mat"])
#
#     model.eval()
#     ypred, _ = model(x, adj)
#
#     save_data = {
#         "adj": A_,
#         "feat": X_,
#         "label": L_,
#         "pred": ypred.cpu().detach().numpy(),
#         "train_idx": train_idx,
#     }
#
#     utils.save_checkpoint(model, optimizer, args, num_epochs=-1, save_data=save_data)




