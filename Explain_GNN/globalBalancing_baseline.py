import tensorflow as tf
import numpy as np
import sklearn.metrics as skm
import os


def f_baseline_globalBalancing(train_or_test, X_in, Y_in, learning_rate, num_steps, tol):
    n, p = X_in.shape
    display_step = 200

    X = tf.placeholder("float", [None, p])  # None表示样本数量可以是任意大小，p表示协变量维度的大小为p
    Y = tf.placeholder("float", [None, 1])
    G = tf.Variable(tf.ones([n, 1]))  # 全局权重

    # prediction
    W = tf.Variable(tf.random_normal([p, 1]))
    b = tf.Variable(tf.random_normal([1]))
    hypothesis_all = tf.nn.sigmoid(tf.matmul(X, W) + b)  # hypothesis_all的维度是n行1列
    hypothesis = tf.slice(hypothesis_all, [0, 0], [n, 1])  # [0, 0]表示每个维度从哪切割， [n, 1]表示切割的大小

    saver = tf.train.Saver([W] + [b])  # 保存W和b这两个关键的参数
    sess = tf.Session()

    if train_or_test == 1:
        loss_balancing = tf.constant(0, tf.float32)
        for j in range(1, p + 1):
            X_j = tf.slice(X, [j * n, 0], [n, p])
            I = tf.slice(X, [0, j - 1], [n, 1])
            balancing_j = tf.divide(tf.matmul(tf.transpose(X_j), G * G * I),tf.maximum(tf.reduce_sum(G * G * I), tf.constant(0.1))) - tf.divide(
                tf.matmul(tf.transpose(X_j), G * G * (1 - I)),
                tf.maximum(tf.reduce_sum(G * G * (1 - I)), tf.constant(0.1)))
            loss_balancing += tf.norm(balancing_j, ord=2)
        loss_regulizer = (tf.reduce_sum(G * G) - n) ** 2
        loss_regulizer_l2 = (tf.reduce_sum(G * G)) ** 2
        loss_l2reg = tf.reduce_sum(tf.abs(W))
        loss_predictive = -tf.reduce_sum(tf.divide(G * G * (Y * tf.log(tf.clip_by_value(hypothesis, 1e-8, 1)) + (1 - Y) * tf.log(tf.clip_by_value(1 - hypothesis, 1e-8, 1))), tf.reduce_sum(G * G)))
        loss = 1 * loss_predictive + 10000.0 / p * loss_balancing + 0.001 / n * loss_regulizer + 0.01 * loss_l2reg
        # loss = 10 * loss_predictive + 10 * loss_balancing + 0.01 * loss_regulizer + 0.01 * loss_l2reg
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())

        X_feed = X_in
        for j in range(p):
            X_j = np.copy(X_in)
            X_j[:, j] = 0
            X_feed = np.vstack((X_feed, X_j))

        l_pre = 0
        for i in range(1, num_steps + 1):
            _, l, l_predictive, l_balancing, l_regulizer, l_regulizer_l2 = sess.run([optimizer, loss, loss_predictive, loss_balancing, loss_regulizer, loss_regulizer_l2], feed_dict={X: X_feed, Y: Y_in})
            # if abs(l - l_pre) <= tol:
            #     print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f ... %f ... %f' % (
            #         i, l, l_predictive, l_balancing, l_regulizer, l_regulizer_l2))
            #     break
            l_pre = l
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f ... %f ... %f ... %f ... %f' % (
                    i, l, l_predictive, l_balancing, l_regulizer, l_regulizer_l2))
                '''
                W_final = sess.run(G)
                fw = open('bl_weight_from_tf_'+str(i)+'.txt', 'wb')
                for items in W_final:
                    fw.write(str(items[0])+'\r\n')
                fw.close()
                '''
        if not os.path.isdir('models/baseline_globalBalancing/'):
            os.makedirs('models/baseline_globalBalancing/')
        saver.save(sess, 'models/baseline_globalBalancing/baseline_globalBalancing.ckpt')
        '''
        RMSE = tf.sqrt(tf.reduce_mean((Y-hypothesis_all)**2))
        RMSE_error, Y_predict = sess.run([RMSE,hypothesis_all], feed_dict={X: X_in, Y:Y_in})
        F1_score = skm.f1_score(Y_in, Y_predict>0.5)
        return  RMSE_error, F1_score
        '''
    else:
        # saver = tf.train.import_meta_graph('models/baseline_globalBalancing/baseline_globalBalancing.ckpt.meta')
        saver.restore(sess, 'models/baseline_globalBalancing/baseline_globalBalancing.ckpt')

    hypothesis_p = tf.nn.sigmoid(tf.matmul(X, W) + b)
    RMSE = tf.sqrt(tf.reduce_mean((Y - hypothesis_p) ** 2))
    RMSE_error, Y_predict = sess.run([RMSE, hypothesis_p], feed_dict={X: X_in, Y: Y_in})
    F1_score = skm.f1_score(Y_in, Y_predict > 0.5)
    W = sess.run(W)
    return RMSE_error, F1_score, W
