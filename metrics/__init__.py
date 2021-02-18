import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(all_predictions, all_labels):
    """寻找最佳的分类边界, 在0到1之间"""
    # 展平所有的预测结果和对应的标记
    # all_predictions为0到1之间的实数
    # numpy中的ravel()、flatten()、squeeze()都有将多维数组转换为一维数组的功能，区别：
    # ravel()：可以“拉平”多维数组，如果没有必要，不会产生源数据的副本
    # flatten()：返回源数据的副本
    # squeeze()：只能对维数为1的维度降维
    # reshape(-1)也可以“拉平”多维数组
    all_predictions = np.ravel(all_predictions)
    all_labels = np.ravel(all_labels)
    # 从0到1以0.01为间隔定义99个备选阈值, 分别是从0.01-0.99之间
    thresholds = [i / 100 for i in range(100)]
    all_f1s = []
    for threshold in thresholds:
        # 计算当前阈值的f1 score
        preds = (all_predictions >= threshold).astype("int")
        f1 = f1_score(y_true=all_labels, y_pred=preds)
        all_f1s.append(f1)
    # 找出可以使f1 socre最大的阈值
    best_threshold = thresholds[int(np.argmax(np.array(all_f1s)))]
    print("best threshold is {}".format(str(best_threshold)))
    print(all_f1s)
    return best_threshold
