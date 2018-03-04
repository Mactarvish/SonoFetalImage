# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import MaUtilities as mu
from torch.autograd import Variable
from visdom import Visdom
viz = Visdom()
assert viz.check_connection()

from sklearn import metrics
import numpy as np

y_true = [0,1,4,1,3,2]
y_pred = [0,2,1,4,3,3]

#####
# Do classification task,
# then get the ground truth and the predict label named y_true and y_pred
def save_matrics(y_true, y_pred, losses, net_name):
    classify_report    = metrics.classification_report(y_true, y_pred)
    confusion_matrix   = metrics.confusion_matrix(y_true, y_pred)
    overall_accuracy   = metrics.accuracy_score(y_true, y_pred)
    precision_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
    average_accuracy   = np.mean(precision_for_each_class)
    score = metrics.accuracy_score(y_true, y_pred)

    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusion_matrix)
    print('acc_for_each_class : \n', precision_for_each_class)
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('overall_accuracy: {0:f}'.format(overall_accuracy))
    print('score: {0:f}'.format(score))
    dic = {'net_name': net_name, 'classify_report': classify_report, 'confusion_matrix': confusion_matrix, 'acc_for_each_class': precision_for_each_class,
     'average_accuracy': average_accuracy, 'overall_accuracy': overall_accuracy, 'score': score, 'losses': losses}
    torch.save(dic, 'matrics/%s' % (net_name))

# save_matrics(y_true, y_pred, [1,2,3,4,5,6], 'SBNet')
a = torch.load("/home/hdl2/Desktop/SonoFetalImage/matrics/ThresholdResnet")
print(a)
