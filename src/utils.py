import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import tree
import pydotplus
from sklearn import metrics
from sklearn.model_selection import cross_val_score

def plot(data, labels, column_1_number, column_2_number, title):
    X = np.array(data)[:, column_1_number]
    Y = np.array(data)[:, column_2_number]
    cs = [labels[i] for i in range(len(labels))]
    plt.scatter(X, Y, c=cs, s=15)
    plt.xlim(0)
    plt.ylim(0)
    plt.title(title)
    plt.show()

def load_dataset(file, data_columns, target_column):
    x = genfromtxt(file, delimiter=',', dtype=float, skip_header=True, usecols=data_columns)
    y = genfromtxt(file, delimiter=',', dtype=float, skip_header=True, usecols=target_column)
    return x, y

def decision_tree_save(clf, iris, file, colorize=False):
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris['feature_names'],
                                    class_names=iris['target_names'],
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    nodes = graph.get_node_list()

    if colorize:
        for node in nodes:
            if node.get_label():
                class_name = node.get_label().split('class = ')[1].split('>')[0]
                if class_name == 'High frequency':
                    node.set_fillcolor('#FF0000')
                elif class_name == 'Middle frequency':
                    node.set_fillcolor('#FFA200')
                elif class_name == 'Low frequency':
                    node.set_fillcolor('#058900')

    graph.write_pdf(file)

def draw_roc_curve(classes, y_test, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.show()

def draw_cross_val_score(clf, X, Y, cv, scalled=False):
    t = cross_val_score(clf, X, Y, cv=cv)
    plt.plot(range(cv), t, 'k--')
    if scalled:
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
    plt.title('Cross-validation result')
    plt.show()
