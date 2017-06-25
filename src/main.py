# -*- coding: utf-8 -*-

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

import utils

# Конфигурация, описание дата-сета
dataset_file = '../data/cars.csv'
target_column = 8
iris = {
    'feature_names': ['Car', 'Price', 'Experience', 'Car age', 'Gender', 'Ownership type', 'Vehicle class', 'Annual mileage'],
    'target_names': ['High frequency', 'Middle frequency', 'Low frequency']
}

# Загружаем dataset
X, Y = utils.load_dataset(dataset_file, tuple(range(len(iris['feature_names']))), (target_column,))
Y_binarized = label_binarize(Y, classes=range(len(iris['target_names'])))
n_classes = Y_binarized.shape[1]

# Разделяем выборку на train и test в пропорции 1/9
X_train, X_test, y_train, y_test = train_test_split(X, Y_binarized, test_size=0.9)

# Берем decision tree классификатор и обучаем его по train выборке
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Визуализируем дерево решений, которое сформировал классификатор, и сохраняем в pdf-файл
utils.decision_tree_save(clf, iris, "decision_tree.pdf")

# Предсказываем target-значения по выборке, опираясь на обученный ранее алгоритм (делаем выбор по дереву решений)
y_score = clf.predict(X_test)

# Строим ROC-кривые для оценки качества классификации (по всем классам)
utils.draw_roc_curve(n_classes, y_test, y_score)

# Делаем кросс-валидацию с указанным кол-вом разбиений и показываем на графике средние ошибки
utils.draw_cross_val_score(clf, X, Y_binarized, cv=10)
