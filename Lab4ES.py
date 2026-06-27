import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import learning_curve, GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE

class Node:
    """Узел многоклассового дерева решений"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class MyDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """Кастомное многоклассовое дерево решений"""
    def __init__(self, max_depth=5, min_samples_split=2, class_weight=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.root = None
        self.weights_dict = None

    def _compute_class_weights(self, y):
        if self.class_weight == 'balanced':
            n_samples = len(y)
            classes = np.unique(y)
            n_classes = len(classes) if len(classes) > 0 else 1
            bincount = np.bincount(y, minlength=n_classes)
            bincount[bincount == 0] = 1
            weights = n_samples / (n_classes * bincount)
            return {c: weights[c] for c in classes}
        return {c: 1.0 for c in np.unique(y)}

    def _weighted_gini(self, y):
        m = len(y)
        if m == 0: return 0
        classes = np.unique(y)
        w_sums = {c: np.sum([self.weights_dict.get(c, 1.0) for val in y if val == c]) for c in classes}
        total_w = np.sum(list(w_sums.values()))
        if total_w == 0: return 0
        gini = 1.0
        for c in classes:
            p_c = w_sums[c] / total_w
            gini -= p_c ** 2
        return gini

    def _best_split(self, X, y, idx_features):
        best_gini = 999
        best_idx, best_thresh = None, None
        n_samples = X.shape[0]

        if n_samples < self.min_samples_split:
            return None, None

        for feat_idx in idx_features:
            thresholds = np.unique(X[:, feat_idx])
            # Ограничение порогов квантилями для ускорения поиска
            if len(thresholds) > 10:
                thresholds = np.percentile(X[:, feat_idx], [10, 30, 50, 70, 90])
                
            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                y_l, y_r = y[left_mask], y[~left_mask]
                if len(y_l) == 0 or len(y_r) == 0:
                    continue

                gini_split = (len(y_l) / n_samples) * self._weighted_gini(y_l) + (len(y_r) / n_samples) * self._weighted_gini(y_r)

                if gini_split < best_gini:
                    best_gini = gini_split
                    best_idx = feat_idx
                    best_thresh = thresh
        return best_idx, best_thresh

    def _build_tree(self, X, y, depth=0):
        classes = np.unique(y)
        w_counts = [np.sum([self.weights_dict.get(c, 1.0) for val in y if val == c]) for c in classes]
        leaf_value = classes[np.argmax(w_counts)] if len(classes) > 0 else 0

        if depth >= self.max_depth or len(classes) <= 1 or len(y) < self.min_samples_split:
            return Node(value=leaf_value)

        n_features = X.shape[1]
        n_sub = int(np.sqrt(n_features))
        if n_sub == 0: n_sub = 1
        idx_features = np.random.choice(n_features, n_sub, replace=False)

        feat_idx, thresh = self._best_split(X, y, idx_features)
        if feat_idx is None:
            return Node(value=leaf_value)

        left_mask = X[:, feat_idx] <= thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=feat_idx, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        self.weights_dict = self._compute_class_weights(y)
        self.root = self._build_tree(X, y)
        return self

    def _predict_row(self, node, x):
        if node.is_leaf(): return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_row(self.root, x) for x in X])


class MyRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Кастомный Случайный Лес, полностью совместимый с sklearn конвейерами"""
    def __init__(self, n_estimators=15, max_depth=5, min_samples_split=2, class_weight=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_b, y_b = X[indices], y[indices]
            
            tree = MyDecisionTreeClassifier(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split, 
                class_weight=self.class_weight
            )
            tree.fit(X_b, y_b)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X)
        tree_preds = np.column_stack([tree.predict(X) for tree in self.trees])
        final_preds = []
        for row in tree_preds:
            vals, counts = np.unique(row, return_counts=True)
            final_preds.append(vals[np.argmax(counts)])
        return np.array(final_preds)

    def score(self, X, y):
        """Необходимо для корректного расчета метрик внутри GridSearchCV"""
        from sklearn.metrics import f1_score
        return f1_score(y, self.predict(X), average='macro')


# Вспомогательная функция разделения и расчета метрик
def my_stratified_train_test_split(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    train_indices, test_indices = [], []
    for c in np.unique(y):
        c_indices = np.where(y == c)[0]
        np.random.shuffle(c_indices)
        n_test = int(len(c_indices) * test_size)
        test_indices.extend(c_indices[:n_test])
        train_indices.extend(c_indices[n_test:])
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def compute_macro_f1(y_true, y_pred):
    classes = np.unique(y_true)
    f1_list = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_list.append(f1)
    return np.mean(f1_list)

# Шаг 0: Загрузка данных
wine = datasets.load_wine()
X_raw, y_raw = wine.data, wine.target
X_train, X_test, y_train, y_test = my_stratified_train_test_split(X_raw, y_raw, test_size=0.3, random_state=42)

results_table = {}

# --- ПУНКТ 1: Базовая модель
rf_base = MyRandomForestClassifier(n_estimators=15, max_depth=5)
rf_base.fit(X_train, y_train)
f1_base = compute_macro_f1(y_test, rf_base.predict(X_test))
results_table["Базовая модель (Практика 3)"] = f1_base


# --- ПУНКТ 2: Применение продвинутого метода SMOTE ---
print("=== Шаг 1: Применение SMOTE ===")
# Инициализируем SMOTE (k_neighbors=3, так как выборка вина маленькая)
smote = SMOTE(k_neighbors=3, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"До SMOTE на train: {np.bincount(y_train)}")
print(f"После SMOTE на train: {np.bincount(y_train_res)}")

# Обучаем модель на ИСКУССТВЕННО СБАЛАНСИРОВАННЫХ данных
rf_smote = MyRandomForestClassifier(n_estimators=15, max_depth=5)
rf_smote.fit(X_train_res, y_train_res)
f1_smote = compute_macro_f1(y_test, rf_smote.predict(X_test))
results_table["+ Продвинутый метод SMOTE"] = f1_smote
print(f"Макро-F1 после SMOTE: {f1_smote:.4f}\n")


# --- ПУНКТ 3: Диагностика модели с помощью кривых обучения ---
print("=== Шаг 2: Построение кривых обучения ===")
# Будем диагностировать лучшую модель после SMOTE
train_sizes, train_scores, val_scores = learning_curve(
    MyRandomForestClassifier(n_estimators=15, max_depth=5),
    X_train_res, y_train_res,
    cv=3,
    scoring='f1_macro',
    train_sizes=np.linspace(0.3, 1.0, 5),
    n_jobs=1  # Однопоточно для стабильности кастомных классов
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Точность на Train")
plt.plot(train_sizes, val_mean, 'o-', color="g", label="Точность на Validation")
plt.title("Кривые обучения (Learning Curves) для нашей модели")
plt.xlabel("Размер обучающей выборки")
plt.ylabel("Macro F1-score")
plt.legend(loc="best")
plt.grid(True)
plt.show()


# --- ПУНКТ 4: Комплексная настройка (GridSearchCV) ---
print("=== Шаг 3: Настройка гиперпараметров через GridSearchCV ===")
# Так как наш Random Forest не требует обязательного масштабирования, 
# мы можем напрямую оптимизировать его параметры с помощью StratifiedKFold
param_grid = {
    'max_depth': [3, 5, 8],
    'n_estimators': [10, 20, 30]
}

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=MyRandomForestClassifier(),
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='f1_macro',
    n_jobs=1
)

# Запуск поиска на сбалансированных данных
grid_search.fit(X_train_res, y_train_res)
print(f"Лучшие параметры: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
f1_grid = compute_macro_f1(y_test, best_model.predict(X_test))
results_table["+ Оптимизация параметров (GridSearchCV)"] = f1_grid


# --- ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ---
print("\n=== ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ДЛЯ ОТЧЕТА ===")
print(f"{'Шаг алгоритма':<40} | {'Итоговый Макро-F1 (test)':<25}")
print("-" * 68)
for step, f1_val in results_table.items():
    print(f"{step:<40} | {f1_val:<25.4f}")
