import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

class MyStandardScaler:
    """Собственный класс стандартизации признаков"""
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.means) / self.stds

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MyLogisticRegression:
    """Бинарная логистическая регрессия с градиентным спуском"""
    def __init__(self, lr=0.05, iters=1000):
        self.lr = lr
        self.iters = iters
        self.w = None
        self.b = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -20, 20)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.iters):
            linear_model = np.dot(X, self.w) + self.b
            y_predicted = self._sigmoid(linear_model)

            # Вычисление градиентов
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Обновление весов
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict_proba(self, X):
        linear_model = np.dot(X, self.w) + self.b
        return self._sigmoid(linear_model)


class MyMulticlassLogisticRegression:
    """Многоклассовая логистическая регрессия по стратегии One-vs-Rest"""
    def __init__(self, lr=0.05, iters=1000):
        self.lr = lr
        self.iters = iters
        self.models = []
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = []
        
        # Обучаем отдельную бинарную модель для каждого класса
        for c in self.classes:
            y_binary = np.where(y == c, 1, 0)
            model = MyLogisticRegression(lr=self.lr, iters=self.iters)
            model.fit(X, y_binary)
            self.models.append(model)
        return self

    def predict(self, X):
        # Собираем вероятности от каждой бинарной модели
        probs = np.column_stack([model.predict_proba(X) for model in self.models])
        # Выбираем класс с максимальной вероятностью
        return np.argmax(probs, axis=1)


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


class MyDecisionTreeClassifier:
    """Многоклассовое дерево решений с поддержкой весов классов"""
    def __init__(self, max_depth=5, min_samples_split=2, class_weight=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.root = None
        self.weights_dict = None

    def _compute_class_weights(self, y):
        """Расчет весов для балансировки многоклассовой выборки"""
        if self.class_weight == 'balanced':
            n_samples = len(y)
            classes = np.unique(y)
            n_classes = len(classes)
            bincount = np.bincount(y, minlength=n_classes)
            bincount[bincount == 0] = 1
            weights = n_samples / (n_classes * bincount)
            return {c: weights[c] for c in classes}
        return {c: 1.0 for c in np.unique(y)}

    def _weighted_gini(self, y):
        """Многоклассовый критерий Джини с учетом весов классов"""
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
        classes, counts = np.unique(y, return_counts=True)
        
        # Мажоритарный класс с учетом весов в листе
        w_counts = [np.sum([self.weights_dict.get(c, 1.0) for val in y if val == c]) for c in classes]
        leaf_value = classes[np.argmax(w_counts)] if len(classes) > 0 else 0

        if depth >= self.max_depth or len(classes) <= 1 or len(y) < self.min_samples_split:
            return Node(value=leaf_value)

        # Поддерживаем случайное подмножество признаков (для Random Forest)
        n_features = X.shape[1]
        n_sub = int(np.sqrt(n_features))
        idx_features = np.random.choice(n_features, n_sub, replace=False)

        feat_idx, thresh = self._best_split(X, y, idx_features)
        if feat_idx is None:
            return Node(value=leaf_value)

        left_mask = X[:, feat_idx] <= thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=feat_idx, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        self.weights_dict = self._compute_class_weights(y)
        self.root = self._build_tree(X, y)
        return self

    def _predict_row(self, node, x):
        if node.is_leaf(): return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X):
        return np.array([self._predict_row(self.root, x) for x in X])


class MyRandomForestClassifier:
    """Собственный Случайный Лес (Ансамбль кастомных деревьев)"""
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, class_weight=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            # Бутстрап выборка (случайный выбор строк с повторениями)
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
        # Собираем предсказания всех деревьев
        tree_preds = np.column_stack([tree.predict(X) for tree in self.trees])
        # Голосование большинства для каждой строки
        final_preds = []
        for row in tree_preds:
            vals, counts = np.unique(row, return_counts=True)
            final_preds.append(vals[np.argmax(counts)])
        return np.array(final_preds)


def my_stratified_train_test_split(X, y, test_size=0.3, random_state=42):
    """Собственная функция разделения данных со стратификацией"""
    np.random.seed(random_state)
    train_indices = []
    test_indices = []
    
    for c in np.unique(y):
        c_indices = np.where(y == c)[0]
        np.random.shuffle(c_indices)
        n_test = int(len(c_indices) * test_size)
        
        test_indices.extend(c_indices[:n_test])
        train_indices.extend(c_indices[n_test:])
        
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def my_classification_report(y_true, y_pred):
    """Собственный генератор отчета по метрикам для N классов"""
    classes = np.unique(y_true)
    report = {}
    macro_p, macro_r, macro_f1 = 0, 0, 0
    total_tp = 0
    
    print(f"{'Класс':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10} | {'Объектов':<10}")
    print("-" * 58)
    
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        total_tp += tp
        macro_p += p
        macro_r += r
        macro_f1 += f1
        
        print(f"Класс {c:<6} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f} | {support:<10}")
        report[c] = {"p": p, "r": r, "f1": f1, "sup": support}
        
    accuracy = total_tp / len(y_true)
    n_classes = len(classes)
    
    print("-" * 58)
    print(f"{'Accuracy':<12} | {'':<10} | {'':<10} | {accuracy:<10.4f} | {len(y_true):<10}")
    print(f"{'Macro Avg':<12} | {macro_p/n_classes:<10.4f} | {macro_r/n_classes:<10.4f} | {macro_f1/n_classes:<10.4f} | {len(y_true):<10}")
    return accuracy, macro_f1 / n_classes, report


def my_confusion_matrix(y_true, y_pred, title="Матрица ошибок"):
    """Ручной расчет и визуализация матрицы ошибок"""
    classes = np.unique(y_true)
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
        
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Отрисовка цифр внутри ячеек
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    plt.show()
    return cm

# Шаг 1: Загрузка и EDA
print("=== Шаг 1: Исследование данных (EDA) ===")
wine = datasets.load_wine()
X_raw, y_raw = wine.data, wine.target

print(f"Целевые классы винограда: {wine.target_names}")
print(f"Количество признаков химического анализа: {X_raw.shape[1]}")

# Проверка дисбаланса классов
classes, counts = np.unique(y_raw, return_counts=True)
for c, cnt in zip(classes, counts):
    print(f"Класс {c} ({wine.target_names[c]}): {cnt} объектов")

# Разделение данных с ручной стратификацией (30% на тест)
X_train, X_test, y_train, y_test = my_stratified_train_test_split(X_raw, y_raw, test_size=0.3, random_state=42)

# Стандартизация признаков (особенно важна для логистической регрессии!)
scaler = MyStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Шаг 2: Обучение базовых моделей
print("\n=== Шаг 2: Обучение базовой Logistic Regression ===")
lr_model = MyMulticlassLogisticRegression(lr=0.1, iters=1500)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
acc_lr, macro_f1_lr, _ = my_classification_report(y_test, y_pred_lr)
my_confusion_matrix(y_test, y_pred_lr, "Матрица ошибок: Логистическая регрессия")

print("\n=== Шаг 2: Обучение базового Random Forest ===")
# Было: rf_model = MyRandomForestClassifier(n_estimators=15, max_depth=5, random_state=42)
# Стало (исправленный вариант):
rf_model = MyRandomForestClassifier(n_estimators=15, max_depth=5)
rf_model.fit(X_train, y_train) # Для деревьев стандартизация не обязательна
y_pred_rf = rf_model.predict(X_test)
acc_rf, macro_f1_rf, report_rf = my_classification_report(y_test, y_pred_rf)
my_confusion_matrix(y_test, y_pred_rf, "Матрица ошибок: Случайный Лес (Базовый)")


# Шаг 3: Добавление балансировки весов классов в Случайный Лес
print("\n=== Шаг 3: Оптимизация Random Forest (class_weight='balanced') ===")
rf_balanced = MyRandomForestClassifier(n_estimators=15, max_depth=5, class_weight='balanced')
rf_balanced.fit(X_train, y_train)
y_pred_rf_bal = rf_balanced.predict(X_test)
acc_rf_bal, macro_f1_rf_bal, _ = my_classification_report(y_test, y_pred_rf_bal)
my_confusion_matrix(y_test, y_pred_rf_bal, "Матрица ошибок: Случайный Лес (Сбалансированный)")
