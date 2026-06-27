import pandas as pd
import numpy as np
import math

class MyStandardScaler:
    """Собственная стандартизация числовых признаков (Z-score)"""
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds == 0] = 1.0

    def transform(self, X):
        return (X - self.means) / self.stds

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class MyOneHotEncoder:
    """Собственный кодировщик категориальных данных"""
    def __init__(self):
        self.categories_ = {}

    def fit(self, X):
        for col_idx in range(X.shape[1]):
            self.categories_[col_idx] = np.unique(X[:, col_idx])

    def transform(self, X):
        encoded_blocks = []
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            cats = self.categories_[col_idx]
            block = np.zeros((len(col_data), len(cats)))
            for idx, cat in enumerate(cats):
                block[col_data == cat, idx] = 1.0
            encoded_blocks.append(block)
        return np.hstack(encoded_blocks)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# --- МОДЕЛЬ 1: Логистическая регрессия ---
class MyLogisticRegression:
    """Собственная логистическая регрессия с градиентным спуском"""
    def __init__(self, lr=0.05, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # Ограничиваем z, чтобы избежать переполнения (overflow) в экспоненте
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Градиентный спуск
        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Вычисляем градиенты
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Обновляем параметры
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return np.array([1 if i > 0.5 else 0 for i in y_predicted])


# --- МОДЕЛЬ 2: Дерево решений ---
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class MyDecisionTreeClassifier:
    """Собственное дерево решений (критерий Джини)"""
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def _gini(self, y):
        m = len(y)
        if m == 0: return 0
        p_true = np.sum(y) / m
        return 1.0 - (p_true**2 + (1 - p_true)**2)

    def _best_split(self, X, y):
        best_gini = 999
        best_idx, best_thresh = None, None
        n_samples, n_features = X.shape

        if n_samples <= 1: return None, None

        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            # Если уникальных значений слишком много, берем квантили для ускорения
            if len(thresholds) > 10:
                thresholds = np.percentile(X[:, feat_idx], [10, 25, 50, 75, 90])

            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                y_l, y_r = y[left_mask], y[~left_mask]
                if len(y_l) == 0 or len(y_r) == 0: continue

                gini_split = (len(y_l) / n_samples) * self._gini(y_l) + (len(y_r) / n_samples) * self._gini(y_r)
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_idx = feat_idx
                    best_thresh = thresh
        return best_idx, best_thresh

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) <= 1 or len(y) < 5:
            return Node(value=1 if np.sum(y) >= len(y)/2 else 0)

        feat_idx, thresh = self._best_split(X, y)
        if feat_idx is None:
            return Node(value=1 if np.sum(y) >= len(y)/2 else 0)

        left_mask = X[:, feat_idx] <= thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=feat_idx, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_row(self, node, x):
        if node.is_leaf(): return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X):
        return np.array([self._predict_row(self.root, x) for x in X])


# --- МОДЕЛЬ 3: Метод k-ближайших соседей (KNN) ---
class MyKNeighborsClassifier:
    """Собственный KNN (Евклидово расстояние)"""
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _predict_row(self, x):
        # Вычисляем Евклидово расстояние от точки x до всех точек обучающей выборки
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # Получаем индексы k ближайших соседей
        k_indices = np.argsort(distances)[:self.k]
        # Берем их метки классов
        k_nearest_labels = self.y_train[k_indices]
        # Возвращаем самый частый класс среди соседей
        return 1 if np.sum(k_nearest_labels) >= self.k / 2 else 0

    def predict(self, X):
        return np.array([self._predict_row(x) for x in X])

# Шаг 1: Загрузка
df = pd.read_csv('C:/Users/user/Documents/telecom_churn (1).csv')
df['churn'] = df['churn'].astype(int)

X_raw = df.drop(columns=['churn', 'phone number'])
y_raw = df['churn'].to_numpy()

cat_cols = X_raw.select_dtypes(include=['object']).columns.tolist()
num_cols = X_raw.select_dtypes(exclude=['object']).columns.tolist()

# Шаг 2: Разделение данных 75/25
np.random.seed(42)
indices = np.arange(len(df))
np.random.shuffle(indices)
train_size = int(0.75 * len(df))
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train_num, X_test_num = X_raw[num_cols].iloc[train_idx].to_numpy(), X_raw[num_cols].iloc[test_idx].to_numpy()
X_train_cat, X_test_cat = X_raw[cat_cols].iloc[train_idx].to_numpy(), X_raw[cat_cols].iloc[test_idx].to_numpy()
y_train, y_test = y_raw[train_idx], y_raw[test_idx]

# Шаг 3: Масштабирование и кодирование
scaler = MyStandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

encoder = MyOneHotEncoder()
X_train_cat_encoded = encoder.fit_transform(X_train_cat)
X_test_cat_encoded = encoder.transform(X_test_cat)

X_train_final = np.hstack([X_train_num_scaled, X_train_cat_encoded])
X_test_final = np.hstack([X_test_num_scaled, X_test_cat_encoded])

# Шаг 4: Обучение всех 3 моделей и сравнение
my_models = {
    "Logistic Regression": MyLogisticRegression(iterations=1500),
    "Decision Tree": MyDecisionTreeClassifier(max_depth=6),
    "K-Neighbors (KNN)": MyKNeighborsClassifier(k=5)
}

accuracy_results = {}

print("--- СТАРТ ОБУЧЕНИЯ ВСЕХ МОДЕЛЕЙ ---")
for name, model in my_models.items():
    print(f"Обучаем {name}...")
    model.fit(X_train_final, y_train)
    preds = model.predict(X_test_final)
    
    # Считаем accuracy вручную: (количество совпадений / общее количество)
    acc = np.mean(preds == y_test)
    accuracy_results[name] = acc

print("\n=== СРАВНЕНИЕ РЕЗУЛЬТАТОВ (Accuracy) ===")
for name, acc in accuracy_results.items():
    print(f"{name}: {acc:.4f}")

# Выбираем лучшую модель
best_model_name = max(accuracy_results, key=accuracy_results.get)
print(f"\nНаилучший результат показала модель: {best_model_name}")

# Шаг 5: Матрица ошибок для лучшей модели (вручную)
best_model = my_models[best_model_name]
best_preds = best_model.predict(X_test_final)

# Считаем компоненты матрицы ошибок
tp = np.sum((y_test == 1) & (best_preds == 1))
tn = np.sum((y_test == 0) & (best_preds == 0))
fp = np.sum((y_test == 0) & (best_preds == 1))
fn = np.sum((y_test == 1) & (best_preds == 0))

print(f"\n=== МАТРИЦА ОШИБОК ДЛЯ ЛУЧШЕЙ МОДЕЛИ ({best_model_name}) ===")
print(f"Истинно лояльные (прогноз совпал): {tn}")
print(f"Истинно ушедшие (прогноз совпал): {tp}")
print(f"Ошибочно предсказаны как ушедшие (False Positive): {fp}")
print(f"Ошибочно предсказаны как лояльные (False Negative): {fn}")
