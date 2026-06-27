import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

class MyLinearRegression:
    """Собственная линейная регрессия с градиентным спуском"""
    def __init__(self, lr=0.01, iters=2000):
        self.lr = lr
        self.iters = iters
        self.coef_ = None       # Веса признаков (w)
        self.intercept_ = None  # Свободный коэффициент (b)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(self.iters):
            # Линейное предсказание: y = X*w + b
            y_pred = np.dot(X, self.coef_) + self.intercept_
            
            # Расчет градиентов функции стоимости (MSE)
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            # Обновление параметров
            self.coef_ -= self.lr * dw
            self.intercept_ -= self.lr * db
        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


class RegressorNode:
    """Узел дерева регрессии"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Среднее значение стоимости недвижимости в листе

    def is_leaf(self):
        return self.value is not None


class MyDecisionTreeRegressor:
    """Собственное дерево решений для задачи регрессии (по критерию дисперсии)"""
    def __init__(self, max_depth=3, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _variance_reduction(self, y, y_l, y_r):
        """Критерий информативности — уменьшение дисперсии"""
        parent_variance = np.var(y) if len(y) > 0 else 0
        weight_l = len(y_l) / len(y)
        weight_r = len(y_r) / len(y)
        
        # Считаем взвешенную дисперсию потомков
        children_variance = (weight_l * np.var(y_l)) + (weight_r * np.var(y_r))
        return parent_variance - children_variance

    def _best_split(self, X, y):
        best_gain = -1
        best_idx, best_thresh = None, None
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None, None

        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            # Ограничиваем пороги квантилями для ускорения вычислений
            if len(thresholds) > 10:
                thresholds = np.percentile(X[:, feat_idx], [10, 30, 50, 70, 90])

            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                y_l, y_r = y[left_mask], y[~left_mask]
                
                if len(y_l) == 0 or len(y_r) == 0:
                    continue

                gain = self._variance_reduction(y, y_l, y_r)

                if gain > best_gain:
                    best_gain = gain
                    best_idx = feat_idx
                    best_thresh = thresh
        return best_idx, best_thresh

    def _build_tree(self, X, y, depth=0):
        # Значение в листе — среднее арифметическое целевой переменной
        leaf_value = np.mean(y) if len(y) > 0 else 0.0

        if depth >= self.max_depth or len(y) < self.min_samples_split or np.var(y) == 0:
            return RegressorNode(value=leaf_value)

        feat_idx, thresh = self._best_split(X, y)
        if feat_idx is None:
            return RegressorNode(value=leaf_value)

        left_mask = X[:, feat_idx] <= thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return RegressorNode(feature=feat_idx, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        return self

    def _predict_row(self, node, x):
        if node.is_leaf(): 
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X):
        return np.array([self._predict_row(self.root, x) for x in X])


# Функции расчета метрик регрессии вручную
def my_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def my_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def my_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def my_train_test_split(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    return X[indices[:split_idx]], X[indices[split_idx:]], y[indices[:split_idx]], y[indices[split_idx:]]

# --- Шаг 1: Исследование данных и целевой переменной ---
print("--- Шаг 1: Загрузка и EDA ---")
california = fetch_california_housing(as_frame=True)
df = california.frame

# Названия ключевых признаков:
# MedInc - медианный доход в районе (очень сильно влияет на цену)
# AveRooms - среднее количество комнат в домах района
# Latitude - географическая широта (влияет на близость к побережью/дорогим городам)

X_selected = df[['MedInc', 'AveRooms', 'Latitude']].to_numpy()
y = df['MedHouseVal'].to_numpy() # Целевая переменная (в сотнях тысяч $)

print(f"Выбранные признаки для анализа недвижимости: ['MedInc', 'AveRooms', 'Latitude']")
print(f"Размер выборки: {X_selected.shape[0]} объектов.")

# Визуализация 1: Гистограмма целевой переменной
plt.figure(figsize=(7, 4))
plt.hist(y, bins=50, color='skyblue', edgecolor='black')
plt.title("Распределение медианной стоимости домов (Target)")
plt.xlabel("Стоимость (в сотнях тысяч $)")
plt.ylabel("Количество районов")
plt.grid(True, alpha=0.3)
plt.show()

# Визуализация 2: Связь дохода населения и комнатности с ценой
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].scatter(df['MedInc'], y, alpha=0.4, color='orange', s=10)
ax[0].set_title("Связь: Медианный доход vs Стоимость дома")
ax[0].set_xlabel("Медианный доход населения в районе")
ax[0].set_ylabel("Стоимость дома")

ax[1].scatter(df['AveRooms'], y, alpha=0.4, color='green', s=10)
ax[1].set_xlim(0, 15) # Ограничим выбросы по комнатам для наглядности
ax[1].set_title("Связь: Среднее число комнат vs Стоимость дома")
ax[1].set_xlabel("Среднее число комнат (AveRooms)")
ax[1].set_ylabel("Стоимость дома")
plt.tight_layout()
plt.show()

# Разделение данных на train и test
X_train, X_test, y_train, y_test = my_train_test_split(X_selected, y, test_size=0.3, random_state=42)

# ВАЖНО: Для корректной работы градиентного спуска линейной регрессии 
# приведем данные к единому масштабу (Z-масштабирование)
means, stds = np.mean(X_train, axis=0), np.std(X_train, axis=0)
stds[stds == 0] = 1.0
X_train_scaled = (X_train - means) / stds
X_test_scaled = (X_test - means) / stds


# --- Шаг 2: Обучение и оценка линейной модели ---
print("\n--- Шаг 2: Результаты Линейной Регрессии ---")
lin_reg = MyLinearRegression(lr=0.05, iters=3000)
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

mae_lin = my_mae(y_test, y_pred_lin)
mse_lin = my_mse(y_test, y_pred_lin)
r2_lin = my_r2(y_test, y_pred_lin)

print(f"MAE (Средняя абсолютная ошибка): {mae_lin:.4f} (~{mae_lin*100:.1f} тыс. $)")
print(f"MSE (Средняя квадратичная ошибка): {mse_lin:.4f}")
print(f"R2 (Коэффициент детерминации): {r2_lin:.4f}")
print(f"Свободный коэффициент (model.intercept_): {lin_reg.intercept_:.4f}")
print(f"Коэффициенты признаков (model.coef_ для ['MedInc', 'AveRooms', 'Latitude']): {lin_reg.coef_}")


# --- Шаг 3: Сравнение с нелинейной моделью ---
print("\n--- Шаг 3: Результаты Дерева Регрессии (max_depth=3) ---")
tree_reg = MyDecisionTreeRegressor(max_depth=3, min_samples_split=10)
tree_reg.fit(X_train, y_train) # Деревьям масштабирование не требуется
y_pred_tree = tree_reg.predict(X_test)

mae_tree = my_mae(y_test, y_pred_tree)
r2_tree = my_r2(y_test, y_pred_tree)

print(f"MAE дерева регрессии: {mae_tree:.4f} (~{mae_tree*100:.1f} тыс. $)")
print(f"R2 дерева регрессии: {r2_tree:.4f}")


# --- Шаг 4: Визуализация предсказаний лучшей модели ---
print("\n--- Шаг 4: Построение Scatter plot для лучшей модели ---")
# Выбираем лучшую модель по метрике R2
best_preds = y_pred_lin if r2_lin > r2_tree else y_pred_tree
best_name = "Линейная регрессия" if r2_lin > r2_tree else "Дерево решений"
print(f"Лучшая модель по коэффициенту R2: {best_name}")

plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_preds, alpha=0.3, color='purple', s=8)
# Идеальная прямая y = x
ideal_line = np.linspace(min(y_test), max(y_test), 100)
plt.plot(ideal_line, ideal_line, color='red', linestyle='--', lw=2, label='Идеальные предсказания (y = x)')
plt.title(f"Реальные vs Предсказанные значения ({best_name})")
plt.xlabel("Реальная стоимость жилья")
plt.ylabel("Предсказанная стоимость жилья")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
