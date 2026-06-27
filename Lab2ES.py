import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve, 
    auc
)

class Node:
    """Узел дерева решений"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature       # Индекс признака
        self.threshold = threshold   # Порог деления
        self.left = left             # Левое поддерево
        self.right = right           # Правое поддерево
        self.value = value           # Значение ответа в листе (0 или 1)

    def is_leaf(self):
        return self.value is not None


# Наследуемся от BaseEstimator и ClassifierMixin, чтобы GridSearchCV принял нашу модель за родную
class MyDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def _collect_gains(self, node, gains):
        """Рекурсивно обходит дерево и собирает, на каких признаках уменьшался Джини"""
        if node is None or node.is_leaf():
            return
        
        # Для упрощения ставим базовый фиксированный прирост информации для каждого узла расщепления
        if node.feature is not None:
            gains[node.feature] += 1.0
            
        self._collect_gains(node.left, gains)
        self._collect_gains(node.right, gains)

    def _compute_feature_importances(self, n_features):
        """Возвращает нормализованный массив важности каждого признака"""
        gains = np.zeros(n_features)
        self._collect_gains(self.root, gains)
        total_gain = np.sum(gains)
        
        if total_w_gain := total_gain:
            return gains / total_w_gain
        return gains
    """Собственное дерево решений с поддержкой сбалансированных весов классов"""
    def __init__(self, max_depth=5, min_samples_split=2, class_weight=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.root = None
        self.classes_ = None
        self.weights_dict = None

    def _compute_class_weights(self, y):
        """Расчет весов для балансировки классов вручную"""
        if self.class_weight == 'balanced':
            n_samples = len(y)
            n_classes = len(np.unique(y))
            bincount = np.bincount(y)
            # Если какого-то класса нет в подвыборке кросс-валидации, защищаем от деления на 0
            bincount[bincount == 0] = 1
            weights = n_samples / (n_classes * bincount)
            return {0: weights[0], 1: weights[1]}
        return {0: 1.0, 1: 1.0}

    def _weighted_gini(self, y):
        """Взвешенный критерий Джини для борьбы с дисбалансом"""
        m = len(y)
        if m == 0: return 0
        
        w_sum_0 = np.sum([self.weights_dict[0] for val in y if val == 0])
        w_sum_1 = np.sum([self.weights_dict[1] for val in y if val == 1])
        total_w = w_sum_0 + w_sum_1
        
        if total_w == 0: return 0
        
        p_0 = w_sum_0 / total_w
        p_1 = w_sum_1 / total_w
        return 1.0 - (p_0**2 + p_1**2)

    def _best_split(self, X, y):
        best_gini = 999
        best_idx, best_thresh = None, None
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None, None

        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            # Ограничиваем количество порогов квантилями для ускорения GridSearchCV
            if len(thresholds) > 8:
                thresholds = np.percentile(X[:, feat_idx], [15, 30, 50, 70, 85])

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
        # Определение мажоритарного класса с учетом весов
        w_0 = np.sum([self.weights_dict[0] for val in y if val == 0])
        w_1 = np.sum([self.weights_dict[1] for val in y if val == 1])
        leaf_value = 1 if w_1 >= w_0 else 0

        if depth >= self.max_depth or len(np.unique(y)) <= 1 or len(y) < self.min_samples_split:
            return Node(value=leaf_value)

        feat_idx, thresh = self._best_split(X, y)
        if feat_idx is None:
            return Node(value=leaf_value)

        left_mask = X[:, feat_idx] <= thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=feat_idx, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        # 1. Защита от разреженных матриц и объектов np.matrix
        if hasattr(X, "toarray"):
            X = X.toarray()
        
        # Принудительно делаем базовым плотным массивом numpy ndarray
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        
        self.classes_ = np.unique(y)
        self.weights_dict = self._compute_class_weights(y)
        self.root = self._build_tree(X, y)
        # Считаем важность признаков после того, как дерево полностью построилось
        self.feature_importances_ = self._compute_feature_importances(X.shape[1])
        return self

    def predict(self, X):
        # 2. Защита от разреженных матриц на этапе предсказания
        if hasattr(X, "toarray"):
            X = X.toarray()
            
        # Принудительно делаем базовым плотным массивом numpy ndarray
        X = np.asarray(X)
        
        # Теперь итерация по строкам гарантированно отработает корректно
        return np.array([self._predict_row(self.root, x) for x in X])

    def predict_proba(self, X):
        # 3. Защита для метода генерации вероятностей
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X)
        
        preds = self.predict(X)
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 1.0 - preds
        probs[:, 1] = preds.astype(float)
        return probs
        
        self.classes_ = np.unique(y)
        self.weights_dict = self._compute_class_weights(y)
        self.root = self._build_tree(X, y)
        return self

    def _predict_row(self, node, x):
        if node.is_leaf(): return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

# Шаг 1: Загрузка и первичная диагностика дисбаланса
print("--- Шаг 1: Диагностика проблемы ---")
df = pd.read_csv('C:/Users/user/Documents/telecom_churn (1).csv')
df['churn'] = df['churn'].astype(int)

X = df.drop(columns=['churn', 'phone number'])
y = df['churn'].to_numpy()

print(f"Общее число лояльных клиентов (0): {np.sum(y == 0)}")
print(f"Общее число ушедших клиентов (1): {np.sum(y == 1)}")

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numeric_features = X.select_dtypes(exclude=['object']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Обучаем БАЗОВУЮ версию твоей модели (как в ЛР1, без балансировки)
preprocessor_base = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
base_pipeline = Pipeline([
    ('preprocessor', preprocessor_base),
    ('model', MyDecisionTreeClassifier(max_depth=6, min_samples_split=2, class_weight=None))
])
base_pipeline.fit(X_train, y_train)
y_pred_base = base_pipeline.predict(X_test)

print("\n[Развернутый classification_report для БАЗОВОЙ модели]:")
print(classification_report(y_test, y_pred_base, target_names=['Лояльные', 'Ушедшие']))

# Шаг 2: Создание надежного конвейера (Pipeline) с оптимизацией
print("--- Шаг 2: Создание конвейера и GridSearchCV ---")
preprocessor_opt = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Объединяем обработку и НАШУ модель со встроенным class_weight='balanced'
opt_pipeline = Pipeline([
    ('preprocessor', preprocessor_opt),
    ('model', MyDecisionTreeClassifier(class_weight='balanced'))
])

# Шаг 3: Системный поиск лучших параметров (GridSearchCV)
# Задаем сетку параметров для НАШЕГО дерева решений
param_grid = {
    'model__max_depth': [4, 6, 8],
    'model__min_samples_split': [2, 5, 10]
}

print("Запуск GridSearchCV с 5-ти блочной кросс-валидацией по твоей модели...")
grid_search = GridSearchCV(opt_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Лучшая комбинация параметров (best_params_): {grid_search.best_params_}")
print(f"Средний результат кросс-валидации (best_score_ по F1): {grid_search.best_score_:.4f}")

# Шаг 4: Финальная оценка и интерпретация
print("\n--- Шаг 4: Оценка финальной модели на тест-выборке ---")
best_model = grid_search.best_estimator_

y_pred_opt = best_model.predict(X_test)
y_prob_opt = best_model.predict_proba(X_test)[:, 1]

# Строим матрицы ошибок для сравнения
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_base, display_labels=['Лояльные', 'Ушли'], cmap='Reds', ax=ax[0])
ax[0].set_title("Матрица ошибок (Старая модель из ЛР1)")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_opt, display_labels=['Лояльные', 'Ушли'], cmap='Blues', ax=ax[1])
ax[1].set_title("Матрица ошибок (Оптимизированная модель)")
plt.tight_layout()
plt.show()

# Расчет ROC-кривой и AUC-ROC
fpr, tpr, _ = roc_curve(y_test, y_prob_opt)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая нашей модели (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая оптимизированной модели')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Извлечение важности признаков
ohe_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_features + ohe_names
importances = best_model.named_steps['model'].feature_importances_

indices = np.argsort(importances)[::-1]
top_k = 10

plt.figure(figsize=(10, 6))
plt.title("Рейтинг важности признаков (ТОП-10 в нашем дереве)")
plt.bar(range(top_k), importances[indices[:top_k]], align="center", color='purple')
plt.xticks(range(top_k), [all_feature_names[i] for i in indices[:top_k]], rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Итоговая сравнительная таблица
print("\n=== СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК ДЛЯ ОТЧЕТА ===")
print(f"{'Метрика':<15} | {'Базовая модель (ЛР1)':<20} | {'Оптимизированная (ЛР2)':<22}")
print("-" * 62)
print(f"{'Accuracy':<15} | {accuracy_score(y_test, y_pred_base):<20.4f} | {accuracy_score(y_test, y_pred_opt):<22.4f}")
print(f"{'Precision':<15} | {precision_score(y_test, y_pred_base):<20.4f} | {precision_score(y_test, y_pred_opt):<22.4f}")
print(f"{'Recall':<15} | {recall_score(y_test, y_pred_base):<20.4f} | {recall_score(y_test, y_pred_opt):<22.4f}")
print(f"{'F1-score':<15} | {f1_score(y_test, y_pred_base):<20.4f} | {f1_score(y_test, y_pred_opt):<22.4f}")
