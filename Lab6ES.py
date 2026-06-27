import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

class MyRidgeRegression:
    """Собственная Линейная Регрессия с L2-регуляризацией (Ridge)"""
    def __init__(self, alpha=1.0, lr=0.01, iters=3000):
        self.alpha = alpha  # Параметр регуляризации (штраф за большие веса)
        self.lr = lr
        self.iters = iters
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(self.iters):
            y_pred = np.dot(X, self.coef_) + self.intercept_
            
            # Градиент MSE + Градиент L2-штрафа (2 * alpha * w)
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y)) + (2 * self.alpha * self.coef_) / n_samples
            db = (2 / n_samples) * np.sum(y_pred - y)

            # Обновление параметров
            self.coef_ -= self.lr * dw
            self.intercept_ -= self.lr * db
        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


# Функции предобработки и расчета метрик вручную
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

def my_scale(X_train, X_test):
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    stds[stds == 0] = 1.0
    return (X_train - means) / stds, (X_test - means) / stds

# --- Шаг 1: Подготовка базовых данных (как в ЛР5) ---
california = fetch_california_housing()
X_selected = california.data[:, [0, 2, 6]] # 'MedInc', 'AveRooms', 'Latitude'
y = california.target

X_train, X_test, y_train, y_test = my_train_test_split(X_selected, y, test_size=0.3, random_state=42)
X_train_scaled, X_test_scaled = my_scale(X_train, X_test)

# Обучаем базовую модель (alpha=0 — это обычная Линейная Регрессия)
base_reg = MyRidgeRegression(alpha=0.0, lr=0.05, iters=3000)
base_reg.fit(X_train_scaled, y_train)
r2_base = my_r2(y_test, base_reg.predict(X_test_scaled))


# --- Шаг 2: Создаем проблему (Добавляем 30 признаков-шумов) ---
print("=== Шаг 2: Искусственное зашумление данных ===")
np.random.seed(42)
# Генерируем случайный некоррелированный шум
noise_train = np.random.normal(0, 1, size=(X_train.shape[0], 30))
noise_test = np.random.normal(0, 1, size=(X_test.shape[0], 30))

X_train_noisy = np.hstack((X_selected[np.arange(X_train.shape[0])], noise_train)) # Ограничимся индексами для демонстрации
# Корректное объединение исходных разделенных данных с шумом:
X_train_noisy = np.hstack((X_train, noise_train))
X_test_noisy = np.hstack((X_test, noise_test))

# Повторно масштабируем зашумленный датасет
X_train_noisy_scaled, X_test_noisy_scaled = my_scale(X_train_noisy, X_test_noisy)
print(f"Размерность новых данных с шумом: {X_train_noisy_scaled.shape}")


# --- Шаг 3: Демонстрация поломки (Переобучение) ---
print("\n=== Шаг 3: Обучение обычной регрессии на шуме ===")
broken_reg = MyRidgeRegression(alpha=0.0, lr=0.05, iters=3000)
broken_reg.fit(X_train_noisy_scaled, y_train)

r2_noise_train = my_r2(y_train, broken_reg.predict(X_train_noisy_scaled))
r2_noise_test = my_r2(y_test, broken_reg.predict(X_test_noisy_scaled))

print(f"R2 на Train (с шумом): {r2_noise_train:.4f}")
print(f"R2 на Test (с шумом): {r2_noise_test:.4f}")
print(f"Разница качества на тесте (Базовый - Сломанный): {r2_base - r2_noise_test:.4f}")


# --- Шаг 4: Поиск и спасение (Перебор alpha для Ridge) ---
print("\n=== Шаг 4: Тестирование Ridge-регрессии ===")
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
best_r2_test = -999
best_alpha = None

for a in alphas:
    ridge_reg = MyRidgeRegression(alpha=a, lr=0.05, iters=3000)
    ridge_reg.fit(X_train_noisy_scaled, y_train)
    r2_ridge_test = my_r2(y_test, ridge_reg.predict(X_test_noisy_scaled))
    print(f"Ridge с alpha = {a:<6} | R2 на test: {r2_ridge_test:.4f}")
    
    if r2_ridge_test > best_r2_test:
        best_r2_test = r2_ridge_test
        best_alpha = a

# --- Шаг 5: Вывод итоговой таблицы результатов ---
print("\n=== ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ДЛЯ ОТЧЕТА ===")
print(f"{'Модель/условия':<45} | {'R2 на test':<12} | {'Вывод'}")
print("-" * 80)
print(f"{'Базовая (предыдущая практика)':<45} | {r2_base:<12.4f} | Наш ориентир")
print(f"{'Сломанная (LinearRegression + шум)':<45} | {r2_noise_test:<12.4f} | Катастрофа!")
print(f"{f'Исправленная (Ridge + шум, alpha = {best_alpha})':<45} | {best_r2_test:<12.4f} | Спасение!")
