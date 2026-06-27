import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("Инициализация текстового процессора и обучение AI")
file_path = "c:/Users/user/Documents/titles_data.csv" 

try:
    try:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, sep=';', encoding='cp1251')
except Exception as e:
    print(f"Ошибка при чтении файла {file_path}: {e}")
    exit()

# Переименовываем колонки и чистим пропуски
df = df.rename(columns={'titles': 'text', 'target': 'is_clickbait'}).dropna(subset=['text', 'is_clickbait'])

# Нижний регистр для исключения разницы в написании слов
X_text = df['text'].str.lower()
y = df['is_clickbait'].astype(int)

# Разделение на обучающую и тестовую выборки (70% на 30%) с балансировкой классов
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)

# Векторизация слов и словосочетаний (TF-IDF)
vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# Обучение Линейной Модели (Логистическая регрессия)
model = LogisticRegression(C=2.0, max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Проверка качества на скрытых тестовых данных
y_pred = model.predict(X_test_tfidf)
print(f"\nНовая ML-модель успешно обучена!")
print(f"Общая точность (Accuracy): {accuracy_score(y_test, y_pred):.4f}")

# Вывод
print("Classification report")
report = classification_report(y_test, y_pred, target_names=['Норма (Класс 0)', 'Кликбейт (Класс 1)'], digits=4)
print(report)

# Извлечение топ-маркеров кликбейта из обученной модели
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_[0]
top_clickbait_indices = np.argsort(coefficients)[-5:][::-1]
top_clickbait_words = feature_names[top_clickbait_indices]

def analyze_text():
    user_input = text_entry.get().strip()
    
    if not user_input:
        result_label.config(text="⚠️ Введите текст для анализа!", foreground="#cca700")
        stats_label.config(text="")
        return

    # Обработка введённой строки
    clean_input = [user_input.lower()]
    X_user = vectorizer.transform(clean_input)
    
    prediction = model.predict(X_user)[0]
    probabilities = model.predict_proba(X_user)[0]
    confidence = probabilities[prediction] * 100

    # Визуальное переключение вердиктов
    if prediction == 1:
        result_label.config(text=f"🚨 КЛИКБЕЙТ! (Уверенность: {confidence:.1f}%)", foreground="#d32f2f")
    else:
        result_label.config(text=f"✅ НАДЁЖНЫЙ ЗАГОЛОВОК (Уверенность: {confidence:.1f}%)", foreground="#388e3c")
        
    # Поиск слов-триггеров в тексте пользователя
    words_in_input = [word for word in user_input.lower().split() if word in vectorizer.vocabulary_]
    found_markers = [word for word in words_in_input if model.coef_[0][vectorizer.vocabulary_[word]] > 0.5]
    
    stats_text = (f"Анализ смысла слов модели:\n"
                  f"• Всего распознано слов из словаря: {len(words_in_input)}\n"
                  f"• Найденные триггеры кликбейта: {', '.join(found_markers) if found_markers else 'Не обнаружены'}\n"
                  f"• Главные триггеры всего датасета: {', '.join(top_clickbait_words)}")
    stats_label.config(text=stats_text)

# Функция жесткой нативной вставки текста
def force_paste(event=None):
    text_entry.event_generate("<<Paste>>")
    return "break"

# Отображение меню правой кнопки мыши
def show_context_menu(event):
    context_menu.tk_popup(event.x_root, event.y_root)

# Конструирование окна Tkinter
root = tk.Tk()
root.title("NLP Система Оценки Смысла Заголовков")
root.geometry("550x420")
root.configure(bg="#f5f5f7")

style = ttk.Style()
style.theme_use('clam')
style.configure('TButton', font=('Helvetica', 11, 'bold'), background='#007aff', foreground='white')
style.map('TButton', background=[('active', '#0056b3')])

main_frame = tk.Frame(root, bg="#f5f5f7", padx=25, pady=25)
main_frame.pack(fill=tk.BOTH, expand=True)

header_label = tk.Label(main_frame, text="Смысловой AI Детектор Кликбейта", font=('Helvetica', 16, 'bold'), bg="#f5f5f7", fg="#1d1d1f")
header_label.pack(anchor="w", pady=(0, 5))

subheader_label = tk.Label(main_frame, text="Модель оценивает скрытый смысл слов и маркеры манипуляций:", font=('Helvetica', 10), bg="#f5f5f7", fg="#86868b")
subheader_label.pack(anchor="w", pady=(0, 15))

# Поле ввода текста
text_entry = ttk.Entry(main_frame, font=('Helvetica', 12))
text_entry.pack(fill=tk.X, ipady=8, pady=(0, 15))
text_entry.focus()

# === КОНТЕКСТНОЕ МЕНЮ И СИСТЕМНЫЕ БИНДЫ ДЛЯ ВСТАВКИ ===
context_menu = tk.Menu(root, tearoff=0)
context_menu.add_command(label="Вставить", command=force_paste)

# Привязка вызова меню к правой кнопке мыши
text_entry.bind("<Button-3>", show_context_menu)

# Принудительная маршрутизация Ctrl+V на нативный метод буфера обмена ОС
root.bind("<Control-v>", force_paste)
root.bind("<Control-V>", force_paste)
root.bind("<Return>", lambda event: analyze_text())

# Интерфейсные кнопки и панели результатов
analyze_button = ttk.Button(main_frame, text="Проверить заголовок", command=analyze_text)
analyze_button.pack(fill=tk.X, ipady=6, pady=(0, 20))

result_label = tk.Label(main_frame, text="Ожидание ввода...", font=('Helvetica', 14, 'bold'), bg="#f5f5f7", fg="#515154")
result_label.pack(pady=(0, 15))

separator = ttk.Separator(main_frame, orient='horizontal')
separator.pack(fill=tk.X, pady=(0, 15))

stats_label = tk.Label(main_frame, text="", font=('Courier New', 10), bg="#ffffff", fg="#1d1d1f", justify=tk.LEFT, anchor="w", relief=tk.SOLID, bd=1, padx=15, pady=10)
stats_label.pack(fill=tk.X)

# Запуск GUI цикла
root.mainloop()
