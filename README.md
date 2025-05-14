# Loan Approval Prediction System

Система прогнозирования одобрения кредита с использованием ML-моделей и веб-интерфейса

## Основные функции:
- Прогноз вероятности одобрения кредита (0-100%)
- Обработка 11 признаков (числовые и категориальные)
- Визуализация важности признаков
- Интерактивный веб-интерфейс с формой заявки
- Поддержка как одиночных, так и пакетных предсказаний

## Технологии:
- Python 3.8+
- Flask (веб-фреймворк)
- Scikit-learn (машинное обучение)
- Pandas/Numpy (обработка данных)
- Pyngrok (туннелирование для демонстрации)
- Matplotlib/Seaborn (визуализация)
Модель:
Алгоритм: Decision Tree Classifier
Оптимизация гиперпараметров через GridSearchCV
Метрики качества: F1-score

## Установка и запуск:
```bash
git clone https://github.com/yourusername/loan-approval-prediction.git 
cd loan-approval-prediction
pip install -r requirements.txt
python app.py
