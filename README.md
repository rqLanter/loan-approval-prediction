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

## Не забудьте установить Ваш токен ngrok
Ниже приведён фрагмент кода где это необходимо сделать.
```python
# 6. Запуск Flask с ngrok
# Установка токена
ngrok.set_auth_token("") # УСТАНОВИТЕ ВАШ ТОКЕН!!!
```

## Установка и быстрый запуск:
```bash
git clone https://github.com/rqLanter/loan-approval-prediction.git 
cd loan-approval-prediction
pip install -r requirements.txt
python app.py
```

## Запуск в Google Colab
1. **Создайте новый ноутбук** в Colab
2. **Загрузите файлы**:
   ```python
   from google.colab import files
   files.upload()  # Загрузите code.py и loan_approval_dataset.csv
   ```
Запустите скрипт :
!python code.py
Не забудьте загрузить файл с обучающими данными.
Далее перейдите по ссылке указанной в терминале.

## Локальный запуск
Клонируйте репозиторий :
```
git clone https://github.com/rqLanter/loan-approval-prediction.git 
cd loan-approval-prediction
```
Установите зависимости :
```
pip install -r requirements.txt
```
Поместите данные :
Добавьте loan_approval_dataset.csv в папку data/ (или в папку проекта)

Запустите приложение :
python code.py

Откройте http://localhost:5000 (или ссылку от ngrok)



## Результаты работы программы
![image](https://github.com/user-attachments/assets/c40188a2-89f1-41e4-a2fb-e64d1523f998)
![image](https://github.com/user-attachments/assets/c3761ebb-5a77-41d4-992c-d1817d5cda01)
![image](https://github.com/user-attachments/assets/486ae085-c866-48c6-bb67-cdadfeaee545)
![image](https://github.com/user-attachments/assets/67cc1710-e1ab-4d09-8f8f-7709e8d6e013)

### 1. Форма заполнения заявки
![image](https://github.com/user-attachments/assets/0de7da24-eb66-46b4-9be8-6d30a6ee28b5)
### 2. Пример заполнения заявки (хороший заёмщик)
![image](https://github.com/user-attachments/assets/b0c644ce-d8e2-4f75-9efa-e5aabe512054)
### 3. Результат обработки заявки
Результат:![image](https://github.com/user-attachments/assets/c24fe0f0-3c8a-4ed0-bea9-c339daaae5fd)

