# 1. Установка зависимостей
!pip install pyngrok flask pandas numpy scikit-learn joblib

# 2. Импорты
import pandas as pd
import numpy as np
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
from flask import Flask, render_template, request
from pyngrok import ngrok
import matplotlib.pyplot as plt
import seaborn as sns

# 3. Загрузка данных
print("Загрузите файл loan_approval_dataset.csv")
uploaded = files.upload()

#считываем CSV без заголовков
df = pd.read_csv('loan_approval_dataset.csv', header=None)

#определяем названия колонок вручную
column_names = [
    'loan_id', 'no_of_dependents', 'education', 'self_employed',
    'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value', 'loan_status'
]

#применяем названия колонок
df.columns = column_names

#удаление первой строки, если она содержит заголовки
if df.iloc[0].astype(str).str.contains('loan_id').any():
    df = df.drop(0).reset_index(drop=True)

#очистка данных
df['education'] = df['education'].str.strip()
df['self_employed'] = df['self_employed'].str.strip()
df['loan_status'] = df['loan_status'].str.strip()

#кодирование целевой переменной
df['loan_status'] = df['loan_status'].replace({
    'Approved': 1,
    'Rejected': 0,
    'approved': 1,
    'rejected': 0,
    'loan_status': np.nan
})

#удаление строк с некорректными значениями
df = df[df['loan_status'].notna()]
df['loan_status'] = df['loan_status'].astype(int)


def plot_distributions(df):
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    #распределение классов
    plt.figure(figsize=(6, 4))
    sns.countplot(x='loan_status', data=df)
    plt.xticks([0, 1], ['Отказано', 'Одобрено'])
    plt.title('Баланс классов')
    plt.show()

    #числовые признаки
    important_numerical = ['income_annum', 'loan_amount', 'cibil_score', 'loan_term']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, col in zip(axes.flatten(), important_numerical):
        sns.histplot(df[col], ax=ax, kde=True)
        ax.set_title(f'Распределение {col}')
    plt.tight_layout()
    plt.show()

    #категориальные признаки
    categorical_features = ['education', 'self_employed']
    for col in categorical_features:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df, palette='viridis')
        plt.title(f'Распределение {col}')
        plt.xticks(rotation=45)
        plt.show()

# 3.1 вызов функции
plot_distributions(df)

#отделение признаков и целевой переменной
X = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status']

#принудительное преобразование числовых признаков
numerical_features = [
    'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
]

for col in numerical_features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

#разделение на числовые и категориальные признаки
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include='object').columns

#применение импьютинга
imputer_num = SimpleImputer(strategy='median')
imputer_cat = SimpleImputer(strategy='most_frequent')

X[num_cols] = imputer_num.fit_transform(X[num_cols])
X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])

#One-hot 
X_encoded = pd.get_dummies(X, columns=cat_cols)

#Масштабирование
scaler = StandardScaler()
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

#Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

#Обучение модели
param_grid = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [10, 20, 30],  # Было [2, 5, 10]
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)
best_tree = grid.best_estimator_

#сохранение препроцессоров и модели
joblib.dump(imputer_num, 'imputer_num.joblib')
joblib.dump(imputer_cat, 'imputer_cat.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(best_tree, 'best_decision_tree_model.joblib')
joblib.dump(X_encoded.columns, 'feature_names.joblib')


# 4. Создание Flask-приложения
app = Flask(__name__)


#функция для загрузки модели
def load_model():
    try:
        model = joblib.load('best_decision_tree_model.joblib')
        imputer_num = joblib.load('imputer_num.joblib')
        imputer_cat = joblib.load('imputer_cat.joblib')
        scaler = joblib.load('scaler.joblib')
        feature_names = joblib.load('feature_names.joblib')
        return model, imputer_num, imputer_cat, scaler, feature_names
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None, None, None, None, None

model, imputer_num, imputer_cat, scaler, feature_names = load_model()

@app.route('/')
def home():
    return render_template('index.html',
                         numerical_features=num_cols.tolist(),
                         categorical_features=cat_cols.tolist())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        df = pd.DataFrame([data])

        #обработка числовых признаков
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        #применение препроцессоров
        df[num_cols] = imputer_num.transform(df[num_cols])
        df[cat_cols] = imputer_cat.transform(df[cat_cols])

        #one-hot кодирование
        df_encoded = pd.get_dummies(df, columns=cat_cols)

        #выравнивание колонок
        df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)

        #масштабирование
        df_encoded[num_cols] = scaler.transform(df_encoded[num_cols])

        #предсказание
        prediction = model.predict(df_encoded)[0]
        proba = model.predict_proba(df_encoded)[0]

        probability = proba[1]
        return render_template('result.html',
                             prediction="Одобрено" if prediction == 1 else "Отказано",
                             probability=f"{probability*100:.2f}%")

    except Exception as e:
        return render_template('error.html', error=str(e))

# 5. Создание HTML-шаблонов
os.makedirs('templates', exist_ok=True)

# index.html
with open('templates/index.html', 'w') as f:
    f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Форма заявки на кредит</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap @5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            font-family: Arial, sans-serif;
        }
        .container {
            max-width: 600px;
            margin-top: 50px;
        }
        input[type="number"], select {
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        button {
            padding: 10px 20px;
            border-radius: 5px;
            background-color: #007bff;
            color: white;
            border: none;
        }
        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1 class="text-center mb-4">Форма подачи заявки на кредит</h1>
        <form action="/predict" method="post">
            <!-- Числовые поля -->
            {% for feature in numerical_features %}
            <div class="mb-3">
                <label class="form-label">
                    {% if feature == 'no_of_dependents' %}Количество иждивенцев{% endif %}
                    {% if feature == 'income_annum' %}Годовой доход (в рублях){% endif %}
                    {% if feature == 'loan_amount' %}Сумма кредита (в рублях){% endif %}
                    {% if feature == 'loan_term' %}Срок кредита (в месяцах){% endif %}
                    {% if feature == 'cibil_score' %}Кредитный рейтинг CIBIL (300–900){% endif %}
                    {% if feature == 'residential_assets_value' %}Стоимость жилой недвижимости (в рублях){% endif %}
                    {% if feature == 'commercial_assets_value' %}Стоимость коммерческой недвижимости (в рублях){% endif %}
                    {% if feature == 'luxury_assets_value' %}Стоимость предметов роскоши (в рублях){% endif %}
                    {% if feature == 'bank_asset_value' %}Стоимость банковских активов (в рублях){% endif %}
                </label>
                <input type="number" class="form-control" name="{{ feature }}" required>
            </div>
            {% endfor %}

            <!-- Категориальные поля -->
            {% for feature in categorical_features %}
            <div class="mb-3">
                <label class="form-label">
                    {% if feature == 'education' %}Образование{% endif %}
                    {% if feature == 'self_employed' %}Самозанятость{% endif %}
                </label>
                <select class="form-select" name="{{ feature }}" required>
                    <option value="">Выберите...</option>
                    {% if feature == 'education' %}
                        <option>Graduate</option>
                        <option>Not Graduate</option>
                    {% elif feature == 'self_employed' %}
                        <option>Yes</option>
                        <option>No</option>
                    {% endif %}
                </select>
            </div>
            {% endfor %}

            <button type="submit" class="btn btn-primary w-100">Отправить заявку</button>
        </form>
    </div>
</body>
</html>''')

# result.html
with open('templates/result.html', 'w') as f:
    f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Результат</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap @5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css " rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            font-family: Arial, sans-serif;
        }
        .card {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1 class="text-center mb-4">Результат рассмотрения заявки</h1>
        <div class="card mb-3">
            <div class="card-body">
                <h4 class="card-title">
                    {% if prediction == 'Одобрено' %}
                        <i class="fas fa-check-circle text-success"></i> {{ prediction }}
                    {% else %}
                        <i class="fas fa-times-circle text-danger"></i> {{ prediction }}
                    {% endif %}
                </h4>
                <p class="card-text">Вероятность: {{ probability }}</p>
            </div>
        </div>
        <a href="/" class="btn btn-primary mt-3 w-100">
          <i class="fas fa-redo"></i> Новая заявка
        </a>
    </div>
</body>
</html>''')



# 6. Запуск Flask с ngrok
# Установка токена
ngrok.set_auth_token("2woWGW6DVNHCocqUzZxiGFoRj7N_3Z9PvVFz92w2eAn9d4dVJ")

# Запуск Flask в фоне
import threading
threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000}).start()

# Проброс порта 
public_url = ngrok.connect(5000)
print(f"🔗 Ссылка для доступа: {public_url}")
