import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка
df = pd.read_csv('data.csv')

# 2. Очистка: Убираем запятые и превращаем в числа
# Это критический момент, 007. Без этого математика не сработает.
df['Number of Deaths'] = df['Number of Deaths'].str.replace(',', '').astype(float)
df['Death Rate Per 100,000'] = df['Death Rate Per 100,000'].str.replace(',', '').astype(float)

print("Данные очищены. Теперь это числа!")

# 3. Анализ: Топ-10 стран по количеству смертей за последний доступный год
latest_year = df['Year'].max()
top_countries = df[df['Year'] == latest_year].groupby('Country Name')['Number of Deaths'].sum().sort_values(ascending=False).head(10)

print(f"\nТоп-10 стран по смертности в {latest_year} году:")
print(top_countries)

# 4. Визуализация
plt.figure(figsize=(12, 6))
top_countries.plot(kind='bar', color='teal')
plt.title(f'Top 10 Countries by Number of Deaths in {latest_year}')
plt.xlabel('Country')
plt.ylabel('Total Deaths')
plt.xticks(rotation=45)
plt.tight_layout()

# Сохраняем график для GitHub
plt.savefig('top_deaths.png')
print("\nГрафик сохранен как top_deaths.png")

plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Подготовка данных для модели
# Мы возьмем Year как признак (X) и Death Rate как цель (y)
# Для простоты выберем одну страну, например, ту, что была первой в топе
example_country = top_countries.index[0]
country_data = df[df['Country Name'] == example_country].copy()

# Удаляем пропуски, если они есть
country_data = country_data.dropna(subset=['Year', 'Death Rate Per 100,000'])

X = country_data[['Year']]
y = country_data['Death Rate Per 100,000']

# Разбиваем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

print(f"\n--- Анализ для {example_country} ---")
print(f"Коэффициент детерминации (R^2): {r2_score(y_test, y_pred):.2f}")