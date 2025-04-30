import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Пример данных: рост, вес, размер обуви, пол
data = [
    [181, 80, 44, "male"],
    [177, 70, 43, "male"],
    [160, 60, 38, "female"],
    [154, 54, 37, "female"],
    [166, 65, 40, "female"],
    [190, 90, 47, "male"],
    [175, 64, 39, "male"],
    [177, 70, 40, "female"],
    [159, 55, 37, "female"],
    [171, 75, 42, "male"],
]

# Создаем DataFrame
df = pd.DataFrame(data, columns=["height", "weight", "shoe_size", "gender"])

# Признаки и целевая переменная
X = df[["height", "weight", "shoe_size"]]
y = df["gender"]

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Обучаем дерево решений
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Предсказываем пол
y_pred = model.predict(X_test)

# Оцениваем точность
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")

# Визуализируем дерево
plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
)
plt.title("Дерево решений: Определение пола по признакам")
plt.show()
