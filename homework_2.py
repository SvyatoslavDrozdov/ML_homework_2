import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')
print("Таблица с данными:")
creditcard = pd.read_csv('creditcard.csv')
print(creditcard.head())
print("-----------------------------------------------------------")
print("Основные статистики:")
print(creditcard.describe())
print("-----------------------------------------------------------")
print("""
Согласно данным kaggle:

v1-v28: - результаты PCA, исходные данные не представлены из-за проблем с конфиденциальностью.
Time - кол-во секунд между каждой транзакцией и первой транзакцией в датасэте.
Amount - сумма транзакции.
Class - переменная, принимающая значение 1 в случае мошенничества и 0 в случае честной транзакции.
""")

s = 0
for i in range(len(creditcard.columns)):
    s += 1
    if i % 4 == 0:
        if i != 0:
            plt.show()
            nothing = 1
        plt.figure(figsize=(12, 8))
        s = 1
    plt.subplot(2, 2, s)
    plt.hist(creditcard.iloc[:, i], bins=100)
    plt.xlabel(creditcard.columns[i])
    plt.ylabel('amount')
plt.show()
print("-----------------------------------------------------------")
print("Как видно из следующей таблицы, данные не сбалансированы")
print(creditcard['Class'].value_counts())
print("-----------------------------------------------------------")
print("Проверим наличие дубликатов, и удалим их, если они есть:")
duplicate_exist = creditcard.duplicated().any()
print(f"дубликаты есть: {duplicate_exist}")
if duplicate_exist:
    creditcard.drop_duplicates(inplace=True)
    print("...удаление дубликатов...")
duplicate_exist = creditcard.duplicated().any()
print(f"дубликаты есть: {duplicate_exist}")
print("-----------------------------------------------------------")
print(
    "Как видно из следующей таблицы, данные с удаленными дубликатами также не сбалансированы, чего и следовало ожидать, "
    "так как отсутствие сбалансированности связано с малым количеством мошеннических операций")
print(creditcard['Class'].value_counts())
print("-----------------------------------------------------------")

creditcard_data = creditcard.drop('Class', axis=1)  # таблица без столбца с ответами
creditcard_data = creditcard_data.drop('Time', axis=1)  # удаляем столбец с временем, т.к. оно не влияет на результат
creditcard_answer = creditcard['Class']  # ответы
rus = RandomUnderSampler()
creditcard_data_balanced, y_balanced = rus.fit_resample(creditcard_data, creditcard_answer)
standard_scaler = StandardScaler()

print("Сравним статистики обучающей и тестовой выборок:")
X_train, X_test, y_train, y_test = train_test_split(creditcard_data_balanced, y_balanced, test_size=0.25,
                                                    random_state=42)
X_train_not_balanced, X_test_not_balanced, y_train_not_balanced, y_test_not_balanced = train_test_split(
    creditcard_data, creditcard_answer, test_size=0.25, random_state=42)
print("X_train")
print(X_train.describe(), "\n")
print("X_test")
print(X_test.describe(), "\n")
print("y_train")
print(y_train.describe(), "\n")
print("y_test")
print(y_test.describe(), "\n")

print("X_train_not_balanced")
print(X_train_not_balanced.describe())
print("X_test_not_balanced")
print(X_test_not_balanced.describe())
print("y_train_not_balanced")
print(y_train_not_balanced.describe())
print("y_test_not_balanced")
print(y_test_not_balanced.describe())

creditcard_data_scaled = standard_scaler.fit_transform(creditcard_data_balanced)
creditcard_data_scaled_not_balanced = standard_scaler.fit_transform(creditcard_data)

X_train, X_test, y_train, y_test = train_test_split(creditcard_data_scaled, y_balanced, test_size=0.25, random_state=42)
X_train_not_balanced, X_test_not_balanced, y_train_not_balanced, y_test_not_balanced = train_test_split(
    creditcard_data_scaled_not_balanced, creditcard_answer, test_size=0.25, random_state=42)

print("-----------------------------------------------------------")
print("Укажем метрики, позволяющие оценить качество работы алгоритма:")
print("""
precision - доля положительных объектов, распознанных моделью как положительные.
recall - доля найденных объектов положительного класса из всех объектов положительного класса. 
f1 - среднее гармоническое описанных метрик.
""")

print("-----------------------------------------------------------")
print("Логистическая регрессия на сбалансированной выборке: \n")
log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(X_train, y_train)
y_predict = log_reg.predict(X_test)
print(classification_report(y_test, y_predict))

print("Логистическая регрессия на несбалансированной выборке: \n")
log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(X_train_not_balanced, y_train_not_balanced)
y_predict = log_reg.predict(X_test_not_balanced)
print(classification_report(y_test_not_balanced, y_predict))

print("-----------------------------------------------------------")
print("Метод k ближайших соседей на сбалансированной выборке: \n")
Kneib = KNeighborsClassifier(n_neighbors=3)
Kneib.fit(X_train, y_train)
y_predict = Kneib.predict(X_test)
print(classification_report(y_test, y_predict))

print("Метод k ближайших соседей на не сбалансированной выборке: \n")
Kneib = KNeighborsClassifier(n_neighbors=3)
Kneib.fit(X_train_not_balanced, y_train_not_balanced)
y_predict = Kneib.predict(X_test_not_balanced)
print(classification_report(y_test_not_balanced, y_predict))

print("-----------------------------------------------------------")
print("Метод опорных векторов на сбалансированной выборке: \n")
svc = LinearSVC(class_weight='balanced', dual=True, max_iter=20000)
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
print(classification_report(y_test, y_predict))

print("Метод опорных векторов на не сбалансированной выборке: \n")
svc = LinearSVC(dual=True, max_iter=100)  # что бы не ждать слишком долго
svc.fit(X_train_not_balanced, y_train_not_balanced)
y_predict = svc.predict(X_test_not_balanced)
print(classification_report(y_test_not_balanced, y_predict))

print("""
Вывод:
Как показали эксперименты, для получения более точных результатов лучше использовать сбалансированную выборку для обучения.
(Хотя разница результатов зависит от конкретного алгоритма, и, что так же важно, от конкретной метрики.)
Отметим, что несбалансированная выборка считается значительно дольше чем сбалансированная недосэмплированная, так 
как имеет в себе большее количество объектов. Эти факты показывают эффективность приема недосэмлирования, для 
решения этой задачи.

Среди используемых методов наиболее точными (по доле найденных объектов положительного класса из всех объектов 
положительного класса) оказались метод опорных векторов и метод логистической регрессии обученные на сбалансированной 
выборке. Их результаты приблизительно равны.
""")
