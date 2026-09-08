# import zipfile, pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# # загрузка
# with zipfile.ZipFile('metaverse_transactions_dataset.csv.zip') as z:
#     with z.open('metaverse_transactions_dataset.csv') as f:
#         df = pd.read_csv(f)

# # исключаем строковые идентификаторы
# X = df.drop(columns=['anomaly', 'timestamp', 'sending_address', 'receiving_address'])
# y = df['anomaly']

# cat_cols = ['transaction_type','location_region','purchase_pattern','age_group']
# num_cols = [c for c in X.columns if c not in cat_cols]

# preprocessor = ColumnTransformer([
#     ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
#     ('num', 'passthrough', num_cols)
# ])

# model = Pipeline([
#     ('prep', preprocessor),
#     ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
# ])

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y)

# model.fit(X_train, y_train)
# preds = model.predict(X_test)
# print(classification_report(y_test, preds))

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# загрузка CSV-файла
df = pd.read_csv('metaverse_transactions_dataset.csv')

print("Данные загружены:", df.shape)
print("Столбцы:", df.columns)
print("\nРаспределение классов:")
print(df['anomaly'].value_counts())

# признаки и целевая переменная
X = df.drop(columns=['anomaly', 'timestamp', 'sending_address', 'receiving_address'])
y = df['anomaly']

cat_cols = ['transaction_type', 'location_region', 'purchase_pattern', 'age_group']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

model = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\nРезультаты модели:")
print(classification_report(y_test, preds))

from sklearn.tree import DecisionTreeClassifier

print("\n" + "=" * 60)
print("ДОПОЛНИТЕЛЬНЫЙ ЭКСПЕРИМЕНТ")
print("Decision Tree без признака risk_score")

# убираем risk_score, чтобы модель не использовала готовую оценку риска
X2 = df.drop(columns=[
    'anomaly',
    'timestamp',
    'sending_address',
    'receiving_address',
    'risk_score'
])

y2 = df['anomaly']

cat_cols2 = ['transaction_type', 'location_region', 'purchase_pattern', 'age_group']
num_cols2 = [c for c in X2.columns if c not in cat_cols2]

preprocessor2 = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols2),
    ('num', 'passthrough', num_cols2)
])

model2 = Pipeline([
    ('prep', preprocessor2),
    ('clf', DecisionTreeClassifier(max_depth=5, random_state=42))
])

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2,
    test_size=0.2,
    random_state=42,
    stratify=y2
)

model2.fit(X2_train, y2_train)

preds2 = model2.predict(X2_test)

print("\nРезультаты Decision Tree без risk_score:")
print(classification_report(y2_test, preds2))