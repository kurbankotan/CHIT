import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import seaborn as sns

# Read dataset
raw_df = pd.read_csv("kidney_disease.csv")
raw_df.set_index("id", inplace=True)

# Column groups
num_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
cat_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "classification"]

# Clean categorical column values
raw_df.replace({
    'dm': {'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'},
    'cad': {'\tno': 'no'},
    'classification': {'ckd\t': 'ckd'},
    'pcv': {'\t?': np.nan, '\t43': 43},
    'wc': {'\t?': np.nan, '\t6200': 6200, '\t8400': 8400},
    'rc': {'\t?': np.nan, '\t3.9': 3.9, '\t4.0': 4.0, '\t5.2': 5.2}
}, inplace=True)

# Fix classification column with proper downcasting (no warning)
raw_df['classification'] = raw_df['classification'].replace({'ckd': 1, 'notckd': 0})
raw_df['classification'] = pd.to_numeric(raw_df['classification'], downcast='integer', errors='coerce')

# Ask user for column and row imputation preferences (in English, with numbers)
print("\n--- Imputation Options ---")
print("Select column-wise method:")
print("1. mean\n2. median\n3. mode\n4. knn\n5. interpolate\n6. ffill")
col_method_options = ["mean", "median", "mode", "knn", "interpolate", "ffill"]
col_choice = int(input("Enter number for column-wise imputation: ").strip()) - 1
col_method = col_method_options[col_choice]

print("\nSelect row-wise regression model:")
print("1. knn\n2. lr\n3. svr\n4. tree\n5. forest\n6. hgb\n7. nn")
row_model_options = ["knn", "lr", "svr", "tree", "forest", "hgb", "nn"]
row_choice = int(input("Enter number for row-wise regression model: ").strip()) - 1
row_model_name = row_model_options[row_choice]

# Convert object columns to numeric where possible
raw_df = raw_df.infer_objects(copy=False)
encoder = LabelEncoder()
raw_df[cat_cols] = raw_df[cat_cols].apply(encoder.fit_transform)

# Convert numeric columns
for col in ['pcv', 'wc', 'rc']:
    raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')

# Normalize data
X_full = raw_df.dropna()
X_missing = raw_df[raw_df.isnull().any(axis=1)]

y_full = X_full['classification']
y_missing = X_missing['classification']

scaler = StandardScaler()
X_full_scaled = pd.DataFrame(scaler.fit_transform(X_full.drop(columns='classification')), columns=X_full.columns[:-1], index=X_full.index)
X_full_scaled['classification'] = y_full

X_missing_scaled = pd.DataFrame(scaler.transform(X_missing.drop(columns='classification')), columns=X_missing.columns[:-1], index=X_missing.index)
X_missing_scaled['classification'] = y_missing

# Column-based imputation
def column_imputation(df, ref_df, method):
    if method == "mean":
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(ref_df[col].mean())
    elif method == "median":
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(ref_df[col].median())
    elif method == "mode":
        for col in df.columns:
            if df[col].isnull().any():
                mode = ref_df[col].mode()[0]
                df[col] = df[col].fillna(mode)
    elif method == "knn":
        knn_imp = KNNImputer(n_neighbors=1)
        df[:] = knn_imp.fit_transform(pd.concat([ref_df, df]))[-df.shape[0]:]
    elif method == "interpolate":
        df[:] = pd.concat([ref_df, df]).interpolate().iloc[-df.shape[0]:]
    elif method == "ffill":
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
    return df

# Row-based model selector
def get_regression_model(name, input_dim):
    models = {
        'knn': KNeighborsRegressor(),
        'lr': LinearRegression(),
        'svr': SVR(),
        'tree': DecisionTreeRegressor(),
        'forest': RandomForestRegressor(),
        'hgb': HistGradientBoostingRegressor(),
        'nn': Sequential([Input(shape=(input_dim,)), Dense(128, activation='relu'), Dense(64, activation='relu'), Dense(1)])
    }
    model = models[name]
    if name == 'nn':
        model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Imputation process - CHIT Algorithm
def CHIT_imputation(X_complete, X_incomplete, col_method='mean', row_model_name='forest'):
    while not X_incomplete.empty:
        min_nan_idx = X_incomplete.isnull().sum(axis=1).idxmin()
        row = X_incomplete.loc[[min_nan_idx]].copy()
        row = column_imputation(row, X_complete, col_method)

        for col in row.columns[row.isnull().any(axis=0)]:
            X_train = X_complete.drop(columns=[col])
            y_train = X_complete[col]
            X_test = row.drop(columns=[col])

            model = get_regression_model(row_model_name, X_train.shape[1])
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)
                if row_model_name == 'nn':
                    prediction = prediction[0][0]
                row.at[min_nan_idx, col] = prediction

        if not row.isnull().any(axis=1).values[0]:
            X_complete = pd.concat([X_complete, row])
            X_incomplete = X_incomplete.drop(index=min_nan_idx)

    return X_complete

# Example usage
filled_data = CHIT_imputation(X_full_scaled.copy(), X_missing_scaled.copy(), col_method=col_method, row_model_name=row_model_name)

# Classification model evaluations
def evaluate_models(df):
    X = df.drop(columns='classification')
    y = df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gaussian NB": GaussianNB(),
        "Logistic Regression": LogisticRegression(),
        "MLP": MLPClassifier(max_iter=500),
        "SVC": SVC()
    }

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{name} Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

# Run evaluation on imputed dataset
evaluate_models(filled_data)