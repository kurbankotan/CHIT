# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import display
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ─── Veri yükleme & temizlik ─────────────────────────────────────────────────
df_raw = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/datasets/kidney_disease/kidney_disease.csv").set_index('id')

num_cols = ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]
cat_cols = ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane","classification"]

df_raw[cat_cols]         = df_raw[cat_cols].apply(lambda c: c.str.strip() if c.dtype == object else c)
df_raw['classification'] = df_raw['classification'].map({'ckd': 1, 'notckd': 0})
for col in ['pcv', 'wc', 'rc']:
    df_raw[col] = pd.to_numeric(df_raw[col].str.strip().replace('?', np.nan), errors='coerce')

# ─── Kategorik sütunlar: sütun tabanlı imputation + encoding ─────────────────
df_raw[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df_raw[cat_cols])
df_raw[cat_cols] = df_raw[cat_cols].apply(LabelEncoder().fit_transform)

# ─── CHIT: Sütun tabanlı geçici dolgu yöntemleri ─────────────────────────────
# Mean | Median | Mode | Forward Fill | Backward Fill | KNN | Interpolation

def col_temp_fill(row, J_nan, Y, num_cols, strategy):
    """i* satırındaki eksik değerleri seçilen sütun tabanlı yöntemle GEÇİCİ doldurur."""
    row = row.copy()

    if strategy == 'mean':
        for col in J_nan:
            row[col] = Y[col].mean()

    elif strategy == 'median':
        for col in J_nan:
            row[col] = Y[col].median()

    elif strategy == 'mode':
        for col in J_nan:
            row[col] = Y[col].mode().iloc[0]

    elif strategy == 'ffill':
        for col in J_nan:
            known = Y[col].dropna()
            row[col] = known.iloc[-1] if len(known) > 0 else Y[col].mean()

    elif strategy == 'bfill':
        for col in J_nan:
            known = Y[col].dropna()
            row[col] = known.iloc[0] if len(known) > 0 else Y[col].mean()

    elif strategy == 'knn':
        temp    = pd.concat([Y[num_cols], pd.DataFrame([row[num_cols]])])
        imputed = KNNImputer(n_neighbors=5).fit_transform(temp)
        for i, col in enumerate(num_cols):
            if col in J_nan:
                row[col] = imputed[-1, i]

    elif strategy == 'interpolation':
        for col in J_nan:
            s = pd.concat([Y[col], pd.Series([np.nan], index=[row.name])])
            row[col] = s.interpolate(method='linear').iloc[-1]

    return row


# ─── CHIT Algoritması (Kotan & Kırışoğlu, Sci Rep 2025) ──────────────────────
def chit_impute(df_in, num_cols, row_model, col_strategy):
    """
    Cyclical Hybrid Imputation Technique — sayısal sütunlar için.

    Parametreler:
      row_model    : Satır tabanlı regresyon modeli
      col_strategy : 'mean' | 'median' | 'mode' | 'ffill' | 'bfill' | 'knn' | 'interpolation'
    """
    df          = df_in.copy()
    feature_cols = [c for c in df.columns if c != 'classification']
    col_missing  = df[num_cols].isna().sum()       # j* sıralaması için sabit

    has_missing = df[num_cols].isna().any(axis=1)
    Y = df[~has_missing].copy()
    X = df[has_missing].copy()

    while len(X) > 0:
        i_star = X[num_cols].isna().sum(axis=1).idxmin()
        row    = X.loc[i_star].copy()
        J_nan  = [c for c in num_cols if pd.isna(row[c])]

        # Adım 3: Geçici sütun tabanlı dolgu
        row = col_temp_fill(row, J_nan, Y, num_cols, col_strategy)

        # Adım 4: Kalıcı satır tabanlı dolgu (j* önceliğiyle)
        for j_star in sorted(J_nan, key=lambda c: col_missing[c], reverse=True):
            X_feat = [c for c in feature_cols if c != j_star]
            mdl    = clone(row_model)
            mdl.fit(Y[X_feat], Y[j_star])
            row[j_star] = mdl.predict(row[X_feat].values.reshape(1, -1))[0]

        Y = pd.concat([Y, pd.DataFrame([row], index=[i_star])])
        X = X.drop(index=i_star)

    return Y


# ─── Sınıflandırıcı modeller & parametre ızgaraları ──────────────────────────
CLASSIFIERS = {
    'KNN': (KNeighborsClassifier(), {
        'n_neighbors': range(1, 11), 'weights': ['uniform', 'distance'], 'leaf_size': [10, 20, 30, 40, 50]}),
    'LR' : (LogisticRegression(),   {
        'C': np.logspace(-4, 4, 50), 'penalty': ['l1', 'l2'], 'random_state': range(6)}),
    'SVC': (SVC(),                  {
        'C': [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}),
    'DT' : (DecisionTreeClassifier(), {
        'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8, 10, 12]}),
    'RFC': (RandomForestClassifier(), {
        'bootstrap': [True, False], 'max_depth': [10, 20, None], 'max_features': ['sqrt'],
        'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10], 'n_estimators': [10, 100]}),
    'GNB': (GaussianNB(),           {'var_smoothing': np.logspace(0, -9, 100)}),
    'MLP': (MLPClassifier(),        {
        'solver': ['lbfgs', 'sgd', 'adam'], 'alpha': [1e-5], 'activation': ['relu', 'logistic', 'tanh']}),
}

def build_nn(input_dim):
    nn = Sequential([
        Dense(1000, input_dim=input_dim, activation='relu'),
        Dense(500,  activation='relu'),
        Dense(300,  activation='relu'),
        Dropout(0.2),
        Dense(2,    activation='softmax'),
    ])
    nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return nn

def run_classifiers(X_tr, X_te, y_tr, y_te, X_all, y_all, show_plots=False):
    """Tüm sınıflandırıcıları çalıştırır, sonuçları döndürür."""
    results = {}
    to_cat  = lambda v: tf.keras.utils.to_categorical(v, 2)

    for name, (model, params) in CLASSIFIERS.items():
        best = GridSearchCV(model, params, cv=2, scoring='accuracy').fit(X_tr, y_tr).best_estimator_
        pred = best.fit(X_tr, y_tr).predict(X_te)
        cv   = cross_val_score(best, X_all, y_all, cv=5)
        cr   = classification_report(y_te, pred)
        print(f"  {name:<5} | CV: {cv.mean():.4f} | Acc: {accuracy_score(y_te, pred):.4f}")
        if show_plots:
            plt.figure(figsize=(4, 4))
            sns.heatmap(confusion_matrix(y_te, pred), annot=True, fmt='d', cmap='Blues')
            plt.title(name); plt.xlabel('Tahmin'); plt.ylabel('Gerçek')
            plt.tight_layout(); plt.show()
        results[name] = cr

    # Keras Sinir Ağı
    nn = build_nn(X_tr.shape[1])
    nn.fit(X_tr, to_cat(y_tr), validation_data=(X_te, to_cat(y_te)),
           batch_size=20, epochs=10, verbose=0)
    pred_NN = np.argmax(nn.predict(X_te), axis=1)
    cr_nn   = classification_report(y_te, pred_NN)
    print(f"  {'NN':<5} | Acc: {accuracy_score(y_te, pred_NN):.4f}")
    if show_plots:
        plt.figure(figsize=(4, 4))
        sns.heatmap(confusion_matrix(y_te, pred_NN), annot=True, fmt='d', cmap='Blues')
        plt.title('NN'); plt.xlabel('Tahmin'); plt.ylabel('Gerçek')
        plt.tight_layout(); plt.show()
    results['NN'] = cr_nn

    return results


# ─── Tüm sütun tabanlı yöntemler için CHIT + sınıflandırma döngüsü ───────────
# Satır tabanlı modeli buradan değiştirin:
#   KNeighborsRegressor | LinearRegression | SVR | DecisionTreeRegressor |
#   RandomForestRegressor | HistGradientBoostingRegressor
ROW_MODEL = KNeighborsRegressor(n_neighbors=5)

COL_STRATEGIES = ['mean', 'median', 'mode', 'ffill', 'bfill', 'knn', 'interpolation']

all_results = {}  # {col_strategy: {classifier: classification_report}}

for strategy in COL_STRATEGIES:
    print(f"\n{'='*60}")
    print(f"  Sütun Yöntemi: {strategy.upper()}  |  Satır Modeli: {ROW_MODEL.__class__.__name__}")
    print(f"{'='*60}")

    df_imp  = chit_impute(df_raw, num_cols, row_model=ROW_MODEL, col_strategy=strategy)
    X_data  = pd.DataFrame(normalize(df_imp.iloc[:, :-1], axis=0))
    y_data  = df_imp.iloc[:, -1]
    X_tr, X_te, y_tr, y_te = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    all_results[strategy] = run_classifiers(X_tr, X_te, y_tr, y_te, X_data, y_data,
                                            show_plots=False)  # True yapınca karışıklık matrisleri görünür


# ─── Korelasyon ısı haritası (son imputation sonrası) ────────────────────────
plt.figure(figsize=(15, 10))
sns.heatmap(df_imp.corr(), xticklabels=df_imp.columns, yticklabels=df_imp.columns, annot=True)
plt.title('Korelasyon Matrisi'); plt.show()


# ─── Genel karşılaştırma tablosu ─────────────────────────────────────────────
def parse_cr(cr):
    s = cr.split()
    return {'Accuracy': float(s[15]), 'Precision': float(s[25]),
            'Recall': float(s[26]), 'F1-Score': float(s[27])}

rows = []
for strategy, clf_results in all_results.items():
    for clf, cr in clf_results.items():
        row = parse_cr(cr)
        row['Col_Strategy'] = strategy
        row['Classifier']   = clf
        rows.append(row)

df_all = (pd.DataFrame(rows)
            .set_index(['Col_Strategy', 'Classifier'])
            .sort_values(['F1-Score', 'Accuracy'], ascending=False))

display(df_all)

# Her sütun yöntemi için en iyi sınıflandırıcıyı göster
print("\n─── Her yöntemin en iyi sonucu ───")
display(df_all.groupby('Col_Strategy')[['Accuracy','F1-Score']].max()
              .sort_values('F1-Score', ascending=False))
