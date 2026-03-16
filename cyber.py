

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

df = pd.read_csv("cyberfeddefender_dataset.csv")
print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData info:")
df.info()
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe(include='all'))


cols_to_drop = ['Timestamp', 'Source_IP', 'Destination_IP']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"\nColumns after dropping identifiers: {df.columns.tolist()}")

le_protocol = LabelEncoder()
df['Protocol'] = le_protocol.fit_transform(df['Protocol'])


print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

plt.figure(figsize=(6,4))
sns.countplot(x='Label', data=df, palette='viridis')
plt.title('Distribution of Target Label (0 = Normal, 1 = Attack)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Protocol', data=df, palette='Set2')
plt.title('Distribution of Protocol Types')
plt.xlabel('Protocol (encoded)')
plt.ylabel('Count')
protocol_names = le_protocol.classes_
plt.xticks(ticks=range(len(protocol_names)), labels=protocol_names, rotation=45)
plt.show()

num_cols = ['Packet_Length', 'Duration', 'Bytes_Sent', 'Bytes_Received',
            'Flow_Packets/s', 'Flow_Bytes/s', 'Avg_Packet_Size']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()
for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()
for i, col in enumerate(num_cols):
    sns.boxplot(x='Label', y=col, data=df, ax=axes[i], palette='Set1')
    axes[i].set_title(f'{col} by Label')
plt.tight_layout()
plt.show()

num_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(20, 16))
sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


X = df.drop('Label', axis=1)
y = df['Label']

X = X.select_dtypes(include=[np.number])
print(f"\nFeatures after dropping non-numeric: {X.columns.tolist()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42, stratify=y
)
print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

scaler_mm = MinMaxScaler()
X_train_mm = scaler_mm.fit_transform(X_train)
X_test_mm = scaler_mm.transform(X_test)


models = {
    'Gaussian NB': GaussianNB(),
    'Multinomial NB': MultinomialNB(),
    'Bernoulli NB': BernoulliNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
plt.figure(figsize=(15, 12))
roc_curves = {}

for idx, (name, model) in enumerate(models.items()):
    print("\n" + "-" * 50)
    print(f"Training {name}...")

    if name == 'Multinomial NB':
        X_tr, X_te = X_train_mm, X_test_mm
    else:
        X_tr, X_te = X_train_std, X_test_std

    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.subplot(2, 3, idx+1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        roc_curves[name] = (fpr, tpr, roc_auc)

plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.show()

results_df = pd.DataFrame(results).T
print("\n" + "=" * 60)
print("MODEL COMPARISON TABLE")
print("=" * 60)
print(results_df)

plt.figure(figsize=(10, 6))
results_df.plot(kind='bar', rot=45, colormap='Set2')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("EXAMPLE PREDICTION (using Gaussian NB)")
print("=" * 60)
sample = X_test.iloc[0:1]
sample_scaled = scaler_std.transform(sample)
gnb = models['Gaussian NB']
pred_label = gnb.predict(sample_scaled)[0]
print(f"Sample features:\n{sample.to_dict(orient='records')[0]}")
print(f"Predicted Label: {pred_label}")
print(f"True Label: {y_test.iloc[0]}")
