import numpy as np   
import pandas as pd       
import matplotlib.pyplot as plt        
import seaborn as sns 
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 
import joblib

# Load the dataset
df = pd.read_csv('C:/Users/abdul/OneDrive/Desktop/ByteWise_ML/Project [Loan Default Rates]_4/Copy of loan.csv')
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Log transformation for LoanAmount
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])

# Filling missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['LoanAmount_log'].fillna(df['LoanAmount_log'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
print(df.isnull().sum())

# Visualizations
fig, axes = plt.subplots(3, 3, figsize=(18, 18))

# Gender countplot
sns.countplot(x='Gender', data=df, palette='Set2', ax=axes[0, 0])
axes[0, 0].set_title('Gender Count')

# Married countplot
sns.countplot(x='Married', data=df, palette='Set2', ax=axes[0, 1])
axes[0, 1].set_title('Married Count')

# Dependents countplot
sns.countplot(x='Dependents', data=df, palette='Set1', ax=axes[0, 2])
axes[0, 2].set_title('Dependents Count')

# Self_Employed countplot
sns.countplot(x='Self_Employed', data=df, palette='Set1', ax=axes[1, 0])
axes[1, 0].set_title('Self Employed Count')

# LoanAmount distribution
sns.histplot(df['LoanAmount'], bins=20, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Loan Amount Distribution')

# Credit History countplot
sns.countplot(x='Credit_History', data=df, palette='Set1', ax=axes[1, 2])
axes[1, 2].set_title('Credit History Count')

# LoanAmount_log distribution
sns.histplot(df['LoanAmount_log'], bins=20, kde=True, ax=axes[2, 0])
axes[2, 0].set_title('Loan Amount Log Distribution')

# TotalIncome_log distribution
sns.histplot(df['TotalIncome_log'], bins=20, kde=True, ax=axes[2, 1])
axes[2, 1].set_title('Total Income Log Distribution')

# Violin plot of LoanAmount by Gender
sns.violinplot(x='Gender', y='LoanAmount', data=df, palette='Set2', ax=axes[2, 2])
axes[2, 2].set_title('Loan Amount by Gender')

plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(df, hue='Loan_Status', palette='Set1')
plt.show()

# Violin plots
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

# Violin plot of LoanAmount by Married
sns.violinplot(x='Married', y='LoanAmount', data=df, palette='Set2', ax=axes[0, 0])
axes[0, 0].set_title('Loan Amount by Married Status')

# Violin plot of LoanAmount by Dependents
sns.violinplot(x='Dependents', y='LoanAmount', data=df, palette='Set1', ax=axes[0, 1])
axes[0, 1].set_title('Loan Amount by Dependents')

# Violin plot of LoanAmount by Self_Employed
sns.violinplot(x='Self_Employed', y='LoanAmount', data=df, palette='Set1', ax=axes[1, 0])
axes[1, 0].set_title('Loan Amount by Self Employed')

# Violin plot of LoanAmount by Credit_History
sns.violinplot(x='Credit_History', y='LoanAmount', data=df, palette='Set1', ax=axes[1, 1])
axes[1, 1].set_title('Loan Amount by Credit History')

plt.tight_layout()
plt.show()

# Select only numeric columns for the correlation matrix
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 14))

sns.histplot(df['LoanAmount_log'], bins=20, kde=True, ax=axes[0, 0], stat="density")
axes[0, 0].set_title('PDF of Loan Amount Log')
sns.ecdfplot(df['LoanAmount_log'], ax=axes[0, 1])
axes[0, 1].set_title('CDF of Loan Amount Log')

sns.histplot(df['TotalIncome_log'], bins=20, kde=True, ax=axes[1, 0], stat="density")
axes[1, 0].set_title('PDF of Total Income Log')
sns.ecdfplot(df['TotalIncome_log'], ax=axes[1, 1])
axes[1, 1].set_title('CDF of Total Income Log')

plt.tight_layout()
plt.show()

x = df.iloc[:, np.r_[1:5, 9:11, 13:15]].values
y = df.iloc[:, 12].values

print(x)
print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
LabelEncoder_x=LabelEncoder()

for i in range (0,5):
    X_train[:,i]=LabelEncoder_x.fit_transform(X_train[:,1])
    X_train[:,7]=LabelEncoder_x.fit_transform(X_train[:,7])
X_train

for i in range (0,5):
    X_test[:,i]=LabelEncoder_x.fit_transform(X_test[:,1])
    X_test[:,7]=LabelEncoder_x.fit_transform(X_test[:,7])
X_test

LabelEncoder_y=LabelEncoder()
y_train=LabelEncoder_y.fit_transform(y_train)
y_train

LabelEncoder_y=LabelEncoder()
y_test=LabelEncoder_y.fit_transform(y_test)
y_test


ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


rf_clf = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

print('Best parameters for Random Forest:', grid_search_rf.best_params_)
print('Best score for Random Forest:', grid_search_rf.best_score_)

nb_clf = GaussianNB()
param_grid_nb = {
    'var_smoothing': np.logspace(0, -9, num=100)
}
grid_search_nb = GridSearchCV(estimator=nb_clf, param_grid=param_grid_nb, cv=5, scoring='accuracy')
grid_search_nb.fit(X_train, y_train)

print('Best parameters for Naive Bayes:', grid_search_nb.best_params_)
print('Best score for Naive Bayes:', grid_search_nb.best_score_)

dt_clf = DecisionTreeClassifier()
param_grid_dt = {
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = GridSearchCV(estimator=dt_clf, param_grid=param_grid_dt, cv=5, scoring='accuracy')
grid_search_dt.fit(X_train, y_train)

print('Best parameters for Decision Tree:', grid_search_dt.best_params_)
print('Best score for Decision Tree:', grid_search_dt.best_score_)

# K-Neighbors Classifier with Hyperparameter Tuning
kn_clf = KNeighborsClassifier()
param_grid_kn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_search_kn = GridSearchCV(estimator=kn_clf, param_grid=param_grid_kn, cv=5, scoring='accuracy')
grid_search_kn.fit(X_train, y_train)

best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
ax.set_title('Random Forest Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.show()

fpr_rf, tpr_rf, _ = roc_curve(y_test, best_rf.predict_proba(X_test)[:, 1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc='lower right')
plt.show()

print('Best parameters for K-Neighbors:', grid_search_kn.best_params_)
print('Best score for K-Neighbors:', grid_search_kn.best_score_)

best_rf_clf = grid_search_rf.best_estimator_  
y_pred = best_rf_clf.predict(X_test)
print('Accuracy of Random Forest Classifier = ', metrics.accuracy_score(y_pred, y_test))

joblib.dump(best_rf_clf, 'loan_default_rf_model.pkl')
