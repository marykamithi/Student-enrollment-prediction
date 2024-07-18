import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Create directories for saving models and plots
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Load the dataset
data = pd.read_csv('data/student_data.csv')
df = pd.DataFrame(data)

# Define feature columns and target columns
feature_cols = ['GPA', 'test_scores', 'age', 'gender', 'socioeconomic_status', 'extracurriculars', 'hours_studied_per_week', 'part_time_job']
target_col_enroll = 'enrolled'
target_col_grad = 'graduated'

# Split data into features and target
X = df[feature_cols]
y_enroll = df[target_col_enroll]
y_grad = df[target_col_grad]

# Split data into training and testing sets
X_train, X_test, y_train_enroll, y_test_enroll = train_test_split(X, y_enroll, test_size=0.2, random_state=42)
_, _, y_train_grad, y_test_grad = train_test_split(X, y_grad, test_size=0.2, random_state=42)

# Preprocessing for numerical data
numerical_features = ['GPA', 'test_scores', 'age', 'hours_studied_per_week']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_features = ['gender', 'socioeconomic_status', 'extracurriculars', 'part_time_job']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the pipeline for modeling
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define parameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train_enroll)

# Best parameters from GridSearchCV
print("Best parameters for enrollment prediction: ", grid_search.best_params_)

# Evaluate the best model
best_model_enroll = grid_search.best_estimator_
y_pred_enroll = best_model_enroll.predict(X_test)

# Evaluate the model
print("Enrollment Prediction Report")
print(classification_report(y_test_enroll, y_pred_enroll))

# Confusion matrix for enrollment prediction
conf_matrix_enroll = confusion_matrix(y_test_enroll, y_pred_enroll)
sns.heatmap(conf_matrix_enroll, annot=True, fmt='d', cmap='Blues')
plt.title('Enrollment Prediction Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('plots/enrollment_confusion_matrix.png')
plt.show()

# Feature importance for enrollment prediction
importances_enroll = best_model_enroll.named_steps['classifier'].feature_importances_
feature_names_enroll = numerical_features + list(best_model_enroll.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
feature_importances_enroll = pd.Series(importances_enroll, index=feature_names_enroll).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 8))
feature_importances_enroll.plot(kind='bar')
plt.title('Feature Importances for Enrollment Prediction')
plt.savefig('plots/enrollment_feature_importances.png')
plt.show()

# Train the model for graduation prediction with cross-validation
cv_scores = cross_val_score(best_model_enroll, X, y_grad, cv=5)
print(f'Cross-Validation Scores for Graduation Prediction: {cv_scores}')
print(f'Mean Cross-Validation Score for Graduation Prediction: {cv_scores.mean()}')

best_model_enroll.fit(X_train, y_train_grad)
y_pred_grad = best_model_enroll.predict(X_test)

# Evaluate the model
print("Graduation Prediction Report")
print(classification_report(y_test_grad, y_pred_grad))

# Confusion matrix for graduation prediction
conf_matrix_grad = confusion_matrix(y_test_grad, y_pred_grad)
sns.heatmap(conf_matrix_grad, annot=True, fmt='d', cmap='Blues')
plt.title('Graduation Prediction Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('plots/graduation_confusion_matrix.png')
plt.show()

# Feature importance for graduation prediction
importances_grad = best_model_enroll.named_steps['classifier'].feature_importances_
feature_importances_grad = pd.Series(importances_grad, index=feature_names_enroll).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 8))
feature_importances_grad.plot(kind='bar')
plt.title('Feature Importances for Graduation Prediction')
plt.savefig('plots/graduation_feature_importances.png')
plt.show()

# Save the best model for future use
joblib.dump(best_model_enroll, 'models/best_model_enroll.pkl')
joblib.dump(best_model_enroll, 'models/best_model_grad.pkl')
print("Models have been saved as 'models/best_model_enroll.pkl' and 'models/best_model_grad.pkl'.")
