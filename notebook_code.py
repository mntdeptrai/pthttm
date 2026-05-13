import pandas as pd

sheet_id = '1ZfnNy48RwtK0D_i_B3CmoXUKcpIBt8rBfdBNqzypEko'
gid = '695684010'
data_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
try:
  df = pd.read_csv(data_url)
  print("Successfully loaded the data.")
  print(df.info())
  print(df.describe())
  print(df.isnull().sum())
except Exception as e:
  print(f"An error occurred while loading the data: {e}")
# Fill missing numerical values with the mean
for col in df.select_dtypes(include=['number']):
  if df[col].isnull().any():
    df[col] = df[col].fillna(df[col].mean())
df.info()
df.describe()
df.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns

# Create a heatmap with target as x-axis
plt.figure(figsize=(2, 10))
sns.heatmap(df.corr()[['target']].sort_values(by='target', ascending=False), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap with target as X-axis')
plt.show()

# Create a figure and subplots
num_cols = len(df.columns)
num_rows = (num_cols + 2) // 3  # Calculate rows needed to fit all plots
fig, axes = plt.subplots(num_rows, min(3, num_cols), figsize=(15, 5 * num_rows))

# Flatten the axes array if it's multi-dimensional
axes = axes.ravel()

# Iterate through columns and create histograms
for i, col in enumerate(df.columns):
    if i < len(axes):
      ax = axes[i]
      ax.hist(df[col].dropna(), bins=10)  # Drop NaN values for histogram
      ax.set_title(f"Distribution of {col}")
      ax.set_xlabel(col)
      ax.set_ylabel("Frequency")
    else:
        break

# Adjust layout and remove empty subplots
plt.tight_layout()
for i in range(num_cols, len(axes)):
    fig.delaxes(axes[i])
plt.show()
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Separate numerical and categorical features
numerical_features = df.select_dtypes(include=['number']).columns
# Normalization for numerical features
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])


print(df.head())
print(df.info())
print(df.describe())
plt.figure(figsize=(2, 10))
sns.heatmap(df.corr()[['target']].sort_values(by='target', ascending=False), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap with target as X-axis')
plt.show()
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=50)

# X_train, y_train: Training data
# X_val, y_val: Validation data
# X_test, y_test: Testing data

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Testing data shape:", X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Initialize classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(random_state=50),
    "KNN": KNeighborsClassifier(n_neighbors=1)
}

# Train and evaluate each classifier
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    results[name] = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred),
        "confusion_matrix": confusion_matrix(y_val, y_pred)
    }

# Comparison of Performances
print("Performance Comparison:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1_score']:.4f}")
    print("  Confusion Matrix:")
    print(metrics['confusion_matrix'])


# Visualization of Confusion Matrices
plt.figure(figsize=(15, 5))
for i, (name, metrics) in enumerate(results.items()):
    plt.subplot(1, 3, i + 1)
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
plt.tight_layout()
plt.show()


# Visualization of Accuracy
accuracy_scores = [results[name]['accuracy'] for name in results]
classifier_names = list(results.keys())

plt.figure(figsize=(8, 6))
plt.bar(classifier_names, accuracy_scores)
plt.xlabel("Classifiers")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Classifiers")
plt.ylim(0,1)
plt.show()
# Predict the test data
y_pred_knn = classifiers["KNN"].predict(X_test)

# Confusion Matrix for KNN on Test Data
cm_test_knn = confusion_matrix(y_test, y_pred_knn)

# Accuracy on test data
accuracy_test_knn = accuracy_score(y_test, y_pred_knn)

# Plotting the Confusion Matrix for Test Data
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test_knn, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - KNN (Test Data)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

print(f"KNN Test Accuracy: {accuracy_test_knn}")

# Since KNN doesn't have epochs, we can't plot accuracy vs epochs.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Reshape the data for 1D CNN
X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)


# Define the 1D CNN model
model = keras.Sequential([
    layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2, strides=1),
    layers.Flatten(),
    layers.Dense(units=100, activation='relu'),
    layers.Dense(units=1, activation='sigmoid') # binary classification
])
# Define early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,          # Stop after 5 epochs with no improvement
    restore_best_weights=True  # Restore the best model weights
)
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks = [early_stopping] )

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions on the test set
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# Calculate metrics
cm_test = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-score: {f1:.4f}")

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - 1D CNN (Test Data)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()