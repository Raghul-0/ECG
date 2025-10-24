import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the extracted features
DATA_PATH = "features/ravdess_features.csv"
df = pd.read_csv(DATA_PATH)

print("🔍 Dataset loaded successfully!")
print(f"Total samples: {len(df)}")
print("Emotion labels:", df['label'].unique())

# Split features and labels
X = df.drop(columns=['label']).values
y = df['label'].values

# Encode emotion labels (convert text to numbers)
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Train the Random Forest classifier
clf = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Emotion Detection - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Save model and encoders
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/emotion_rf.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n💾 Models saved successfully in 'models/' folder.")
