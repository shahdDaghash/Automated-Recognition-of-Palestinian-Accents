import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Define the data paths
train_data = {
    'Jerusalem': r'Training_Data\Jerusalem',
    'Nablus': r'Training_Data\Nablus',
    'Hebron': r'Training_Data\Hebron',
    'Ramallah': r'Training_Data\Ramallah_Reef'
}

test_data = {
    'Jerusalem': r'Testing_Data\Jerusalem',
    'Nablus': r'Testing_Data\Nablus',
    'Hebron': r'Testing_Data\Hebron',
    'Ramallah': r'Testing_Data\Ramallah_Reef'
}


# Function to extract features from audio files
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20) # 8192 (samples) / rate 44.1kHz (samples/sec) = 185.8 sec  , n_fft=8192, hop_length=2048
    return np.mean(mfccs, axis=1)


# Function to load data
def load_data(data_paths):
    X, y = [], []
    for label, path in data_paths.items():
        for file_name in os.listdir(path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(path, file_name)
                features = extract_features(file_path)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)


# Load training and testing data
X_train, y_train = load_data(train_data)
X_test, y_test = load_data(test_data)

# Create and train the model
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Print original and predicted values
def print_test_results(y_test, y_pred):
    print("Original vs Predicted values:")
    for original, predicted in zip(y_test, y_pred):
        print(f"Original: {original}, Predicted: {predicted}")

# Generate detailed report
def generate_report(y_test, y_pred):
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()

    # # Plot classification report heatmap
    # plt.figure(figsize=(10, 6))
    # sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap='Blues')
    # plt.title("Classification Report")
    # plt.show()

    # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    # plt.title("Confusion Matrix")
    # plt.show()

    # Print accuracy and classification report
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Generate and display the report
generate_report(y_test, y_pred)

print_test_results(y_test, y_pred)
