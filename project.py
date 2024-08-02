import csv
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Example data
data = {
    'equipment_id': [1122, 1123, 1112],
    'timestamp': [30, 25, 31],
    'sensor_reading': [1, 2, 3],
    'failure': [1, 1, 0]
}

df = pd.DataFrame(data)

# Features and target
X = df[['equipment_id', 'timestamp', 'sensor_reading']]
y = df['failure']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Classification report
report = classification_report(y_test, y_pred, zero_division=1)
print(report)

# Save data to CSV
csv_file = r'C:\Users\Kartik sharma\OneDrive\Desktop\equipment_data.csv'
data = [
    ["equipment_id", "timestamp", "sensor_reading", "failure"],
    [1122, 30, 1, 1],
    [1123, 25, 2, 1],
    [1112, 31, 3, 0]
]

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

# Load and process dataset
data = pd.read_csv(csv_file)
data.fillna(method='ffill', inplace=True)

# Normalize sensor data
scaler = StandardScaler()
data['sensor_reading'] = scaler.fit_transform(data[['sensor_reading']])

# Rolling average of sensor readings
data['sensor_reading_rolling_mean'] = data['sensor_reading'].rolling(window=2).mean()

# Drop rows with NaN values created by rolling mean
data.dropna(inplace=True)

# Define features and target
X = data.drop(['equipment_id', 'timestamp', 'failure'], axis=1)
y = data['failure']

# Split data for training and testing
if len(data) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, r'C:\Users\Kartik sharma\OneDrive\Desktop\predictive_maintenance_model.pkl')
    
    # Load and use the model
    model = joblib.load(r'C:\Users\Kartik sharma\OneDrive\Desktop\predictive_maintenance_model.pkl')
else:
    print("Not enough data to split into training and testing sets.")
