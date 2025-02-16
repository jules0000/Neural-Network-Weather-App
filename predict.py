import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from datetime import datetime

# Replace with your active WeatherAPI.com API key
API_KEY = 'e178fd28eedf49f6bab70205251602'
# Request forecast for 3 days
url = f'http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q=Manila&days=3'
response = requests.get(url)
data = response.json()

# Check for API errors or unexpected structure
if "error" in data:
    print("API Error:", data["error"])
    exit()
if 'forecast' not in data or 'forecastday' not in data['forecast']:
    print("Unexpected data format:", data)
    exit()

today = datetime.today().date()

# Define the numeric features to extract
FEATURES = ['temp_c', 'humidity', 'pressure_mb', 'wind_kph', 'precip_mm']

weather_data = []
conditions = []   # Weather condition (e.g., "Sunny", "Rainy")
dates = []        # Date string from the API response
day_labels = []   # Human-friendly label ("Today", "Tomorrow", etc.)

# Loop through each forecast day
for forecast_day in data['forecast']['forecastday']:
    date_str = forecast_day.get('date', '')
    try:
        forecast_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception as e:
        forecast_date = None

    # Compute a human-friendly label based on the forecast date
    if forecast_date:
        diff = (forecast_date - today).days
        if diff == 0:
            label = "Today"
        elif diff == 1:
            label = "Tomorrow"
        elif diff == 2:
            label = "Day After Tomorrow"
        else:
            label = forecast_date.strftime("%A")  # e.g., "Monday"
    else:
        label = "Unknown"

    # Process each hourly forecast in this day
    for hour in forecast_day.get('hour', []):
        # Extract numeric features (insert np.nan if a field is missing)
        row = [hour.get(field, np.nan) for field in FEATURES]
        weather_data.append(row)
        # Extract the weather condition text from the nested 'condition' object
        condition_text = hour.get('condition', {}).get('text', 'Unknown')
        conditions.append(condition_text)
        dates.append(date_str)
        day_labels.append(label)

# Create a DataFrame with all the data
df = pd.DataFrame(weather_data, columns=FEATURES)
df['condition'] = conditions
df['date'] = dates
df['day_label'] = day_labels

# Optional: check for missing values
if df.isnull().any().any():
    print("Warning: Some fields are missing in the API response:")
    print(df.head())

# Define features and target
X = df[FEATURES].values
y = df['condition'].values
extra_info = df[['date', 'day_label']].values

# Encode the categorical weather condition target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Scale numeric features to the range [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# --- Create sequences to capture temporal context ---
# For example, using a window of 3 consecutive hours.
window_size = 3

def create_sequences(X, y, extra, window_size):
    X_seq = []
    y_seq = []
    extra_seq = []
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i : i + window_size])
        # Use the condition of the last hour in the window as the target.
        y_seq.append(y[i + window_size - 1])
        extra_seq.append(extra[i + window_size - 1])
    return np.array(X_seq), np.array(y_seq), np.array(extra_seq)

X_sequences, y_sequences, extra_sequences = create_sequences(X_scaled, y_cat, extra_info, window_size)

# Split the sequences into training and testing sets
X_train, X_test, y_train, y_test, extra_train, extra_test = train_test_split(
    X_sequences, y_sequences, extra_sequences, test_size=0.2, random_state=42
)

# --- Build the LSTM Classifier ---
num_classes = y_cat.shape[1]
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(window_size, len(FEATURES))),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'\nModel Loss: {loss}, Accuracy: {accuracy}')

# Make predictions on the test set
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

predicted_labels = le.inverse_transform(predicted_classes)
true_labels = le.inverse_transform(true_classes)

# Create a DataFrame of predictions with extra information
df_preds = pd.DataFrame({
    'date': extra_test[:, 0],
    'day_label': extra_test[:, 1],
    'predicted': predicted_labels,
    'true': true_labels
})

# Aggregate predictions by date using the mode (most frequent value)
aggregated = df_preds.groupby(['date', 'day_label']).agg(lambda x: x.value_counts().index[0]).reset_index()

# Sort by date (if needed)
aggregated = aggregated.sort_values('date')

print("\nAggregated Predictions on test samples (one per day):")
for _, row in aggregated.iterrows():
    print(f"Date: {row['date']}, Day: {row['day_label']}, Predicted: {row['predicted']}, True: {row['true']}")
