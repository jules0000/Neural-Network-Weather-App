````markdown
# Weather Forecast LSTM Classifier

This project demonstrates how to build an LSTM-based deep learning model to classify weather conditions using forecast data retrieved from WeatherAPI.com. The script fetches a 3-day weather forecast for Manila, processes hourly data into sequences, and trains an LSTM model to predict weather conditions (e.g., Sunny, Rainy, etc.) based on several numerical features.

## Overview

- **Data Acquisition:** Retrieves weather forecast data (temperature, humidity, pressure, wind speed, precipitation) from WeatherAPI.com.
- **Data Processing:** Extracts numeric features and weather conditions from the API response, handles missing data, scales the features using MinMaxScaler, and encodes the categorical weather conditions.
- **Sequence Creation:** Forms time sequences (using a sliding window of 3 consecutive hours) to capture temporal context.
- **Modeling:** Constructs and trains an LSTM-based neural network with dropout regularization to classify weather conditions.
- **Evaluation:** Evaluates the model's performance and aggregates predictions by date for clearer insights.

## Prerequisites

Make sure you have Python 3.x installed along with the following packages:

- numpy
- pandas
- requests
- tensorflow
- scikit-learn

You can install the required packages via pip:

```bash
pip install numpy pandas requests tensorflow scikit-learn
```
````

## Setup

### API Key

Replace the placeholder API key in the script:

```python
API_KEY = 'e178fd28eedf49f6bab70205251602'
```

with your active WeatherAPI.com API key.

### Location & Forecast Days

The script is configured to request a 3-day forecast for Manila. To change the location or the number of days, modify the URL in the script accordingly:

```python
url = f'http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q=Manila&days=3'
```

## Running the Script

Simply run the Python script:

```bash
python predict.py
```

The script will:

- Fetch the weather forecast data.
- Process and prepare the dataset.
- Create sequences from the hourly data.
- Build and train the LSTM model.
- Evaluate the model and output aggregated predictions by date.

## Code Structure

**Data Acquisition:**

- Uses the `requests` library to retrieve weather data from WeatherAPI.com.

**Data Preprocessing:**

- Converts the JSON response into a Pandas DataFrame, extracts relevant features (`temp_c`, `humidity`, `pressure_mb`, `wind_kph`, `precip_mm`), handles missing values, and encodes the weather condition using LabelEncoder and one-hot encoding.

**Sequence Creation:**

- Implements a function to create sequences (with a window size of 3 hours) to capture the temporal dynamics of weather conditions.

**Model Building:**

- Constructs a Sequential LSTM model with dropout layers using TensorFlow/Keras:
  - Two LSTM layers (with 50 units each).
  - Dropout layers to prevent overfitting.
  - A Dense layer with softmax activation for multi-class classification.

**Training and Evaluation:**

- Splits the dataset into training and testing sets, trains the model for 50 epochs with a batch size of 8, evaluates the model performance, and aggregates predictions by date using the mode (most frequent prediction).

## Example Output

After running the script, you might see an output similar to:

```yaml
Model Loss: 0.XX, Accuracy: 0.XX

Aggregated Predictions on test samples (one per day):
Date: 2025-XX-XX, Day: Today, Predicted: Sunny, True: Cloudy
Date: 2025-XX-XX, Day: Tomorrow, Predicted: Rainy, True: Rainy
...
```

## Acknowledgements

- WeatherAPI.com for providing the weather data.
- The TensorFlow and Keras teams for their excellent deep learning frameworks.
- Contributors to the open-source Python libraries used in this project.

```

```
