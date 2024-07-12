import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = 'E:\\INRERN PROJECTS\\TASK-4\\Sentiment.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the data
data.head(), data.info()

# Drop unnecessary columns and rename columns for clarity
data_cleaned = data.drop(columns=['2401', 'Borderlands'])
data_cleaned.columns = ['Sentiment', 'Text']

# Display the cleaned data
data_cleaned.head()

# Function to get sentiment polarity
def get_sentiment_polarity(text):
    if isinstance(text, str):  # Check if the input is a string
        return TextBlob(text).sentiment.polarity
    else:
        return np.nan  # Return NaN for non-string values

# Apply the function to get sentiment polarity for each text
data_cleaned['SentimentPolarity'] = data_cleaned['Text'].apply(get_sentiment_polarity)

# Remove rows with NaN values
data_cleaned.dropna(inplace=True)

# Display the updated dataframe with sentiment polarity
print(data_cleaned.head())

# Plot the distribution of sentiment polarity
plt.figure(figsize=(10, 6))
sns.histplot(data_cleaned['SentimentPolarity'], bins=20, kde=True)
plt.title('Distribution of Sentiment Polarity')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()
