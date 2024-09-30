import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import os
from scipy.cluster.vq import vq, kmeans, whiten

nlp = spacy.load("en_core_web_sm")

# Load the data
data_df = pd.read_csv('Travel_Chalenge.csv', sep=";")

# Remove rating without a value
# Convert the column to numeric, coercing errors to NaN
data_df["Overall_Rating"] = pd.to_numeric(data_df["Overall_Rating"],
                                          errors='coerce')

# Drop rows with NaN values in the specified column
data_df = data_df.dropna(subset=["Overall_Rating"])

columns_to_be_analyzed = ["Review", "Review_Title"]

pre_pickled_file = "preprocessed_data.pkl"
if not os.path.exists(pre_pickled_file) or os.path.getsize(
        pre_pickled_file) == 0:
    for column in columns_to_be_analyzed:
        # Convert to lowercase
        data_df[column] = data_df[column].apply(lambda x: x.lower()
                                                if isinstance(x, str) else x)
        # Remove non-words and punctuation
        data_df[column].replace(to_replace=r'[^\w\s]',
                                value='',
                                regex=True,
                                inplace=True)

        # Tokenize and remove stopwords
        data_df[column] = data_df[column].apply(lambda x: " ".join(
            [token.text for token in nlp(x) if not token.is_stop]))

        # Apply lemmatization
        data_df[column] = data_df[column].apply(lambda x: " ".join(
            [token.lemma_ for token in nlp(x)], leave=False))
    data_df.to_pickle(pre_pickled_file)
else:
    data_df = pd.read_pickle(pre_pickled_file)

# Plot the distribution of the ratings
rating_counts = data_df["Overall_Rating"].value_counts().sort_index()
plt.figure(1)
plt.subplot(121)
sns.barplot(x=rating_counts.index, y=rating_counts.values, color="Blue")
plt.title("Distribution of Ratings")

df_without_nan_aircraft = data_df.dropna(subset=["Aircraft"]).copy()

# Remove non-words and punctuation
df_without_nan_aircraft["Aircraft_doc"] = df_without_nan_aircraft[
    "Aircraft"].replace(to_replace=r'[^\w\s]', value='-', regex=True)

df_without_nan_aircraft["Aircraft_doc"] = df_without_nan_aircraft[
    "Aircraft_doc"].apply(lambda x: x.lower() if isinstance(x, str) else x)

# Tokenize and remove stopwords
df_without_nan_aircraft["Aircraft_doc"] = df_without_nan_aircraft[
    "Aircraft_doc"].apply(lambda x: nlp(x))

df_without_nan_aircraft["Aircraft_vec"] = df_without_nan_aircraft[
    "Aircraft_doc"].apply(lambda x: x.vector)

# features = df_without_nan_aircraft["Aircraft_doc"].values
# whitened_features = whiten(features)
