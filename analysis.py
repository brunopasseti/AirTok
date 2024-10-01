import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import os
from scipy.cluster.vq import vq, kmeans, whiten

nlp = spacy.load("en_core_web_md")

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
        # data_df[column] = data_df[column].apply(lambda x: " ".join(
        #     [token.text for token in nlp(x) if not token.is_stop]))
        data_df[column + "_doc"] = data_df[column].apply(lambda x: nlp(x))
        data_df[column + "_doc_tokens"] = data_df[column + "_doc"].apply(
            lambda doc: " ".join(
                [token.text for token in doc if not token.is_stop]))
        # Apply lemmatization
        data_df[column] = data_df[column + "_doc"].apply(
            lambda doc: [token.lemma_ for token in doc])
    data_df.to_pickle(pre_pickled_file)
else:
    data_df = pd.read_pickle(pre_pickled_file)

# Plot the distribution of the ratings
rating_counts = data_df["Overall_Rating"].value_counts().sort_index()
plt.figure(1)
sns.barplot(x=rating_counts.index, y=rating_counts.values, color="Blue")
plt.title("Distribution of Ratings")

df_without_nan_aircraft = data_df.dropna(subset=["Aircraft"]).copy()

# Remove non-words and punctuation
df_without_nan_aircraft["Aircraft_doc"] = df_without_nan_aircraft[
    "Aircraft"].replace(to_replace=r'[^\w\s]', value=' ', regex=True)

df_without_nan_aircraft["Aircraft_doc"] = df_without_nan_aircraft[
    "Aircraft_doc"].apply(lambda x: x.lower() if isinstance(x, str) else x)

# Tokenize and remove stopwords
df_without_nan_aircraft["Aircraft_doc"] = df_without_nan_aircraft[
    "Aircraft_doc"].apply(lambda x: nlp(x))

df_without_nan_aircraft["Aircraft_vec"] = df_without_nan_aircraft[
    "Aircraft_doc"].apply(lambda x: x.vector)

whitened_features = whiten(df_without_nan_aircraft["Aircraft_vec"].to_list())

codebook, distortion = kmeans(whitened_features, 45)
labels, test = vq(whitened_features, codebook)
df_without_nan_aircraft[
    "Aircraft_label"] = df_without_nan_aircraft.Aircraft_doc[
        df_without_nan_aircraft.Aircraft_doc.index[labels]].to_list()
df_without_nan_aircraft["Aircraft_label"] = df_without_nan_aircraft[
    "Aircraft_label"].apply(lambda x: x.__repr__())

plt.figure(2)
ax = sns.boxplot(x="Aircraft_label",
                 y="Overall_Rating",
                 data=df_without_nan_aircraft)
plt.setp(ax.get_xticklabels(),
         rotation=45,
         ha="right",
         rotation_mode="anchor",
         fontsize=8)
plt.title("K-means clustering of Aircrafts 1, 2")

# Creating word cloud of reviews
from wordcloud import WordCloud
# Getting all words with review is less than equal to 3
all_words_leq_3 = " ".join(
    data_df["Review_doc_tokens"][data_df["Overall_Rating"] <= 3])
wordcloud = WordCloud(width=1600, height=800).generate(all_words_leq_3)
fig3 = plt.figure(3)
ax3 = fig3.add_axes([0, 0, 1, 1])
ax3.set_axis_off()
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Word Cloud of Reviews with Rating <= 3")
#Getting all words with review is greater than equal to 8
all_words_geq_8 = " ".join(
    data_df["Review_doc_tokens"][data_df["Overall_Rating"] >= 8])
wordcloud = WordCloud(width=1600, height=800).generate(all_words_geq_8)
fig4 = plt.figure(4)
ax4 = fig4.add_axes([0, 0, 1, 1])
ax4.set_axis_off()
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Word Cloud of Reviews with Rating >= 8")

# Compare the similarity of the reviews
messages = data_df["Review_doc_tokens"].to_list()

doc_to_compare_with = nlp("delay flight")
delay_flights = []
# for doc in messages:
#     if doc.similarity(doc_to_compare_with) > 0.5:
#         # print(doc, "<->", doc_to_compare_with, doc.similarity(doc_to_compare_with))
#         delay_flights.append(doc.text)

# # writing reviews with delayed flights to file
# with open("delayed_flights.txt", "w") as file:
#     for review in delay_flights:
#         file.write(review + "\n")

plt.show()
