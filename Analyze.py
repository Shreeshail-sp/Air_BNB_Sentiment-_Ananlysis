import csv
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords  # stopwords to detect language
from nltk import word_tokenize  # function to split up our words
from nltk import RegexpTokenizer
import re
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from langdetect import detect
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager
import seaborn as sns
import argparse
import sys
import os
from datetime import datetime
import emoji
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar

"""
This program will take a .csv file of Airbnb listings as input, analyze
the sentiments of all the reviews for each given listing, and then output
various forms of data (csv as well as visualization of the most frequently
mentioned features within those reviews).

The overall rankings of the chosen listings will be generated in a .csv file
in the root directory. All other .csv files and graphs will be generated in
the 'results' subdirectory.

For more information on how this program works or how to use it, please refer
to the README.

By Kashif Khan
"""


def main():
    # Parse over the arguments and load the appropriate data into a dataframe
    df, args = initialize_program()

    print("====== Starting Analysis ======")

    # Initialize the sentiment analyzer object
    sid = SentimentIntensityAnalyzer()

    # Update the VADER lexicon with words that indicate positive/negative emotions as observed/needed
    sid.lexicon.update({
        "canceled": -0.1,
        "decent": 0.1,
        "convenient": 0.1,
        "homey": 0.1,
        "cozy": 0.1,
        "stop": 0.15,
        "block": 0,
        "blocks": 0,
    })

    # Remove all NaN values from the dataframe
    if 'comments' not in df.columns:
        print("ERROR: 'comments' column not found in input file. Exiting.")
        return
    df = df[pd.notnull(df["comments"])]

    # Add new analysis columns with robust checks
    def safe_analyze_emojis(x):
        try:
            return len(analyze_emojis(str(x)))
        except Exception:
            return 0
    df['emoji_count'] = df['comments'].apply(safe_analyze_emojis)

    def safe_review_length(x):
        try:
            return analyze_review_length(str(x))
        except Exception:
            return 'Unknown'
    df['review_length'] = df['comments'].apply(safe_review_length)

    # Only add seasonal analysis if date column exists
    has_date = False
    if 'date' in df.columns:
        try:
            df['season'] = df['date'].apply(get_season)
            has_date = True
        except Exception as e:
            print(f"Warning: Could not process date column for seasonal analysis: {e}")
            has_date = False
    else:
        print("Warning: 'date' column not found. Skipping seasonal and temporal analysis.")

    # Calculate sentiment score for each review if not already present
    if 'sentiment_score' not in df.columns:
        def safe_sentiment(x):
            try:
                return sid.polarity_scores(str(x))['compound']
            except Exception:
                return 0
        df['sentiment_score'] = df['comments'].apply(safe_sentiment)

    # Read in the data from the dataframe into a dictionary grouped by listing id as keys
    if 'listing_id' not in df.columns:
        print("ERROR: 'listing_id' column not found in input file. Exiting.")
        return
    base_dict = dict(tuple(df.groupby("listing_id")))

    # Save the NLTK-defined stopwords into a dictionary for quick access
    cached_stopwords = stopwords.words("english")
    stopwords_dict = Counter(cached_stopwords)

    sentiment_dict = {}
    listing_ids = []

    # A Noun Phrase = A phrase which containS a noun (feature) surrounded by an adjective which can be used to
    # determine sentiment towards that feature (noun)
    pattern = "NP: {(<NN>|<NNS>)(<VBZ>|<VBD>)*<CC>*<RB>*(<JJ>|<NN>)|<JJ>(<NN>|<NNS>)|(<NN>|<NNS>)(<VBZ>|<VBD>)<RB>*<JJ>}"

    for listing_id in base_dict:
        print("==== Analyzing Listing ID: ", listing_id, " ====")

        # Re-format the comments for each listing_id to a list instead of series
        base_dict[listing_id] = base_dict[listing_id]["comments"].values.tolist()

        # These four variables will hold the comments/features classified as positive/negative based on polarity
        negative_comments_list = []
        positive_comments_list = []
        negative_features_list = {}
        positive_features_list = {}
        sentiment_score = 0  # The overall sentiment polarity of the listing

        # For every review in each listing_id
        for comment in base_dict[listing_id]:

            # Use a try/catch block to catch errors where langdetect cannot identify the language of the comment
            try:

                # Only proceed if the comment is in English
                if detect(comment) == "en":

                    # Maintain a list of listing_ids with valid English reviews
                    if listing_id not in listing_ids:
                        listing_ids.append(listing_id)

                    # Generate a sentiment score (with polarity) for the given review
                    ss = sid.polarity_scores(comment)

                    """ 
                    We classify neutral (0) reviews as positive since manual analysis of tweets indicate that neutrally classified comments 
                    simply used 'neutral language' to praise a listing as opposed to the sentiments actually being neutral
                    """
                    # If the comment is negative, place it in the negative comments list and update the sentiment score of the overall listing
                    if ss["compound"] < 0:
                        # f_neg.write(" ".join(comment4) + "\n")
                        # f_neg_full.write(" ".join(comment2) + "\n")
                        negative_comments_list.append(comment.strip())
                        sentiment_score += ss["compound"]

                    # If the comment is positive, place it in the positive comments list and update the sentiment score of the overall listing
                    if ss["compound"] >= 0:
                        # f_pos.write(" ".join(comment4) + "\n")
                        # f_pos_full.write(" ".join(comment2) + "\n")
                        positive_comments_list.append(comment.strip())
                        sentiment_score += ss["compound"]

                    """
                    Now that the comment's overall sentiment is generated, we can break down the comment to
                    extract features and the sentiments about those features
                    """

                    # Define a noun phrase as a phrase matching the earlier regex pattern
                    cp = nltk.RegexpParser(pattern)

                    # Tokenize the comment
                    tokenized_word = word_tokenize(comment.lower())

                    # Apply POS tags to each token
                    pos_word = nltk.pos_tag(tokenized_word)

                    # Identify noun chunks based on the regex defined earlier
                    cs = cp.parse(pos_word)

                    # Find all labelled noun phrases and add them to the key phrases list
                    for subtree in cs.subtrees():
                        if subtree.label() == "NP":
                            noun_phrase = " ".join(
                                word for word, tag in subtree.leaves()
                            )

                            # Identify the root noun and the root adjective describing that noun for each noun phrase
                            negation = False
                            for word, tag in subtree.leaves():
                                if (
                                    tag == "NN" or tag == "NNS"
                                ) and word not in stopwords_dict:
                                    root_noun = word
                                if tag == "JJ":
                                    root_adj = word
                                if word == "not":
                                    negation = True

                            # If negation is true, we negate the adjective
                            if negation:
                                root_adj = "not " + root_adj

                            # Identify the overall sentiment of the noun phrase
                            ss = sid.polarity_scores(noun_phrase)

                            # If the sentiment of the noun phrase is negative, classify the noun as a negative feature
                            if ss["compound"] < 0:

                                if root_noun in negative_features_list.keys():
                                    negative_features_list[root_noun][
                                        "adjectives"
                                    ].append(root_adj)
                                    negative_features_list[root_noun]["count"] += 1
                                else:
                                    negative_features_list[root_noun] = {
                                        "adjectives": [root_adj]
                                    }
                                    negative_features_list[root_noun]["count"] = 1

                            # If the sentiment of the noun phrase is positive, classify the noun as a positive feature
                            elif ss["compound"] > 0:
                                if root_noun in positive_features_list.keys():
                                    positive_features_list[root_noun][
                                        "adjectives"
                                    ].append(root_adj)
                                    positive_features_list[root_noun]["count"] += 1
                                else:
                                    positive_features_list[root_noun] = {
                                        "adjectives": [root_adj]
                                    }
                                    positive_features_list[root_noun]["count"] = 1

            # An exception being thrown here indicates that the text was not in any identifable language - so we can ignore it
            except Exception as identifier:
                pass

        # Populate the listing_id key with the accumulated results of the sentiment analysis over all comments for a given listing
        if len(positive_comments_list) + len(negative_comments_list) == 0:
            sentiment_score = 0
        else:
            sentiment_score = sentiment_score / (
                len(positive_comments_list) + len(negative_comments_list)
            )

        sentiment_dict[listing_id] = {
            "positive_comments": positive_comments_list,
            "negative_comments": negative_comments_list,
            "positive_features": positive_features_list,
            "negative_features": negative_features_list,
            "sentiment_score": (
                sentiment_score  # This score is an average score over all listing reviews
            ),
        }

        # Now that population of each comment and feature is complete, we can generate data for this listing
        for sentiment in ["positive_features", "negative_features"]:

            # Initialize a dataframe from the sentiment dictionary and sort by # of times feature was mentioned
            sent_df = pd.DataFrame.from_dict(
                sentiment_dict[listing_id][sentiment],
                orient="index",
                columns=["count", "adjectives"],
            ).sort_values(by="count", ascending=False)

            # Generate noun + adjective .csv files if the user requests it
            if args.output_type in [3, 5]:

                # If there is any data to show, generate a .csv file
                if len(sent_df.index.values.tolist()) > 0:
                    sent_df.to_csv(
                        "results/{}_nouns_and_adjectives_{}.csv".format(
                            listing_id, sentiment.split("_")[0]
                        )
                    )

            # Generate noun frequency bar graphs if the user requests it
            if args.output_type in [2, 5]:
                # Generate the bar graph of top 20 features and their frequency
                generate_bar_graph(sent_df.head(20), listing_id, sentiment)

            # Generate noun word clouds if the user requests it
            if args.output_type in [4, 5]:
                # Generate a wordcloud that highlights the most frequently mentioned features
                generate_wordcloud(sentiment_dict, listing_id, sentiment)

    # After processing all listings, generate enhanced visualizations
    if args.output_type in [4, 5]:
        # Create interactive wordclouds
        try:
            positive_text = ' '.join([comment for listing in sentiment_dict.values() 
                                    for comment in listing.get('positive_comments', []) if isinstance(comment, str)])
            negative_text = ' '.join([comment for listing in sentiment_dict.values() 
                                    for comment in listing.get('negative_comments', []) if isinstance(comment, str)])
            if positive_text.strip():
                pos_wordcloud = create_interactive_wordcloud(positive_text, 'Positive Features Word Cloud')
                pos_wordcloud.write_html('results/positive_wordcloud_interactive.html')
            else:
                print("Warning: No positive comments for word cloud.")
            if negative_text.strip():
                neg_wordcloud = create_interactive_wordcloud(negative_text, 'Negative Features Word Cloud')
                neg_wordcloud.write_html('results/negative_wordcloud_interactive.html')
            else:
                print("Warning: No negative comments for word cloud.")
        except Exception as e:
            print(f"Warning: Could not generate word clouds: {e}")

        # Generate temporal analysis only if date column exists
        if has_date:
            try:
                temporal_fig = analyze_temporal_patterns(df)
                temporal_fig.write_html('results/sentiment_trends.html')
            except Exception as e:
                print(f"Warning: Could not generate sentiment trends over time: {e}")
            # Generate seasonal sentiment analysis directly from DataFrame
            try:
                if 'season' in df.columns:
                    seasonal_df = df.groupby('season')['sentiment_score'].mean().reset_index()
                    if not seasonal_df.empty:
                        seasonal_fig = px.bar(seasonal_df, x='season', y='sentiment_score',
                                            title='Average Sentiment by Season')
                        seasonal_fig.write_html('results/seasonal_sentiment.html')
                    else:
                        print("Warning: Not enough data for seasonal sentiment analysis.")
            except Exception as e:
                print(f"Warning: Could not generate seasonal sentiment analysis: {e}")

        # Generate review length analysis directly from DataFrame
        try:
            if 'review_length' in df.columns:
                length_df = df.groupby('review_length')['sentiment_score'].mean().reset_index()
                if not length_df.empty:
                    length_fig = px.bar(length_df, x='review_length', y='sentiment_score',
                                      title='Average Sentiment by Review Length')
                    length_fig.write_html('results/length_sentiment.html')
                else:
                    print("Warning: Not enough data for review length sentiment analysis.")
        except Exception as e:
            print(f"Warning: Could not generate review length sentiment analysis: {e}")

        # Generate emoji analysis
        try:
            emoji_list = []
            for comment in df['comments']:
                try:
                    emoji_list.extend(analyze_emojis(str(comment)))
                except Exception:
                    continue
            if emoji_list:
                emoji_df = pd.DataFrame(Counter(emoji_list).most_common(20),
                                      columns=['Emoji', 'Count'])
                emoji_fig = px.bar(emoji_df, x='Emoji', y='Count',
                                  title='Top 20 Most Used Emojis')
                emoji_fig.write_html('results/emoji_analysis.html')
            else:
                print("Warning: No emojis found for analysis.")
        except Exception as e:
            print(f"Warning: Could not generate emoji analysis: {e}")

    # Generate the final ranked listings CSV
    try:
        generate_ranked_listings(sentiment_dict, listing_ids)
    except Exception as e:
        print(f"Warning: Could not generate ranked listings CSV: {e}")

    print("--- Complete ---")


## This function will parse the input arguments and return a parser, as well as generate a 'results' folder where
## the optional graphs and adjectives .csv files will be stored
def initialize_program():

    # Ensure all the proper NLTK packages are downloaded correctly
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
        nltk.data.find("sentiment/vader_lexicon.zip")
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except Exception as e:
        print("Could not find all required NLTK modules. Downloading now...")
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("stopwords")
        nltk.download("vader_lexicon")

    # Ensure the arguments are valid and parse over them
    try:
        # Parse the command line arguments and extract the input variables
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "source_file", type=str
        )  # The location of the reviews input file
        parser.add_argument("output_type", type=int)
        parser.add_argument("range", nargs="?", type=str, default="")

        args = parser.parse_args()

        # Load the source file as a dataframe
        df = pd.read_csv(args.source_file)[["listing_id", "comments"]]

        # Determine the form of the output and ensure it's a valid option
        if args.output_type not in range(1, 6):
            raise Exception

        # Determine whether there is any restriction on listing ids by the user
        if args.range == "":
            pass
        elif ":" in args.range and args.range[0] == "[" and args.range[-1] == "]":
            df_lower = int(args.range.split(":")[0][1:])
            df_upper = int(args.range.split(":")[1][:-1])
            df = df.loc[df_lower:df_upper]
        elif "," in args.range and args.range[0] == "[" and args.range[-1] == "]":
            listings = [int(i) for i in args.range[1:-1].split(",")]
            df = df.loc[df["listing_id"].isin(listings)]
        elif "," not in args.range and args.range[0] == "[" and args.range[-1] == "]":
            listings = [int(i) for i in [args.range[1:-1]]]
            df = df.loc[df["listing_id"].isin(listings)]
        else:
            raise Exception

    except Exception as e:
        print(e)
        print(
            "Please enter valid arguments, i.e: python analyzer.py data/reviews_chicago.csv 1 [0:1000]"
        )
        print("If unsure, please refer to the README for valid arguments.")
        sys.exit()

    # Now attempt to create a 'results' folder
    try:
        if not os.path.exists("results"):
            os.makedirs("results")
    except Exception as identifier:
        print(
            "Unable to create '/results' folder. Please make sure a 'results' folder exists in the main program directory."
        )
        sys.exit()

    return df, args


# This helper function generates a bar-graph of the top 20 most frequently mentioned nouns/features
# in all the reviews for this specific listing.
def generate_bar_graph(sent_df, listing_id, sentiment):

    sentiment = sentiment.split("_")[0]

    if len(sent_df.index.values.tolist()) > 0:

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.rcParams["text.color"] = "#000000"
        plt.rcParams["axes.labelcolor"] = "#000000"
        plt.rcParams["xtick.color"] = "#000000"
        plt.rcParams["ytick.color"] = "#000000"
        plt.rcParams["font.size"] = 11

        color_palette_list = [
            "#009ACD",
            "#ADD8E6",
            "#63D1F4",
            "#0EBFE9",
            "#C1F0F6",
            "#0099CC",
        ]
        # plt.show(sns)
        ind = np.arange(len(sent_df.index))

        fig = sns.barplot(
            x=sent_df.index.values.tolist(),
            y=sent_df["count"],
            data=sent_df,
            palette=color_palette_list,
            ax=ax,
            ci=None,
        )
        ax.set_title(
            "Most Frequently Referenced Features in a {} Sense for Listing {}".format(
                sentiment.capitalize(), listing_id
            )
        )
        ax.set_ylabel("Number of Times Referenced")
        ax.set_xlabel("Noun")
        ax.set_xticks(range(0, len(ind)))
        ax.set_xticklabels(list(sent_df.index.values.tolist()), rotation=45)

        # plt.show(fig)
        # plt.autoscale()
        plt.savefig(
            "results/{}_bar_graph_{}.png".format(listing_id, sentiment),
            bbox_inches="tight",
        )
        plt.cla()
        plt.close()


# This helper function generates wordclouds of the most frequent nouns in all reviews for a given listing
def generate_wordcloud(sentiment_dict, listing_id, sentiment_name):

    words = []
    for key in sentiment_dict[listing_id][sentiment_name]:
        words.append([key, sentiment_dict[listing_id][sentiment_name][key]["count"]])

    if len(words) > 0:

        concat_str = ""
        for word in words:
            str = " ".join(word[0] for i in range(word[1]))
            concat_str = concat_str + str + " "

        wordcloud = WordCloud(
            width=800,
            height=800,
            random_state=21,
            max_font_size=110,
            background_color="white",
            collocations=False,
        ).generate(concat_str)

        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(
            "results/{}_wordcloud_{}.png".format(
                listing_id, sentiment_name.split("_")[0]
            )
        )
        plt.cla()
        plt.close()


def analyze_emojis(text):
    """Extract and analyze emojis from text"""
    emoji_list = [c for c in text if c in emoji.EMOJI_DATA]
    return emoji_list

def get_season(date_str):
    """Determine season from date string"""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    except:
        return 'Unknown'

def analyze_review_length(text):
    """Analyze review length and return category"""
    words = len(text.split())
    if words < 50:
        return 'Short'
    elif words < 150:
        return 'Medium'
    else:
        return 'Long'

def create_interactive_wordcloud(text_data, title):
    """Create an interactive wordcloud using plotly"""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    words = wordcloud.words_
    
    fig = go.Figure(data=[go.Scatter(
        x=[np.random.rand() for _ in range(len(words))],
        y=[np.random.rand() for _ in range(len(words))],
        mode='text',
        text=[word for word in words.keys()],
        textfont=dict(
            size=[words[word] * 100 for word in words.keys()],
            color=['rgb({},{},{})'.format(
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            ) for _ in range(len(words))]
        )
    )])
    
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )
    
    return fig

def analyze_temporal_patterns(reviews_df):
    """Analyze how sentiment changes over time"""
    reviews_df['date'] = pd.to_datetime(reviews_df['date'])
    reviews_df['month'] = reviews_df['date'].dt.month
    reviews_df['year'] = reviews_df['date'].dt.year
    
    monthly_sentiment = reviews_df.groupby(['year', 'month'])['sentiment_score'].mean().reset_index()
    monthly_sentiment['date'] = pd.to_datetime(monthly_sentiment[['year', 'month']].assign(day=1))
    
    fig = px.line(monthly_sentiment, x='date', y='sentiment_score',
                  title='Sentiment Trends Over Time',
                  labels={'sentiment_score': 'Average Sentiment Score', 'date': 'Date'})
    
    return fig

def generate_ranked_listings(sentiment_dict, listing_ids):
    # Generate a .csv file containing the ranked average sentiment scores of each listing
    pair_list = []
    for listing_id in sentiment_dict.keys():
        pair_list.append([listing_id, sentiment_dict[listing_id]["sentiment_score"]])

    df = pd.DataFrame(pair_list, columns=["listing_id", "sentiment_score"]).sort_values(
        by="sentiment_score", ascending=False
    )
    df.to_csv("listings_ranked.csv", index=False)

if __name__ == "__main__":
    main()
