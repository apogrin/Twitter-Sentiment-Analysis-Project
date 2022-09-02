from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd

# open file of scraped tweets
with open("tweetsSearch.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()

    # list to store tweet and sentiment data
    data = []

    # create counters for number of each tweet
    positive_counter = 0
    negative_counter = 0
    neutral_counter = 0

    # loop through tweets and set current tweet
    for i in range(0, len(lines), 2):
        tweet = lines[i]

        # preprocess tweet
        tweet_words = []

        for word in tweet.split(' '):
            if word.startswith('@') and len(word) > 1:
                word = '@user'

            elif word.startswith('http'):
                word = "http"
            tweet_words.append(word)

        tweet_proc = " ".join(tweet_words)

        # load model and tokenizer
        roberta = "cardiffnlp/twitter-roberta-base-sentiment"

        model = AutoModelForSequenceClassification.from_pretrained(roberta)
        tokenizer = AutoTokenizer.from_pretrained(roberta)

        labels = ['Negative', 'Neutral', 'Positive']

        # sentiment analysis
        encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
        # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
        output = model(**encoded_tweet)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        # for j in range(len(scores)):
        #     label = labels[j]
        #     score = scores[j]
        #     print(label, score)

        # appends tweet and sentiment data to the list
        data.append([tweet, scores[0], scores[1], scores[2]])

        if scores[0] > scores[1] and scores[0] > scores[2]:
            negative_counter += 1
        elif scores[1] > scores[0] and scores[1] > scores[2]:
            neutral_counter += 1
        else:
            positive_counter += 1


market_sentiment = ""
if negative_counter > neutral_counter and negative_counter > positive_counter:
    market_sentiment = "negative"
elif neutral_counter > negative_counter and neutral_counter > positive_counter:
    market_sentiment = "neutral"
elif positive_counter == negative_counter:
    market_sentiment = "neutral"
else:
    market_sentiment = "positive"

df = pd.DataFrame(data, columns=['Tweet', 'Negativity', 'Neutrality', 'Positivity'])
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3,):
    print(df)

print(f"Out of {len(lines)/2} tweets, there were {negative_counter} negative tweets, {neutral_counter} neutral tweets and"
      f" {positive_counter} positive tweets, with the overall sentiment being {market_sentiment}.")

