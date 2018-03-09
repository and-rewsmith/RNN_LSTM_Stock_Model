import got3
import arrow
from textblob import TextBlob
import numpy as np


def dates_to_sentiment(dates, ticker, max_tweets):

    ticker = "$" + ticker

    print(dates)

    sentiments = []
    for d in dates:
        arrow_date = arrow.get(d, "YYYY-MM-DD")

        tweetCriteria = got3.manager.TweetCriteria().setQuerySearch("{}{}".format("#", ticker)).setMaxTweets(max_tweets) \
            .setSince(arrow_date.replace(days=-1).format("YYYY-MM-DD")) \
            .setUntil(arrow_date.format("YYYY-MM-DD"))
        tweets = got3.manager.TweetManager.getTweets(tweetCriteria)

        sents_per_date = []
        for t in tweets:
            #print(t.text)
            blob = TextBlob(t.text)
            sent = blob.sentiment[0] #get the polarity (subjectivity seems less important)
            sents_per_date.append(sent)

        print("length: " + str(len(sents_per_date)))

        mean_sentiment = sum(sents_per_date) / len(sents_per_date)

        sentiments.append(mean_sentiment)

        # #warning insight
        # try:
        #     sentiments.append(sents_per_date.mean())
        # except RuntimeWarning:
        #     print("RUNTIME WARNING")
        #     print(d)
        #     print(sents_per_date)
        #     for t in tweets:
        #         print(t.text)

    sentiments = np.asarray(sentiments)

    return sentiments


 #UNIT TEST
dates = ['2018-03-01']
ticker = '$GOOGL'
max_tweets = 500
print(dates_to_sentiment(dates, ticker, max_tweets))
