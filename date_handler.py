import got3
import arrow
from textblob import TextBlob
import numpy as np

def format_date(date):
    months = {"Jan" : 1, "Feb" : 2, "Mar" : 3, "Apr" : 4, "May" : 5, "Jun" : 6, "Jul" : 7, "Aug" : 8, "Sep" : 9, "Oct" : 10, "Nov" : 11, "Dec" : 12}
    date = date.split('-')

    DD = str(date[0])

    MM = int(months[date[1]])
    if MM < 10:
        MM = "0" + str(MM)
    else:
        MM = str(MM)

    YYYY = int(date[2])
    if YYYY < 25:
        YYYY = str(YYYY + 2000)
    else:
        YYYY = str(YYYY + 1900)

    output_date = YYYY + "-" + MM + "-" + DD
    return arrow.get(output_date)


def date_to_sentiment(dates, ticker, max_tweets):

    sentiments = []
    for d in dates:
        arrow_date = format_date(d)
        tweetCriteria = got3.manager.TweetCriteria().setQuerySearch("{}{}".format("#", ticker)).setMaxTweets(max_tweets) \
            .setSince(arrow_date.format("YYYY-MM-DD")) \
            .setUntil(arrow_date.replace(days=1).format("YYYY-MM-DD"))

        tweets = got3.manager.TweetManager.getTweets(tweetCriteria)

        sents_per_date = []
        for t in tweets:
            print(t.text)
            blob = TextBlob(t.text)
            sent = blob.sentiment[0] #get the polarity (subjectivity seems less important)
            sents_per_date.append(sent)

        sents_per_date = np.asarray(sents_per_date)

        sentiments.append(sents_per_date.mean())

    return sentiments


# #UNIT TEST
# dates = ['10-Aug-16', '11-Aug-16']
# ticker = 'GOOGL'
# max_tweets = 200
#
# print(date_to_sentiment(dates, ticker, max_tweets))

