import pandas as pd

from textblob import TextBlob

corpus = pd.read_pickle('files/corpus.pkl')
data = pd.read_pickle('files/creep.pkl')

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data['polarity'] = corpus['story'].apply(pol)
data['subjectivity'] = corpus['story'].apply(sub)

data.to_pickle('files/creep.pkl')