import re
import pandas as pd
import string
import pickle

from sklearn.feature_extraction import text 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from collections import defaultdict

titles = []
dates = []
subgenres = []
ratings = []
times = []
authors = []
stories = []

story_text = ''

with open('files/stories.txt', 'r') as f:
    lines = f.readlines()
    
    for i, line in enumerate(lines):

        if re.match('Title: ', line):
            if story_text is not '':
                stories.append(story_text)
            story_text = ''
            titles.append(re.split('Title: ', line)[1][:-1])
        elif re.match('Date: ', line):
            if re.match('Subgenre: ', lines[i+1]):
                dates.append(re.split('Date: ', line)[1][:-1])
        elif re.match('Subgenre: ', line):
            subgenres.append(re.split('Subgenre: ', line)[1][:-1])
        elif re.match('Rating: ', line):
            ratings.append(re.split('Rating: ', line)[1][:-1])
        elif re.match('Reading Time: ', line):
            times.append(re.split('Reading Time: ', line)[1][:-1])
            
            if re.match('Author: ', lines[i+1]):
                authors.append(re.split('Author: ', lines[i+1])[1][:-1]) 
            else:
                authors.append('')            
        else: 
            if re.match('Author: ', line):
                continue
                
            story_text += line
            
stories.append(story_text)

creep = pd.DataFrame({'title': titles,
                      'date': dates,
                      'subgenre': subgenres,
                      'rating': ratings,
                      'time': times,
                      'author': authors,
                      'story': stories})

def clean_title(title):
    title = title.lower()
    title = re.sub('[“‘’”…\*]', '', title)
    title = re.sub('[-–]', ' ', title)
    title = re.sub('[%s]' % re.escape(string.punctuation), ' ', title)
    title = title.strip()
    
    return title

def clean_author(name):
    if re.search('\(', name):
        name = re.split('\(', name)[0].strip()
    if re.search('a.k.a', name):
        name = re.split('a.k.a', name)[0].strip()
    if re.search('Anonymous', name):
        name = 'Author Unknown'
    if re.search('\.', name):
        name = re.sub('\.', '', name).strip()
    
    return name

def clean_subgenre(subgenre):
    subgenre = subgenre.lower()
    subgenre = re.sub('and|the', '', subgenre).strip()
    subgenre = re.sub(',', '', subgenre).strip()
    
    return subgenre

def clean_stories(story):
    
    if re.search("Author’s note:", story):
        temp = re.split("Author’s note:", story)[0]
        if len(temp) > 10:
            story = temp
            
    story = story.lower()
    story = re.sub('\xa0|\n', ' ', story)
    story = re.sub('\(.*?\)', ' ', story)
    story = re.sub('\[.*?\]', ' ', story)
    story = re.sub('[“‘’”…\*]', '', story)
    story = re.sub('[-–]', ' ', story)
    story = re.sub('\d+', '', story)
    #needs extra space to avoid word merging 
    story = re.sub('[%s]' % re.escape(string.punctuation), ' ', story)
    story = re.sub('[^\x00-\x7f]', '', story)
    story = story.strip()
    
    return story

def clean_stories_for_BERT(story):
    
    if re.search("Author’s note:", story):
        temp = re.split("Author’s note:", story)[0]
        if len(temp) > 10:
            story = temp
            
    story = story.lower()
    story = re.sub('\xa0|\n', ' ', story)
    story = re.sub('\(.*?\)', ' ', story)
    story = re.sub('\[.*?\]', ' ', story)
    story = re.sub('[“‘’”…\*]', '', story)
    story = re.sub('[-–]', ' ', story)
    story = re.sub('\d+', '', story)
    #needs extra space to avoid word merging 
    story = re.sub(',', ' ', story)
    story = re.sub('[^\x00-\x7f]', '', story)    
    story = story.strip()
    
    return story

creep['title'] = creep['title'].apply(clean_title)
creep['author'].replace('', 'Author Unknown', inplace=True)
creep['author'] = creep['author'].apply(clean_author)
creep['date'] = pd.to_datetime(creep['date'])
creep['time'] = creep['time'].replace('< 1', 1)
creep['subgenre'] = creep['subgenre'].apply(clean_subgenre)
creep.loc[:, 'rating':'time'] = creep.loc[:, 'rating':'time'].apply(pd.to_numeric, errors='coerce')

creep_BERT = creep.loc[:, ['rating','story']]

creep['story'] = creep['story'].apply(clean_stories)
creep.drop(index = (creep[creep.story == '']).index, inplace=True)
creep.reset_index(drop=True, inplace=True)

creep_BERT['story'] = creep_BERT['story'].apply(clean_stories_for_BERT)
creep_BERT.drop(index = (creep_BERT[creep_BERT.story == '']).index, inplace=True)
creep_BERT.reset_index(drop=True, inplace=True)

def lemmatize_story(story):
    lem = WordNetLemmatizer()
    pos_map = defaultdict(lambda : wn.NOUN)
    pos_map['J'] =  wn.ADJ
    pos_map['V'] =  wn.VERB
    pos_map['R'] =  wn.ADV
    
    lemma = [word for word in word_tokenize(story) if word not in text.ENGLISH_STOP_WORDS]

    lemma = [lem.lemmatize(word, pos_map[tag[0]]) for word,tag in pos_tag(lemma)]
    return ' '.join(lemma)

creep['story'] = creep['story'].apply(lemmatize_story)
creep_BERT['story'] = creep_BERT['story'].apply(lemmatize_story)

corpus = pd.DataFrame(data=creep, columns =['title','story'])
creep.drop('story', axis=1, inplace=True)

corpus.to_pickle('files/corpus.pkl')
creep.to_pickle('files/creep.pkl')
creep_BERT.to_pickle('files/BERT_corpus.pkl')