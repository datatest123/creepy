# creepy

This project explores a sample of Internet horror stories called CreepyPastas.

The objective of this project is answer the question "What makes a good CreepyPasta?" In order to answer this question, I scraped a sample of around 3500 CreepyPastas from the website creepypasta.com. These stories were published on the website from 2008-2020 and represent a good cross-section of these kind of stories and how they've evolved as a genre over time.

How does one determine a good CreepyPasta? To answer this question quantitatively, I recorded the rating out of 10 for each scraped story. This rating captures how users on the website felt about each story, and it can be used as a good enough proxy for story quality. 

I collected and created additional story metadata to use as predictors of rating. I then explored the story data using EDA and NLP techniques. Finally, I attempted to model rating as a function of story metadata and story word embeddings. My findings are shown in the jupyter notebooks in the main repo. 

There are several scripts I created to parse the story data:

scraper.py - crawls creepypasta.com and processes the raw html to extract data
cleaner.py - cleans and processes story text and metadata
sentiment.py - performs sentiment analysis on the story text for use as metadata

The .ipynb notebooks contain story analysis:

EDA - exploratory data analysis on the story word counts and metadata
LSA - latent semantic analysis of hidden topics in the stories along with clustering 
LDA - latent dirichlet allocation for hidden topic discovery
Regression - regression analysis of story data
Evaluation - analysis of model results

The model I created for predicting rating is a deep learning model comprised of 2 parts. First, the stories are embeded using an RNN-based word vectorizer called BERT. Then, the embedings are feed into a deep feed-foward network to learn patterns that predict rating. The files containing the model are listed:

train_network.py - define the NN models and train on a story subset
test_network.py - test the NN model on validation and test subsets

Finally, several files are created when running these notebooks/scripts. These are stored in the files directory
