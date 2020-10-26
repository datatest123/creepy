#--------------------------------------
# This script scrapes the creepypasta.com
# archives to obtain several thousand stories
# for use in NLP tasks. The stories are stored 
# in a .txt file in the same directory as the
# scraper. 
# 
# Sleep times are built into the script to 
# avoid flooding the creepypasta.com servers
# with requests. As a result, this script takes
# several hours to run to completion. The
# sleep times can be adjusted as needed, but 
# overloading the server may result in an IP
# ban. 
#--------------------------------------


import requests 
from bs4 import BeautifulSoup as bs
import re
from random import randint
from time import sleep

PATH = 'stories.txt'

# Create .txt file for the stories
f = open(PATH, 'a')

# Iterate through the archive pages
for i in range(1,167): 
    # parse each page to obtain story links
    base_url = 'https://www.creepypasta.com/archive/?_page=' + str(i)
    base_page = requests.get(base_url)
    base_soup = bs(base_page.content, 'html.parser')
    link_list = base_soup.body.find_all('a', class_="_self cvplbd")
    
    # store story links
    story_urls = [item['href'] for item in link_list]
    
    # main parsing loop 
    for url in story_urls:
        
        # randomize sleep time to avoid bot detection
        sleep(randint(10,20))
        
        # parse story page
        page = requests.get(url)
        soup = bs(page.content, 'html.parser')

        # extract story title
        title = soup.find('h1').text

        # extract date of publication
        # skip date if story was deleted from archive
        try:
            publication_date = soup.find('span', class_='published').text.strip()
        except AttributeError:
            continue

        # extract story subgenres; set to None if missing
        try:
            subgenre = soup.find('span', class_='cat-links').text
        except AttributeError:
            subgenre = 'None'

        # extract story rating from the JSON metadata
        rate_text = soup.body.find('script', attrs={'type': 'application/ld+json'}).text
        idx = rate_text.find('ratingValue')
        rating = rate_text[idx+15:idx+19]

        # extract expected reading time
        est_read_time = soup.find('span', class_='rt-time').text.strip()

        # Organize story metadata 
        meta = []
        meta.append('Title: ' + title + '\n')
        meta.append('Date: ' + publication_date + '\n')
        meta.append('Subgenre: ' + subgenre + '\n')
        meta.append('Rating: ' + rating + '\n')
        meta.append('Reading Time: ' + est_read_time + '\n')

        # extract the story itself
        lines = []
        for para in soup.find_all('p'):
            lines.append(para.text)

        # Extracting the author for each story is difficult.
        # Early stories (~2008-2011) were written anonymously,
        # while newer stories are credited. However, the 
        # format for crediting authors is not standard across 
        # stories or time. Authorship is credited to user 
        # accounts in the JSON metadata, but names are given 
        # in the text bodies for many stories. As a result, the
        # decision was made to use the author data in the text, 
        # not the metadata. 
        # 
        # However, many authors are missing, and some are [redacted]
        # emails kept hidden for privacy concerns. The intent in
        # this script was to extract author data where available 
        # while defaulting to no author in missing and edge cases. 
        #
        # The format of author credit is variable over time. In the
        # block below, I've attempted to parse the specific format
        # used on many pages I reviewed, but these are not standard.
        
        author = ''

        # search the text for author data
        for line in lines:
            # if no credit given, leave author empty 
            if not re.search('credit|CREDIT|Credit|WRITTEN BY', line):
                continue
            else:
                # attempt to parse formatting; default to empty 
                try:
                    author = re.split('-|:|–', line)[1].strip()
                except IndexError:
                    try:
                        author = re.split('to|To', line)[1].strip()
                    except IndexError:
                        author = ''

        # merge metadata with story
        meta.append('Author: ' + author + '\n')
        lines = meta + lines

        # Many stories have out-of-context links to social media
        # and/or other extraneous advertising. There are also
        # frequent author's notes and other commentary about the
        # stories present in several stories. This information
        # should be removed to prevent NLP tasks from including 
        # junk data. 
        
        # skip final 2 lines of junk
        for line in lines[:-2]:
            # avoid writing author line to file
            if re.search('credit|CREDIT|Credit|WRITTEN BY', line):
                if line.find(author) != -1 and author != '':
                    continue
                else:
                    f.write(line)
            # avoid writing author's notes
            elif re.search("Publisher(’|')s Note|Author(’|')s Note", line):
                continue 
            # avoid writing procedural line
            elif re.search("This story was submitted", line):
                continue
            # avoid writing empty space
            elif line is not '\xa0' and line is not '':
                f.write(line)

        f.write('\n\n')
        
f.close()