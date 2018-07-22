#--Import libraries
from bs4 import BeautifulSoup
import os
import pandas as pd
from modules import collectHTML as collect
from modules import extractHTML as extract


#--Switches
#PRODUCE = True to scrape
#PRODUCE = False to produce df from html
PRODUCE = False
EXTRACT = not PRODUCE

source = 'GameSpot'
HTMLfldr = '../data/raw/'
# source = 'GamesRadar'
# HTMLfldr = '../data/raw_gamesradar/'
# source = 'Polygon'
# HTMLfldr = '../data/raw_polygon/'

datafldr = '../data/'


#--Produce HTML files
if PRODUCE:
    reviewLinks = collect.getURL(1, source)
    for reviewLink in reviewLinks: collect.getHTML(reviewLink, source)
    reviewLinks = collect.getURL(2, source)
    for reviewLink in reviewLinks: collect.getHTML(reviewLink, source)


#--Make soup, extract, and save as df
if EXTRACT:
    #Make soup of all files in a specified folder
    soupCauldron = []
    def makeSoups(fldr):
        for filename in os.listdir(fldr):
            content = open(fldr + filename, 'r', errors='replace', encoding='utf-8')
            soup = BeautifulSoup(content.read(), 'html5lib')
            yield (soup, filename)

    soupCauldron = makeSoups(HTMLfldr)


    # Extract data from each HTML file and combine
    if source == 'GameSpot':
        def getContent():
            for soup, filename in soupCauldron:
                df_main, df_platform, df_developer, df_publisher, df_genre = extract.extractSoup(soup, filename, 'GameSpot')
                yield (df_main, df_platform, df_developer, df_publisher, df_genre)
        
        df_cb_main      = pd.DataFrame()
        df_cb_platform  = pd.DataFrame()
        df_cb_developer = pd.DataFrame()
        df_cb_publisher = pd.DataFrame()
        df_cb_genre     = pd.DataFrame()

        for content in getContent():
            df_cb_main      = pd.concat([content[0], df_cb_main], ignore_index = True)
            df_cb_platform  = pd.concat([content[1], df_cb_platform], ignore_index = True)
            df_cb_developer = pd.concat([content[2], df_cb_developer], ignore_index = True)
            df_cb_publisher = pd.concat([content[3], df_cb_publisher], ignore_index = True)
            df_cb_genre     = pd.concat([content[4], df_cb_genre], ignore_index = True)

        # Save data for future analysis
        df_cb_main.to_csv(datafldr + 'df_cb_main.csv')
        df_cb_platform.to_csv(datafldr + 'df_cb_platform.csv')
        df_cb_developer.to_csv(datafldr + 'df_cb_developer.csv')
        df_cb_publisher.to_csv(datafldr + 'df_cb_publisher.csv')
        df_cb_genre.to_csv(datafldr + 'df_cb_genre.csv')


    else:
        df_cb_main = []
        for soup, filename in soupCauldron:
            df_main = extract.extractSoup(soup, filename, source)
            df_cb_main.append(df_main)

        df_cb_main = pd.concat(df_cb_main, ignore_index = True)
        #Save data for future analysis
        df_cb_main.to_csv(datafldr + 'df_cb_main_' + source + '.csv')
