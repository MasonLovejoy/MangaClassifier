# -*- coding: utf-8 -*-
"""

Author: Mason Lovejoy-Johnson
Contact Email: malovej2@ncsu.edu
Start Date: 1/12/2022
Last Update: 3/16/2022

The Goal of this program is to clean our files removing all non-jpgs while also 
removing the title cover and the last panels for clarity. This will allow for 
us to more directly call all images from a manga
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver

before_chapter_link = 'https//'
after_chapter_link  = '.com'

save_path =  'file path for where to save'
manga_panels = 'file path'
webdriver_path = 'file path to the downloaded webdriver'


def website_scraper(part1_url, part2_url, file_path, manga, initial_chapter, final_chapter, chromedriver):
    '''
    This function allows for the scraping of manga panels from most websites using selenium's webdriver
    module. I can then directly add whatever manga I want to its own folder allowing for the computer
    to download the panels on its own.

    Parameters
    ----------
    part1_url : string
        The part of the URL that comes before the chapter number. Allows for us to search through multiple
        chapters of a manga in a loop.
    part2_url : string
        The part fo the URL that comes after the chapter number.
    file_path : string
        The path where you want to download the manga images in. 
    manga : string
        The name of the manga currently being downloaded. It is used to create a file with the
        name of the manga as the saved image. 
    initial_chapter : Integer
        The first chapter you want to download panels from.
    final_chapter : Integer
        The last chapter you want to download panels from.
    chromedriver : String
        The file path that links to where the selenium webdriver is installed. This allows for 
        python to independently open its own chrome tab and scroll through it in order to load all the
        html that is required for some websites while scraping. 

    Returns
    -------
    None.

    '''
    for chapter in range(initial_chapter, final_chapter + 1): 
        
        url = part1_url + str(chapter) + part2_url
        print(chapter)
        
        driver = webdriver.Chrome(executable_path = chromedriver)
        driver.get(url)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        page_content = driver.page_source
        driver.quit()  
        
        soup = BeautifulSoup(page_content,'html.parser')
        img_tags = soup.find_all('img')
        urls = [img['src'] for img in img_tags]
        
        panel_idx = 0


        for url_idx in range(4, len(urls)-6):

            if manga in urls[url_idx]:
  
                download = requests.get(urls[url_idx])
                
                panel_idx += 1
                
                with open(file_path + manga + '_chapter' +str(chapter)+ '_' +str(panel_idx)+ '.jpg', 'wb') as file:
                    
                    file.write(download.content) 
    
    
    