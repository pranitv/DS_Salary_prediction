# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 22:25:10 2021

@author: PRANIT
"""

from glassdoor_scrapper import get_jobs

path = 'C:/Users/PRANIT/Desktop/Coding/selenium/web_drivers/chromedriver.exe'

df = get_jobs('data Scientist',5,False,path,10)