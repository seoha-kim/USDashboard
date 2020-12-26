import requests
from datetime import datetime
import pandas as pd
import time
from selenium.webdriver import Chrome
import re
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

delay = 3
comment_data = pd.DataFrame({'youtube_id':[], 'comment':[], 'like_num':[]})

#chrome_options=options
browser = Chrome("E:\chromedriver_win32\chromedriver.exe")
browser.implicitly_wait(delay)
browser.maximize_window()

# youtube video url
start_url = "https://www.youtube.com/watch?v=5cathmZFeXs"
browser.get(start_url)
body = browser.find_element_by_tag_name('body')

time.sleep(2)

# page down until arrange button appear
num_page_down = 1
while num_page_down:
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(1.5)
    num_page_down -= 1

num_page_down = 1000
while num_page_down:
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(1.5)
    num_page_down -= 1
    if num_page_down % 100 == 0:
        print(num_page_down)

html_s0 = browser.page_source
html_s = BeautifulSoup(html_s0, 'html.parser')
comment0 = html_s.find_all('ytd-comment-renderer', {'class': 'style-scope ytd-comment-thread-renderer'})

for i in range(len(comment0)):
    # comment
    comment = comment0[i].find('yt-formatted-string',
                               {'id': 'content-text', 'class': 'style-scope ytd-comment-renderer'}).text
    try:
        like_num_ = comment0[i].find('span', {'id': 'vote-count-left'}).text
        like_num = "".join(re.findall('[0-9]', like_num_))
    except:
        like_num = 0

    youtube_id_ = comment0[i].find('a', {'id': 'author-text'}).find('span').text
    youtube_id = "".join(re.findall('[가-힣0-9a-zA-Z]', youtube_id_))

    insert_data = pd.DataFrame({'youtube_id': [youtube_id],
                                'comment': [comment],
                                'like_num': [like_num]})

    comment_data = comment_data.append(insert_data)

comment_data.index = range(len(comment_data))
comment_data.to_csv('data/nbc.csv', index=False)