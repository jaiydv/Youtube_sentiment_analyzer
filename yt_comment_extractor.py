from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

import demo


def fetch_comments(video_url,total_comments):
    # Create a new instance of the Chrome driver
    # driver = webdriver.Chrome()
    driver = webdriver.Chrome(executable_path='chromedriver-win64\chromedriver-win64\chromedriver.exe')
    
    # Open the YouTube video URL
    driver.get(video_url)
    
    # Wait for the comments section to load
    time.sleep(5)  # You may need to adjust this depending on your internet speed and the page loading time
    
    # Scroll down to load more comments
    body = driver.find_element_by_tag_name('body')
    load_time = int(total_comments/10)
    
    if load_time <1:
        load_time=1

    for _ in range(load_time):  # Increase the range if you want to load more comments
        body.send_keys(Keys.END)
        time.sleep(2)  # Wait for comments to load
        
    
    # Extract comments from the loaded HTML content
    comments = driver.find_elements_by_xpath('//*[@id="content-text"]')
    comments_text = [comment.text for comment in comments]
    
    # Close the browser
    driver.quit()
    
    return comments_text

def yt_run(video_url,total_comments):

    # video_url = input("Enter the YouTube video URL: ")


    comments = fetch_comments(video_url,total_comments)
    
    print("Comments extracted successfully:")
    
    demo.inference(comments[:total_comments])
    
    print("total comment extracted: {}".format(len(comments)))

    # demo.inference(comments)
