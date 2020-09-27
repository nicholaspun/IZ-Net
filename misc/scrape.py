import argparse
import requests
from selenium import webdriver
import time
from urllib.parse import urljoin, urlparse

parser = argparse.ArgumentParser()
parser.add_argument('url')
args = parser.parse_args()

URL = args.url
driver = webdriver.Chrome(executable_path=r'./chromedriver.exe')
driver.get(URL)
time.sleep(5)
images = driver.find_elements_by_tag_name('img')

with open('seq') as seqfile:
    seq = int(seqfile.readline())

for index, img in enumerate(images):
    print('{}/{}'.format(index, len(images)))
    try:
        src = img.get_attribute('src')        
        image_url = src.split('?')[0]
        print(image_url)

        img_data = requests.get(image_url).content
        with open('{}.jpg'.format(seq), 'wb') as outfile:
            outfile.write(img_data)
        
        seq += 1
    except:
        continue

with open('seq', 'w') as seqfile:
    seqfile.write(str(seq))

driver.close()
