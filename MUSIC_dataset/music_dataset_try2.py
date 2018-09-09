from __future__ import unicode_literals
import re
import glob

import youtube_dl
import urllib
import shutil
import pandas as pd
import os

text_file = open("videos_left.txt", "r")
lines = text_file.readlines()
present_path = os.path.dirname('__file__')
df = pd.DataFrame()
for i in range(len(lines)):
    text=lines[i]
    m = re.findall("u\'(.+?)\'", text)
    class1=m[0]
    video=m[1]
    print(class1,video)
    try:
            #YouTube("https://www.youtube.com/watch?v="+video).streams.first().download(os.path.join(present_path,class1))
            ydl_opts = {}
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download(['https://www.youtube.com/watch?v='+video])
            file1 = glob.glob("*.mp4")
            print("hello------------------------------------------------------------------------------")
            
            if(file1):
                destination_folder = os.path.join(present_path,class1,file1[0])
                print('11111111111111111111111111111111111111111111111111111111111111111111111111')
                shutil.move(file1[0], destination_folder)
            elif(glob.glob("*.mkv")):
                print('22222222222222222222222222222222222222222222222222222222222222222222222222222')
                file1=glob.glob("*.mkv")
                destination_folder = os.path.join(present_path,class1,file1[0])
                shutil.move(file1[0], destination_folder)
            else:
                print('333333333333333333333333333333333333333333333333333333333333333333333333333')
                file1=glob.glob("*.webm")
                destination_folder = os.path.join(present_path,class1,file1[0])
                shutil.move(file1[0], destination_folder)
    except:
            #not_downloaded.append({class1:video})
            df = df.append({'class': class1, 'video': video}, ignore_index=True)

with open('videos_left_2nd_time.txt', 'w+') as f:
    for item in not_downloaded:
        f.write("%s\n" % item)
df.to_csv("videos_left_2nd_time.csv", sep='\t')        
       # df = pd.DataFrame()
#df = df.append({'name': 'Zed', 'age': 9, 'height': 2}, ignore_index=True)