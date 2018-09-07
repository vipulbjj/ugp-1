import json
import os
from pytube import YouTube

present_path = os.path.dirname(__file__)
with open ("MUSIC_solo_videos.json") as f:
    data=json.load(f)
not_downloaded=[]
classes=data['videos'].keys()
for class1 in classes:
    if not os.path.exists(os.path.join(present_path, class1)):
		os.makedirs(os.path.join(present_path, class1))
    videos=data['videos'][class1]
    for video in videos:

        print(video)
        try:
            YouTube("https://www.youtube.com/watch?v="+video).streams.first().download(os.path.join(present_path,class1))

        except:
            not_downloaded.append({class1:video})