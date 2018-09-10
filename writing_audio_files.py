import os
import glob
import re
import shutil
import pandas as pd
import gc
import moviepy.editor as mp

present_path = os.path.dirname(__file__)

processed_path = os.path.join(present_path, 'audio')

dirs = glob.glob( 'videos/*')
df = pd.DataFrame()
for dir in dirs:
    print(dir)
    dir_name=dir.split("/")[1]
    if not os.path.exists(os.path.join(present_path, 'audio', dir_name)):
		os.makedirs(os.path.join(present_path, 'audio', dir_name))
    files = [ glob.glob(dir+'/*')]
    for file in files[0]:
        print(file)
        file_name=file.split("/")[2]
        try:
            clip = mp.VideoFileClip(file)
            if file_name.endswith('.webm'):
                clip.audio.write_audiofile("/users/gpu/vipulbjj/NEWDATA_MOUNT/computer_vision/multimodal_alignment/music_datset_complete/audio"+"/"+dir_name+"/"+file_name[:-5]+".mp3")
            else:
                clip.audio.write_audiofile("/users/gpu/vipulbjj/NEWDATA_MOUNT/computer_vision/multimodal_alignment/music_datset_complete/audio"+"/"+dir_name+"/"+file_name[:-4]+".mp3")
        except:
            print("Cannot Read File !")
            df = df.append({'class': dir_name, 'video': file_name}, ignore_index=True)
        gc.collect()

df.to_csv("videos_left_audio_conversion.csv", sep='\t')   