import os
import glob
import re
import shutil

dirs = glob.glob( 'videos/*')

for dir in dirs:
	print(dir)
	dir_name=dir.split("/")[1]
	files = [ glob.glob(dir+'/*')]
	i=0
	for file in files[0]:
		print(file)
		file_name=file.split("/")[2]
		old_file = file
		if file_name.endswith('.webm'):
			
			new_file = os.path.join("videos",dir_name,str(i)+file_name[-5:])
		else:
			
			new_file = os.path.join("videos",dir_name,str(i)+file_name[-4:])
		os.rename(old_file, new_file)
#-----------------------audio part
		if file_name.endswith('.webm'):
			
			audio_file_name=file_name[:-5]+".mp3"
			audio_file=os.path.join("audio",dir_name,audio_file_name)
		else:
			audio_file_name=file_name[:-4]+".mp3"
			audio_file=os.path.join("audio",dir_name,audio_file_name)

		os.rename(audio_file,os.path.join("audio",dir_name,str(i)+".mp3"))
		i=i+1