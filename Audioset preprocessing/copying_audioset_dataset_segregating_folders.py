import numpy as np
import pandas as pd
import json

import os
import shutil


def num(file):
			if(file<100000):
					return "ubt_0l/audioset"
			elif(file<200000):
					return "ubt_1l"
			elif(file<300000):
					return "ubt_2l"
			elif(file<400000):
					return "ubt_3l"
			elif(file<600000):
					return "ubt_4l"
			elif(file<900000):
					return "ubt_6l"
			elif(file<1100000):
					return "ubt_9l"
			elif(file<1200000):
					return "ubt_11l"
			elif(file<1500000):
					return "ubt_12l"
			elif(file<1900000):
					return "ubt_15l"
			else:
					return "ubt_19l"
				
    	
print ("hello")

with open("mappings.json") as f:
    data = json.load(f)

reversed_dictionary = dict(map(reversed, data.items()))

df = pd.read_csv("unbalanced_train_segments.csv", sep=' ', header=None)
df_ids=df[3]

list_imp=[]#put in guitar videos

print("loaded both files")

for name,code in reversed_dictionary.iteritems():
	for i in range(2041789):#no.of videos
		df_i_list=df_ids.iloc[i].split(",")
		
		if code in df_i_list:#guitar videos
			list_imp.append(i)



	# list_imp_0l=[]#put in guitar videos upto 1l
	# for item in list_imp:
	# 	if item<100000:
	# 		list_imp_0l.append(str(item)+".mp4")

	total=len(list_imp)

	print (total)

	
	#src_files = os.listdir(src)
	count=0
	for file_name in list_imp: 
			print(file_name)
			src="/new_data/gpu/raviteja/l3-net/datasets/audioset/unbalanced_train/"+num(file_name)
			if(num(file_name)=='ubt_4l'):
				src="/users/gpu/vipulbjj/NEWDATA_MOUNT/computer_vision/multimodal_alignment/Audioset_guitar_videos/ubt_4l/audioset2/"
			if(num(file_name)=='ubt_11l'):
				src="/users/gpu/vipulbjj/NEWDATA_MOUNT/computer_vision/multimodal_alignment/Audioset_guitar_videos/ubt_11l/ubt11/"
			count=count+1;
			full_file_name = os.path.join(src, str(file_name)+".mp4")
			print(full_file_name)
			if (os.path.isfile(full_file_name)):
				print(str(count)+"of"+str(total) )
				dest="/users/gpu/vipulbjj/NEWDATA_MOUNT/computer_vision/multimodal_alignment/Audioset_guitar_videos/audioset_segregated/"+name
				if not os.path.exists(dest):
					os.makedirs(dest)
				shutil.copy(full_file_name, dest+"/")
			else:
				print("not found")