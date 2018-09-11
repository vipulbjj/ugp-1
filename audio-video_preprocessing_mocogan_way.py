import os
import glob
import re
import shutil
import pandas as pd
import skvideo.io
import glob
#from matplotlib import pyplot as plt
import numpy as np
import scipy.misc
import scipy.io.wavfile

import subprocess
import librosa
import numpy as np
import time
import gc

if not os.path.exists("audio_downsampled"):
	os.makedirs("audio_downsampled")
pre_outputfolder='audio_downsampled/'

def downsample_16k(input_filepath,output_folderpath,filename):
	"""
	Convert all audio files to have sampling rate 16k.
	"""
	print('this is input path')
	print(input_filepath)
	print('this is output path')
	print(output_folderpath)
	print('Downsampling : {}'.format(input_filepath))
	completed_process = subprocess.run(
						'sox {} -r 16k {}'
						.format(input_filepath, os.path.join(output_folderpath, filename)),
						shell=True, check=True)


present_path = os.path.dirname(__file__)

processed_path_audio  = os.path.join(present_path, 'audio_processed_mocogan_way')

processed_path_video  = os.path.join(present_path, 'video_processed_mocogan_way')

dirs = glob.glob( 'videos/*')

for dir in dirs:
	print(dir)
	dir_name=dir.split("/")[1]
	if not os.path.exists(os.path.join(processed_path_audio, dir_name)):
		os.makedirs(os.path.join(processed_path_audio, dir_name))
	if not os.path.exists(os.path.join(processed_path_video, dir_name)):
		os.makedirs(os.path.join(processed_path_video, dir_name))
	files = [ glob.glob(dir+'/*')]
	j=-1
	k=-1
	for file in files[0]:
		print(file)
		file_name=file.split("/")[2]
		print("loading video")
		videodata=skvideo.io.vread(file)
		print("loaded video")
		b=videodata.shape
		#temp=[]
		
		for i in range(0,b[0]):
			img=videodata[i]
			#img_crop=img[y1:y2,x1:x2]
			img_crop=scipy.misc.imresize(img,(64,64))
			#print('ii',i)
			if (i)%30==0 or (i==(b[0]-1)and b[0]%30==0):
				if i!=0:

					j=j+1
					#print("i,j",i,j,b[0])
					scipy.misc.imsave(os.path.join(processed_path_video,dir_name,"img_%d.jpg"%(j)),data)
					#scipy.misc.imsave("img_%d.jpg"%(j),data)
				data=None
				data = img_crop
			else:
				data = np.concatenate([data,img_crop])

	#gc.collect()
			#video done
		print("video done----audio start")
#-------------------------------------------------------------------------------------------------------------
			#audio start
		if file_name.endswith('.webm'):
			
			audio_file_name=file_name[:-5]+".mp3"
			audio_file=os.path.join("audio",dir_name,audio_file_name)
		else:
			audio_file_name=file_name[:-4]+".mp3"
			audio_file=os.path.join("audio",dir_name,audio_file_name)

		downsample_16k(audio_file,pre_outputfolder,audio_file_name)
		sample_rate = 16000
		filepath=os.path.join(pre_outputfolder, audio_file_name)
		print("audio loading")
		wav, sr = librosa.load(filepath, sr=sample_rate)
		print("audio loaded")
		n_samples = wav.shape[0]
		n_audio_files=int(n_samples/16000)
		
		for i in range(n_audio_files):
			start_idx=i*16000
			end_idx=start_idx+16384
			slice_sig = wav[start_idx:end_idx]
			k=k+1
			clip_name="img_"+str(k)+".wav"
			#k=k+1
			#print(i)
			scipy.io.wavfile.write(os.path.join(processed_path_audio,dir_name,clip_name), sample_rate, slice_sig)
		if(k!=j):
			print("------------------------------anomaly------------------------------------------")
			if(k>j):
				print(k,j)
				print("------------------------------anomaly------------audio>video-----------------------------------------")
				os.remove(os.path.join(processed_path_audio,dir_name,clip_name))
				k=k-1
			else:
				print(k,j)
				print("------------------------------anomaly------------video>audio-----------------------------------------")
				os.remove(os.path.join(processed_path_video,dir_name,"img_%d.jpg"%(j)))
				j=j-1
		print("audio completed")
		os.remove(filepath)
		gc.collect()
os.rmdir('audio_downsampled')
