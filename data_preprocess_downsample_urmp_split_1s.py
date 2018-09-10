import os
import glob
import re
import shutil
import numpy as np
import scipy.misc
import scipy.io.wavfile
import os
import subprocess
import librosa
import numpy as np
import time
print('Warnnig: plz delete the last file form the Tuba and Horn (they have 255 but should have 254 bcoz video clips =254')
# current_path = os.path.dirname(__file__)
present_path=os.path.dirname(__file__)
dirs = glob.glob(os.path.join(present_path, 'video_raw/*'))
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


#j=0
#print(dirs)
for dir in dirs:
	dir_name=dir.split("/")[1]
	file=glob.glob(dir+'/*.wav')
	#print(file)
	filename=file[0].split("/")[2]
	downsample_16k(file[0],pre_outputfolder,filename)
	sample_rate = 16000
	filepath=os.path.join(pre_outputfolder, filename)
	wav, sr = librosa.load(filepath, sr=sample_rate)
	n_samples = wav.shape[0]
	n_audio_files=int(n_samples/16000)
	if not os.path.exists(os.path.join("audio_processed/", dir_name)):
		os.makedirs(os.path.join("audio_processed/", dir_name))
	for i in range(n_audio_files):
		start_idx=i*16000
		end_idx=start_idx+16384
		slice_sig = wav[start_idx:end_idx]
		clip_name="img_"+str(i)+".wav"

		print(i)
		scipy.io.wavfile.write(os.path.join("audio_processed/"+str(dir_name),clip_name), sample_rate, slice_sig)
	os.remove(filepath)
os.rmdir('audio_downsampled')
print('Warnnig: plz delete the last file form the Tuba and Horn (they have 255 but should have 254 bcoz video clips =254')



	#print(len(speakers))
	# for speaker in speakers:
	# 	files = [ glob.glob(speaker+'/*')]
	# 	num_images=(len(files[0]))
	# 	num_videos=num_images/20
	# 	speaker_name=speaker.split("/")[2]
	# 	#data=None
	# 	check_empty=1
	# 	i=0
	# 	sorted_files=sorted(files[0])
	# 	for file in sorted_files:#sort files[0]

	# 		#file_name=file.split("/")[3]
	# 		print('outputfolder',outputfolder)
	# 		print('dir',dir)
	# 		print('speaker',speaker)
	# 		print('file',file)
	# 		print('Done')
	# 		downsampled_file=downsample_16k(file,str(outputfolder)+file)
	# 		sample_rate,audio_numpy=scipy.io.wavfile.read(str(outputfolder)+file)
	# 		# img_numpy=scipy.misc.imresize(img_numpy,(64,64))
	# 		#print(audio_numpy.shape)

	# 		if i%20==0 :
	# 			if i!=0:
	# 				# print(data.shape)
	# 				# horizontally_stacked_array=np.hstack(data)
	# 				# print(horizontally_stacked_array.shape)
	# 				j=j+1
	# 				#scipy.misc.imsave(processed_path+"/img_h_%d.png"%(j),data)
	# 				#print(data.shape)
	# 				np.save(processed_path+"/audio_%d.npy"%(j), arr=data)
	# 				print(str(i/20)+"of"+str(num_videos)+"---"+str(dir_name)+"---"+str(speaker_name))
	# 			data=None
	# 			data = audio_numpy



	# 		else:

	# 		#print(img.shape)
	# 		#print(data.shape)
	# 			data = np.concatenate([data,audio_numpy])

	# 		i=i+1
