import os
import glob
import re
import shutil


print('Warnnig: plz delete the last file form the Tuba and Horn (they have 255 but should have 254 bcoz video clips =254')

current_path = os.path.dirname(__file__)#NU why it's giving blank?
#print(current_path)
processed_path = os.path.join(current_path, 'audio_combined/')
dirs = glob.glob(os.path.join(current_path, 'audio_processed/*'))
if not os.path.exists("audio_combined"):
	os.makedirs("audio_combined")
#print(len(dirs))
#got a list of directories
i=0
for dir in dirs:

	# dir_name=str(dir-"img/train"
	dir_name=dir.split("/")[1]
	files = [ glob.glob(dir+'/*')]
	#print(len(files[0]))
	#str_req= dir_name+'(.+?)_'
	sorted_files=sorted(files[0])
	for j in range(len(files[0])):
			#print(file)
			print(j)
			src="audio_processed/"+dir_name+"/img_"+str(j)+".wav"
			print("------------")
			dest=processed_path+"img_"+str(i)+".wav"
			print(src)
			os.rename(src,dest)
			i=i+1
import shutil
shutil.rmtree('audio_processed')
print('Warnnig: plz delete the last file form the Tuba and Horn (they have 255 but should have 254 bcoz video clips =254')
		# file_name=file.split("/")[3]#str
        # present_path="/home/vinodkk/Codes/Dataset/Sub-URMP/"
        # full_file_name = os.path.join(present_path, file)
		# 	#print(full_file_name)
		# 	dest_folder="/home/vinodkk/Codes/Dataset/Sub-URMP/processed_data/"
		# 	dest_path=os.path.join(dest_folder,dir_name, speaker_num)
		# 	#print(dest_path)
		# 	if (os.path.isfile(full_file_name)):
		# 		#print("file_present")
		# 		shutil.copy(full_file_name,dest_path)
		# 		print("Copying")
		#print(file_name.type)
		# str_req= dir_name+'(.+?)_'
		# m = re.search(str_req, file_name)
		# text = 'gfgfdAAA1234ZZZuijjk'

		# m = re.search('AAA(.+?)ZZZ', text)
		# present_path="/home/vinodkk/Codes/Dataset/Sub-URMP/"

		# if m:

		# 	speaker_num=m.group(1)

		# 	if not os.path.exists(os.path.join(processed_path,dir_name, speaker_num)):

	    #    			os.makedirs(os.path.join(processed_path,dir_name, speaker_num))

		# 	full_file_name = os.path.join(present_path, file)
		# 	#print(full_file_name)
		# 	dest_folder="/home/vinodkk/Codes/Dataset/Sub-URMP/processed_data/"
		# 	dest_path=os.path.join(dest_folder,dir_name, speaker_num)
		# 	#print(dest_path)
		# 	if (os.path.isfile(full_file_name)):
		# 		#print("file_present")
		# 		shutil.copy(full_file_name,dest_path)
		# 		print("Copying")





	#print(file_name.type)

#print((files))
	# for file in files:
	# 	# file_name=file.split("/")[3]
	# 	# file_name_chars=list(str(file_name))
	# 	print(len(files))
	# #print(files)
	# while True:
	# 	i=00





	# 	i++




# files = [ glob.glob(dir+'/*') for dir in dirs ]
# print(files.shape)
# files = sum(files, []) # flatten
