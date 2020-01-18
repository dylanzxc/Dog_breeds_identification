import os
import shutil
import struct
import pandas as pd
from sklearn.utils import shuffle

# loading the data

#$ wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
#and once downloaded...
#$ tar xvf images.tar

#This will create a folder called Images

path = 'Images'
os.mkdir(path + '/train_new')
os.mkdir(path + '/test_new')

breeds = []
ids = []
images_0 = []
images_1 = []

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
	for i in d:
		if '-' in i:
			ids.append(i.split('-')[0]) 
			breeds.append(i.split('-')[1])
			inner_path = path + '/' + i
			for r, d, f in os.walk(inner_path):
				for j in f:
					shutil.move(inner_path + '/' + j, path + '/train_new/' + j)
					images_0.append(j.split('_')[0])
					images_1.append(j)




df_breeds = pd.DataFrame(breeds, columns=['breed_name'])
df_breeds['breed'] = ids

df_imgs = pd.DataFrame(images_0, columns=['breed'])
df_imgs['id'] = images_1
print(df_breeds)
print(df_imgs)
df_both = pd.merge(df_imgs, df_breeds, on='breed', how='left')
df_shuffle = shuffle(df_both)
print(df_shuffle)

df_shuffle.to_csv('labels_stanford.csv')


