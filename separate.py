import scipy.io
import os
import shutil
import splitfolders

mat = scipy.io.loadmat('imagelabels.mat')
i = 0

for filename in sorted(os.listdir('./jpg')):
	x = mat['labels'][0][i]
	if not os.path.exists(f'./images/{x}'):
		os.makedirs(f'./images/{x}')
	shutil.move(f'./jpg/{filename}', f'./images/{x}/{filename}')
	i += 1

splitfolders.ratio('./images', output='split', seed=42, ratio=(0.6, 0.2, 0.2))