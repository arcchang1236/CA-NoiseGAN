import os
import pandas as pd 

from collections import Counter


a = pd.read_excel('NR.xlsx', sheet_name=None)['Sheet1']

a = a.sort_values("name")

kk = ['clean', 'g', 'nf', 'noisy', 'ours', 'pg', 'real', 'real_ours']

cam_dict = {'Pixel-Google-google': 0, 'iPhone9,3 back camera': 1, 'SM-G925I-samsung-samsung': 2, 'Nexus 6-motorola-google': 3, 'LG-H815-LGE-lge': 4}

data_root = 'NR_output'

cnt = {}

s = []
for idx, row in a.iterrows():
	ss = '%d%d%04d' % (cam_dict[row['cam']], row['scene'], row['iso'])
	s.append(ss)

b = Counter(s)
#print(b)


ddd = []

for idx, row in a.iterrows():
	#print(row['cam'])
	ss = '%d%d%04d' % (cam_dict[row['cam']], row['scene'], row['iso'])
	cnt = b[ss]
	b[ss] -= 1
	for i in kk:
		png_file = '%06d_%s.jpg' % (row['name'], i)
		if i == 'real_ours':
			i = 'realours'
		new_name = '%d_%d_%04d_%02d_%s.jpg' % (cam_dict[row['cam']], row['scene'], row['iso'], cnt, i)
		os.rename(os.path.join(data_root, png_file), os.path.join(data_root, new_name))
