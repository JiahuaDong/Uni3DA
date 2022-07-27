import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import h5py
import scipy.io as sio
from scipy.io import loadmat

datasets_path = './dataset/PointDA_data'
domain = 'modelnet'


data_path_pts = './dataset/shapenetcore_partanno_segmentation_benchmark_v0/04099429/points'

def showpoints(file):
	points = np.load(file)
	skip = 1
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	point_range = range(0, points.shape[0], skip)
	ax.scatter(points[point_range, 0],
			   points[point_range, 1],
			   points[point_range, 2],
			   c=points[point_range, 2],
			   cmap='spring',
			   marker=".")
	# ax.axis('equal')
	plt.show()


def showpoints_txt(file):
	data = [[], [], []]
	f = open(file, 'r')
	line = f.readline().strip('\n')
	num = 0
	while line:
		l0, l1, l2, l3, l4, l5 = line.split(',')
		data[0].append(l0)
		data[1].append(l1)
		data[2].append(l2)
		num = num + 1
		# data[0].append(l3)
		# data[1].append(l4)
		# data[2].append(l5)
		# num = num + 1
		line = f.readline().strip('.\n')
	f.close()
	x = [float(data[0]) for data[0] in data[0]]
	y = [float(data[1]) for data[1] in data[1]]
	z = [float(data[2]) for data[2] in data[2]]
	point = [x, y, z]
	points = np.array(point)
	points = points.transpose(1, 0)
	# np.save("dataset/modelnet_npy/airplane/airplane1", points)
	# points = np.load(file)
	# skip = 1
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# point_range = range(0, points.shape[0], skip)
	# ax.scatter(points[point_range, 0],
	# 		   points[point_range, 1],
	# 		   points[point_range, 2],
	# 		   c=points[point_range, 2],
	# 		   cmap='spring',
	# 		   marker=".")
	# ax.axis('equal')
	# plt.show()
	return points


def showpoints_pts(file):
	data = [[], [], []]
	f = open(file, 'r')
	line = f.readline().strip('\n')
	num = 0
	while line:
		l0, l1, l2= line.split(' ')
		data[0].append(l0)
		data[1].append(l1)
		data[2].append(l2)
		num = num + 1
		# data[0].append(l3)
		# data[1].append(l4)
		# data[2].append(l5)
		# num = num + 1
		line = f.readline().strip('.\n')
	f.close()
	x = [float(data[0]) for data[0] in data[0]]
	y = [float(data[1]) for data[1] in data[1]]
	z = [float(data[2]) for data[2] in data[2]]
	point = [x, y, z]
	points = np.array(point)
	points = points.transpose(1, 0)
	# points = np.load(file)
	# skip = 1
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# point_range = range(0, points.shape[0], skip)
	# ax.scatter(points[point_range, 0],
	# 		   points[point_range, 1],
	# 		   points[point_range, 2],
	# 		   c=points[point_range, 2],
	# 		   cmap='spring',
	# 		   marker=".")
	# ax.axis('equal')
	# plt.show()
	return points

def showpoints_mat(file):

	p = loadmat(file)
	points = p['points']
	for idx in range(points.shape[0]):
		data = [[], [], []]
		sample = points[idx]
		for point_idx in range(sample.shape[0]):
			single_point = sample[point_idx]
			data[0].append(single_point[0])
			data[1].append(single_point[1])
			data[2].append(single_point[2])
		x = [float(data[0]) for data[0] in data[0]]
		y = [float(data[1]) for data[1] in data[1]]
		z = [float(data[2]) for data[2] in data[2]]
		point = [x, y, z]
		pointss = np.array(point)
		pointss = pointss.transpose(1, 0)
		skip = 1
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		point_range = range(0, points.shape[0], skip)
		ax.scatter(pointss[point_range, 0],
				   pointss[point_range, 1],
				   pointss[point_range, 2],
				   c=pointss[point_range, 2],
				   cmap='spring',
				   marker=".")
		# ax.axis('equal')
		plt.show()




	# np.save("dataset/modelnet_npy/airplane/airplane1", points)
	# points = np.load(file)
	# skip = 1
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# point_range = range(0, points.shape[0], skip)
	# ax.scatter(points[point_range, 0],
	# 		   points[point_range, 1],
	# 		   points[point_range, 2],
	# 		   c=points[point_range, 2],
	# 		   cmap='spring',
	# 		   marker=".")
	# ax.axis('equal')
	# plt.show()
	return points

def load_dir(data_dir, name='train_files.txt'):
	with open(os.path.join(data_dir,name),'r') as f:
		lines = f.readlines()
	return [os.path.join(data_dir, line.rstrip().split('/')[-1]) for line in lines]


# data_path = './dataset/PointDA_data/modelnet/table/train'
#
# character_folders = os.listdir(data_path)
# index = 0
# for i in character_folders:
# 	file = os.path.join(data_path, i)
# 	showpoints(file)
# 	# if index % 10 == 0:
# 	# 	np.save("dataset/shapenet_npy/Rocket/test/" + i.split('.')[0], point_array)
# 	# else:
# 	# 	np.save("dataset/shapenet_npy/Rocket/train/" + i.split('.')[0], point_array)
# 	# index = index + 1
# 	plt.close()
# pc_root = "./dataset/PointDA_data/scannet"
# data_pth = load_dir(pc_root, name='test_files.txt')
# point_list = []
# label_list = []
# for pth in data_pth:
# 	data_file = h5py.File(pth, 'r')
# 	point = data_file['data'][:]
# 	label = data_file['label'][:]
#
# 	# idx = [index for index, value in enumerate(list(label)) if value in self.label_map]
# 	# point_new = point[idx]
# 	# label_new = np.array([self.label_map.index(value) for value in label[idx]])
#
# 	point_list.append(point)
# 	label_list.append(label)
# data = np.concatenate(point_list, axis=0)
# label = np.concatenate(label_list, axis=0)
#
# point_idx = np.arange(0, 1024)
# num = data.shape[0]
# class0 = []
# class1 = []
# class2 = []
# class3 = []
# class4 = []
# class5 = []
# class6 = []
# class7 = []
# class8 = []
# class9 = []
# for idx in range(num):
# 	if label[idx] == 0:
# 		class0.append(data[idx])
# 	if label[idx] == 1:
# 		class1.append(data[idx])
# 	if label[idx] == 2:
# 		class2.append(data[idx])
# 	if label[idx] == 3:
# 		class3.append(data[idx])
# 	if label[idx] == 4:
# 		class4.append(data[idx])
# 	if label[idx] == 5:
# 		class5.append(data[idx])
# 	if label[idx] == 6:
# 		class6.append(data[idx])
# 	if label[idx] == 7:
# 		class7.append(data[idx])
# 	if label[idx] == 8:
# 		class8.append(data[idx])
# 	if label[idx] == 9:
# 		class9.append(data[idx])
#
# class0 = np.array(class0)
# class1 = np.array(class1)
# class2 = np.array(class2)
# class3 = np.array(class3)
# class4 = np.array(class4)
# class5 = np.array(class5)
# class6 = np.array(class6)
# class7 = np.array(class7)
# class8 = np.array(class8)
# class9 = np.array(class9)
#
# sio.savemat('scan_class0.mat', {"points": class0})
# sio.savemat('scan_class1.mat', {"points": class1})
# sio.savemat('scan_class2.mat', {"points": class2})
# sio.savemat('scan_class3.mat', {"points": class3})
# sio.savemat('scan_class4.mat', {"points": class4})
# sio.savemat('scan_class5.mat', {"points": class5})
# sio.savemat('scan_class6.mat', {"points": class6})
# sio.savemat('scan_class7.mat', {"points": class7})
# sio.savemat('scan_class8.mat', {"points": class8})
# sio.savemat('scan_class9.mat', {"points": class9})
	# point = data[idx][point_idx][:, :3]
	# label_s = label[idx]

data_path = './dataset/PointDA_data/scannet/train_scan/'

character_folders = os.listdir(data_path)
index = 0
for i in character_folders:
	file = './dataset/PointDA_data/scannet/train_scan/scan_class0.mat' #os.path.join(data_path, i)
	showpoints_mat(file)
	plt.close()