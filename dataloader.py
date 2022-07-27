import torch
import torch.utils.data as data
import os
import sys
import h5py
import numpy as np
import glob
import random
from data_utils import *
from scipy.io import loadmat

label_tabel_s = {'bathtub': 0, 'cabinet': 1, 'lamp': 2, 'plant': 3, 'table': 4,
				 'bookshelf': 5, 'chair': 6, 'monitor': 7, 'sofa': 8, 'bed': 9,
				 'stairs': 20, 'door': 20,
				 'Bag': 20, 'Rocket': 20}

label_tabel_t = {'bathtub': 0, 'cabinet': 1, 'lamp': 2, 'plant': 3, 'table': 4,
				 'bookshelf': 10, 'chair': 11, 'monitor': 12, 'sofa': 13, 'bed': 14,
				 'stairs': 20, 'door': 20,
				 'Bag': 20, 'Rocket': 20}


def load_dir(data_dir, name='train_files.txt'):
	with open(os.path.join(data_dir, name), 'r') as f:
		lines = f.readlines()
	return [os.path.join(data_dir, line.rstrip().split('/')[-1]) for line in lines]


def get_info(shapes_dir, isView=False):
	names_dict = {}
	if isView:
		for shape_dir in shapes_dir:
			name = '_'.join(os.path.split(shape_dir)[1].split('.')[0].split('_')[:-1])
			if name in names_dict:
				names_dict[name].append(shape_dir)
			else:
				names_dict[name] = [shape_dir]
	else:
		for shape_dir in shapes_dir:
			name = os.path.split(shape_dir)[1].split('.')[0]
			names_dict[name] = shape_dir

	return names_dict


class Modelnet40_data1(data.Dataset):
	def __init__(self, pc_root, status='train', pc_input_num=1024, aug=True):
		super(Modelnet40_data1, self).__init__()

		self.status = status
		self.pc_list = []
		self.lbl_list = []
		self.pc_input_num = pc_input_num
		self.aug = aug

		categorys = glob.glob(os.path.join(pc_root, '*'))
		categorys = [c.split(os.path.sep)[-1] for c in categorys]
		# sorted(categorys)
		categorys = sorted(categorys)

		if status == 'train':
			npy_list = glob.glob(os.path.join(pc_root, '*', 'train', '*.npy'))
		else:
			npy_list = glob.glob(os.path.join(pc_root, '*', 'test', '*.npy'))
		# names_dict = get_info(npy_list, isView=False)

		for _dir in npy_list:
			self.pc_list.append(_dir)
			self.lbl_list.append(categorys.index(_dir.split('/')[-3]))

		print(f'{status} data num: {len(self.pc_list)}')

	def __getitem__(self, idx):
		lbl = self.lbl_list[idx]
		pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)
		pc = normal_pc(pc)
		if self.aug:
			pc = rotation_point_cloud(pc)
			pc = jitter_point_cloud(pc)
		# print(pc.shape)
		pc = np.expand_dims(pc.transpose(), axis=2)
		return torch.from_numpy(pc).type(torch.FloatTensor), lbl

	def __len__(self):
		return len(self.pc_list)


class Modelnet40_data(data.Dataset):
	def __init__(self, pc_root, class_list, status='train', pc_input_num=1024, aug=True, dtype=None):
		super(Modelnet40_data, self).__init__()

		self.status = status
		self.pc_list = []
		self.lbl_list = []
		self.pc_input_num = pc_input_num
		self.aug = aug

		categorys = class_list

		if status == 'train':
			if dtype == 'source':
				plist = './dataset/Uni3DA_data/modelnet_train_s.txt'
			else:
				plist = './dataset/Uni3DA_data/modelnet_train_t.txt'
			with open(plist, 'r') as f:
				data = [[line.split()[0], line.split()[1] if len(line.split()) > 1 else '0'] for line in f.readlines()
						if line.strip()]  # avoid empty lines
				self.datas = [x[0] for x in data]
				try:
					self.labels = [int(x[1]) for x in data]
				except ValueError as e:
					print('invalid label number, maybe there is a space in the image path?')
					raise e
			ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if y in categorys]
			self.pc_list, self.lbl_list = zip(*ans)
		elif status == 'test':
			if dtype == 'source':
				plist = './dataset/Uni3DA_data/modelnet_test_s.txt'
			else:
				plist = './dataset/Uni3DA_data/modelnet_test_t.txt'
			with open(plist, 'r') as f:
				data = [[line.split()[0], line.split()[1] if len(line.split()) > 1 else '0'] for line in f.readlines()
						if line.strip()]  # avoid empty lines
				self.datas = [x[0] for x in data]
				try:
					self.labels = [int(x[1]) for x in data]
				except ValueError as e:
					print('invalid label number, maybe there is a space in the image path?')
					raise e
			ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if y in categorys]
			self.pc_list, self.lbl_list = zip(*ans)
		print(f'{status} data num: {len(self.pc_list)}')

	def __getitem__(self, idx):
		lbl = self.lbl_list[idx]
		pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)
		pc = normal_pc(pc)
		if self.aug:
			pc = rotation_point_cloud(pc)
			pc = jitter_point_cloud(pc)
		# print(pc.shape)
		pc = np.expand_dims(pc.transpose(), axis=2)
		return torch.from_numpy(pc).type(torch.FloatTensor), lbl, idx

	def __len__(self):
		return len(self.pc_list)


class Shapenet_data1(data.Dataset):
	def __init__(self, pc_root, status='train', pc_input_num=1024, aug=True, data_type='*.npy'):
		super(Shapenet_data1, self).__init__()

		self.status = status
		self.pc_list = []
		self.lbl_list = []
		self.pc_input_num = pc_input_num
		self.aug = aug
		self.data_type = data_type

		categorys = glob.glob(os.path.join(pc_root, '*'))
		categorys = [c.split(os.path.sep)[-1] for c in categorys]
		# sorted(categorys)
		categorys = sorted(categorys)

		if status == 'train':
			pts_list = glob.glob(os.path.join(pc_root, '*', 'train', self.data_type))
		elif status == 'test':
			pts_list = glob.glob(os.path.join(pc_root, '*', 'test', self.data_type))
		else:
			pts_list = glob.glob(os.path.join(pc_root, '*', 'validation', self.data_type))
		# names_dict = get_info(pts_list, isView=False)

		for _dir in pts_list:
			self.pc_list.append(_dir)
			self.lbl_list.append(categorys.index(_dir.split('/')[-3]))

		print(f'{status} data num: {len(self.pc_list)}')

	def __getitem__(self, idx):
		lbl = self.lbl_list[idx]
		if self.data_type == '*.pts':
			pc = np.array([[float(value) for value in xyz.split(' ')]
						   for xyz in open(self.pc_list[idx], 'r') if len(xyz.split(' ')) == 3])[:self.pc_input_num, :]
		elif self.data_type == '*.npy':
			pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)
		pc = normal_pc(pc)
		if self.aug:
			pc = rotation_point_cloud(pc)
			pc = jitter_point_cloud(pc)
		pad_pc = np.zeros(shape=(self.pc_input_num - pc.shape[0], 3), dtype=float)
		pc = np.concatenate((pc, pad_pc), axis=0)
		pc = np.expand_dims(pc.transpose(), axis=2)
		return torch.from_numpy(pc).type(torch.FloatTensor), lbl

	def __len__(self):
		return len(self.pc_list)


class Shapenet_data(data.Dataset):
	def __init__(self, pc_root, class_list, status='train', pc_input_num=1024, aug=True, dtype=None, data_type='*.npy'):
		super(Shapenet_data, self).__init__()

		self.status = status
		self.pc_list = []
		self.lbl_list = []
		self.pc_input_num = pc_input_num
		self.aug = aug
		self.data_type = data_type

		categorys = class_list

		if status == 'train':
			if dtype == 'source':
				plist = './dataset/Uni3DA_data/shapenet_train_s.txt'
			else:
				plist = './dataset/Uni3DA_data/shapenet_train_t.txt'
			with open(plist, 'r') as f:
				data = [[line.split()[0], line.split()[1] if len(line.split()) > 1 else '0'] for line in f.readlines()
						if line.strip()]  # avoid empty lines
				self.datas = [x[0] for x in data]
				try:
					self.labels = [int(x[1]) for x in data]
				except ValueError as e:
					print('invalid label number, maybe there is a space in the image path?')
					raise e
			ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if y in categorys]
			self.pc_list, self.lbl_list = zip(*ans)
		elif status == 'test':
			if dtype == 'source':
				plist = './dataset/Uni3DA_data/shapenet_test_s.txt'
			else:
				plist = './dataset/Uni3DA_data/shapenet_test_t.txt'
			with open(plist, 'r') as f:
				data = [[line.split()[0], line.split()[1] if len(line.split()) > 1 else '0'] for line in f.readlines()
						if line.strip()]  # avoid empty lines
				self.datas = [x[0] for x in data]
				try:
					self.labels = [int(x[1]) for x in data]
				except ValueError as e:
					print('invalid label number, maybe there is a space in the image path?')
					raise e
			ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if y in categorys]
			self.pc_list, self.lbl_list = zip(*ans)
		print(f'{status} data num: {len(self.pc_list)}')

	def __getitem__(self, idx):
		lbl = self.lbl_list[idx]
		if self.data_type == '*.pts':
			pc = np.array([[float(value) for value in xyz.split(' ')]
						   for xyz in open(self.pc_list[idx], 'r') if len(xyz.split(' ')) == 3])[:self.pc_input_num, :]
		elif self.data_type == '*.npy':
			# print(self.pc_list[idx])
			pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)

		pc = normal_pc(pc)
		if self.aug:
			pc = rotation_point_cloud(pc)
			pc = jitter_point_cloud(pc)
		pad_pc = np.zeros(shape=(self.pc_input_num - pc.shape[0], 3), dtype=float)
		pc = np.concatenate((pc, pad_pc), axis=0)
		pc = np.expand_dims(pc.transpose(), axis=2)
		return torch.from_numpy(pc).type(torch.FloatTensor), lbl, idx

	def __len__(self):
		return len(self.pc_list)


class Scannet_data_h5(data.Dataset):

	def __init__(self, pc_root, class_list, status='train', pc_input_num=1024, aug=True, dtype=None):
		super(Scannet_data_h5, self).__init__()
		self.num_points = pc_input_num
		self.status = status
		self.aug = aug
		# self.label_map = [2, 3, 4, 5, 6, 7, 9, 10, 14, 16]

		if self.status == 'train':
			pc_root = './dataset/Uni3DA_data/scannet/train_scan'
			data_pth = os.listdir(pc_root)
		else:
			pc_root = './dataset/Uni3DA_data/scannet/test_scan'
			data_pth = os.listdir(pc_root)

		point_list = []
		label_list = []

		if dtype == 'source':
			label_tabel = label_tabel_s
		else:
			label_tabel = label_tabel_t

		for pth in data_pth:
			data_file = loadmat(os.path.join(pc_root, pth))
			point = data_file['points'][:]
			labell = label_tabel[pth.split('.')[0]]
			label = [labell for x in range(point.shape[0])]
			label = np.array(label)
			# label = data_file['label'][:]

			# idx = [index for index, value in enumerate(list(label)) if value in self.label_map]
			# point_new = point[idx]
			# label_new = np.array([self.label_map.index(value) for value in label[idx]])

			point_list.append(point)
			label_list.append(label)
		self.data = np.concatenate(point_list, axis=0)
		self.label = np.concatenate(label_list, axis=0)

	def __getitem__(self, idx):
		point_idx = np.arange(0, self.num_points)
		np.random.shuffle(point_idx)
		point = self.data[idx][point_idx][:, :3]
		label = self.label[idx]

		pc = normal_pc(point)
		if self.aug:
			pc = rotation_point_cloud(pc)
			pc = jitter_point_cloud(pc)
		# print(pc.shape)
		pc = np.expand_dims(pc.transpose(), axis=2)
		return torch.from_numpy(pc).type(torch.FloatTensor), label, idx

	def __len__(self):
		return self.data.shape[0]


class Scannet_data_h51(data.Dataset):

	def __init__(self, pc_root, status='train', pc_input_num=1024, aug=True):
		super(Scannet_data_h51, self).__init__()
		self.num_points = pc_input_num
		self.status = status
		self.aug = aug
		# self.label_map = [2, 3, 4, 5, 6, 7, 9, 10, 14, 16]

		if self.status == 'train':
			data_pth = load_dir(pc_root, name='train_files.txt')
		else:
			data_pth = load_dir(pc_root, name='test_files.txt')

		point_list = []
		label_list = []
		for pth in data_pth:
			data_file = h5py.File(pth, 'r')
			point = data_file['data'][:]
			label = data_file['label'][:]

			# idx = [index for index, value in enumerate(list(label)) if value in self.label_map]
			# point_new = point[idx]
			# label_new = np.array([self.label_map.index(value) for value in label[idx]])

			point_list.append(point)
			label_list.append(label)
		self.data = np.concatenate(point_list, axis=0)
		self.label = np.concatenate(label_list, axis=0)

	def __getitem__(self, idx):
		point_idx = np.arange(0, self.num_points)
		np.random.shuffle(point_idx)
		point = self.data[idx][point_idx][:, :3]
		label = self.label[idx]

		pc = normal_pc(point)
		if self.aug:
			pc = rotation_point_cloud(pc)
			pc = jitter_point_cloud(pc)
		# print(pc.shape)
		pc = np.expand_dims(pc.transpose(), axis=2)
		return torch.from_numpy(pc).type(torch.FloatTensor), label, idx

	def __len__(self):
		return self.data.shape[0]


if __name__ == "__main__":
	source_private = 0
	common = 10
	target = 0
	num_total = 10

	common_class = [i for i in range(common)]
	source_private_class = [i + common for i in range(source_private)]
	target_private_class = [i + common + source_private for i in range(target)]

	source_class = common_class + source_private_class
	target_class = common_class + target_private_class
	print(source_class, '\n', target_class)


	data = Scannet_data_h5(pc_root='./dataset/Uni3DA_data/scannet')
	print(len(data))
	point, label = data[0]
	print(point.shape, label)






