import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import Model
from dataloader import Modelnet40_data, Shapenet_data, Scannet_data_h5
import time


import argparse
import datetime
from testacc import testtongji
from tensorboardX import SummaryWriter
from easydl import *
from Model import MetricLoss




# Command setting
source_private = 5
common = 5
target_private = 5
num_total =15

common_class = [i for i in range(common)]
source_private_class = [i + common for i in range(source_private)]
target_private_class = [i + common + source_private for i in range(target_private)]


source_class = common_class + source_private_class
target_class = common_class + target_private_class

print(source_class, '\n', target_class)

parser = argparse.ArgumentParser(description='Main')
parser.add_argument('-source', '-s', type=str, help='source dataset', default='modelnet')
parser.add_argument('-target', '-t', type=str, help='target dataset', default='shapenet')
parser.add_argument('-batchsize', '-b', type=int, help='batch size', default=64)
parser.add_argument('-gpu', '-g', type=str, help='cuda id', default='0')
parser.add_argument('-epochs', '-e', type=int, help='training epoch', default=400)
parser.add_argument('-models', '-m', type=str, help='alignment model', default='MDA')
parser.add_argument('-lr', type=float, help='learning rate', default=0.001)
parser.add_argument('-scaler', type=float, help='scaler of learning rate', default=1.)
parser.add_argument('-weight', type=float, help='weight of src loss', default=1.)
parser.add_argument('-datadir', type=str, help='directory of data', default='./dataset/')
parser.add_argument('-tb_log_dir', type=str, help='directory of tb', default='./logs')
parser.add_argument('-entropy_start', type=int, help='begin epoch of entropy sperate', default=40)
parser.add_argument('-entropy_th', type=float, help='begin epoch of entropy sperate', default=0.35)
args = parser.parse_args()




device = 'cuda:0'
gpu_ids = [0]  # select_GPUs(1)
output_device = gpu_ids[0]

now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('./logs', now)
os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

BATCH_SIZE = args.batchsize * len(args.gpu.split(','))
LR = args.lr
weight_decay = 5e-4
momentum = 0.9
max_epoch = args.epochs
en_start = args.entropy_start
en_th = args.entropy_th
num_class = 6
dir_root = os.path.join(args.datadir, 'PointDA_data/')
class_acc = [0] * 16

# resume_path = './logs/Mar03_09-16-28/current.pkl'


def get_entropy(before_softmax, class_temperature=10.0):
	before_softmax = before_softmax / class_temperature
	after_softmax = nn.Softmax(-1)(before_softmax)
	entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
	entropy_norm = entropy / np.log(200) #after_softmax.size(1)
	return entropy_norm


def normalize_weight(x):
	min_val = x.min()
	max_val = x.max()
	x = (x - min_val) / (max_val - min_val)
	x = x / torch.mean(x)
	return x.detach()


def reverse_sigmoid(y):
	return torch.log(y / (1.0 - y + 1e-10) + 1e-10)


def get_target_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
	return - get_source_share_weight(domain_out, before_softmax, domain_temperature, class_temperature)


def get_source_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
	before_softmax = before_softmax / class_temperature
	after_softmax = nn.Softmax(-1)(before_softmax)
	entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
	tem = np.log(200)
	entropy_norm = entropy / tem  # after_softmax.size(1)
	weight = entropy_norm - domain_out
	weight = weight.detach()
	return weight

def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by half by every 5 or 10 epochs"""
	if epoch > 0:
		if epoch <= 30:
			lr = args.lr * args.scaler * (0.5 ** (epoch // 5))
		else:
			lr = args.lr * args.scaler * (0.5 ** (epoch // 10))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		writer.add_scalar('lr_dis', lr, epoch)

def discrepancy(out1, out2):
	"""discrepancy loss"""
	out = torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))
	return out

def make_variable(tensor, volatile=False):
	"""Convert Tensor to Variable."""
	if torch.cuda.is_available():
		tensor = tensor.cuda()
	return Variable(tensor, volatile=volatile)

# print(dir_root)
def main():
	print('Start Training\nInitiliazing\n')
	print('src:', args.source)
	print('tar:', args.target)
	# Data loading
	# Data loading

	data_func = {'modelnet': Modelnet40_data, 'scannet': Scannet_data_h5, 'shapenet': Shapenet_data}

	source_train_dataset = data_func[args.source](pc_input_num=1024, class_list=source_class, status='train', aug=True,
												  pc_root=dir_root + args.target)
	source_test_dataset = data_func[args.source](pc_input_num=1024, class_list=source_class, status='test', aug=True,
												 pc_root=dir_root + args.target)
	target_train_dataset1 = data_func[args.target](pc_input_num=1024, class_list=target_class, status='train', aug=True,
												   pc_root=dir_root + args.target)
	target_test_dataset1 = data_func[args.target](pc_input_num=1024, class_list=target_class, status='test', aug=True,
												  pc_root=dir_root + args.target)

	num_source_train = len(source_train_dataset)
	num_source_test = len(source_test_dataset)
	num_target_train1 = len(target_train_dataset1)
	num_target_test1 = len(target_test_dataset1)

	source_train_dataloader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
										 drop_last=True)
	source_test_dataloader = DataLoader(source_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
										drop_last=True)
	target_train_dataloader1 = DataLoader(target_train_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
										  drop_last=True)
	target_test_dataloader1 = DataLoader(target_test_dataset1, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
										 drop_last=True)

	print('num_source_train: {:d}, num_source_test: {:d}, num_target_test1: {:d} '.format(num_source_train,
																						  num_source_test,
																						  num_target_test1))
	print('batch_size:', BATCH_SIZE)

	# Model

	model = Model.Net_MDA()
	print(torch.cuda.current_device())
	model = model.to(device=device)
	feature_extractor = nn.DataParallel(model.g, device_ids=gpu_ids, output_device=output_device).train(True)
	classifier = nn.DataParallel(model.c1, device_ids=gpu_ids, output_device=output_device).train(True)
	discriminator = nn.DataParallel(model.d, device_ids=gpu_ids, output_device=output_device).train(True)
	discriminator_separate = nn.DataParallel(model.ds, device_ids=gpu_ids,
											 output_device=output_device).train(True)
	scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
	optimizer_finetune = OptimWithSheduler(
		optim.SGD(feature_extractor.parameters(), lr=args.lr *1.0, weight_decay=0.0005, momentum=0.9,
				  nesterov=True),
		scheduler)
	optimizer_cls = OptimWithSheduler(
		optim.SGD(classifier.parameters(), lr=args.lr / 1, weight_decay=0.0005, momentum=0.9, nesterov=True),
		scheduler)
	optimizer_discriminator = OptimWithSheduler(
		optim.SGD(discriminator.parameters(), lr=args.lr * 10, weight_decay=0.0005, momentum=0.9, nesterov=True),
		scheduler)
	optimizer_discriminator_separate = OptimWithSheduler(
		optim.SGD(discriminator_separate.parameters(), lr=args.lr * 5, weight_decay=0.0005, momentum=0.9,
				  nesterov=True),
		scheduler)
	criterion = nn.CrossEntropyLoss()
	metric_criterion = MetricLoss()
	criterion = criterion.to(device=device)

	# assert os.path.exists(resume_path)
	# data = torch.load(open(resume_path, 'rb'))
	# feature_extractor.load_state_dict(data['feature_extractor'])
	# classifier.load_state_dict(data['classifier'])
	# discriminator_separate.load_state_dict(data['discriminator_separate'])

	best_target_test_acc = 0
	global_step = 0
	for epoch in range(max_epoch):
		since_e = time.time()

		model.train()

		weight_list_s = [0] * 21
		weight_list_t = [0] * 21
		sim_list_s = [0] * 21
		sim_list_t = [0] * 21
		en_list_s = [0] * 21
		en_list_t = [0] * 21
		sample_num_s = [0] * 21
		sample_num_t = [0] * 21

		loss_total = 0
		loss_adv_total = 0
		loss_adv_sperate = 0
		loss_node_total = 0
		correct_total = 0
		data_total = 0
		data_t_total = 0
		loss_gl = 0
		loss_w = 0

		# Training

		for batch_idx, (batch_s, batch_t) in enumerate(zip(source_train_dataloader, target_train_dataloader1)):

			data, label,_ = batch_s
			data_t, label_t,_ = batch_t

			data = data.to(device=device)
			label = label.to(device=device).long()
			data_t = data_t.to(device=device)
			label_t = label_t.to(device=device).long()
			w_loss =  torch.zeros(1, 1).to(output_device)


			fc_s, features_s, nod_fea_s = feature_extractor.forward(data, node=True)
			global_feature1_s = features_s[1].squeeze(2)
			refs1_s = features_s[0:1]
			fc_t, features_t, nod_fea_t = feature_extractor.forward(data_t, node=True)
			global_feature1_t = features_t[1].squeeze(2)
			refs1_t = features_t[0:1]

			fc_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc_s)
			fc_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc_t)

			domain_prob_discriminator_source = discriminator.forward(fc_s)
			domain_prob_discriminator_target = discriminator.forward(fc_t)

			domain_prob_discriminator_source_separate = discriminator_separate.forward(fc_s.detach())
			domain_prob_discriminator_target_separate = discriminator_separate.forward(fc_t.detach())

			source_share_weight = get_source_share_weight(domain_prob_discriminator_source_separate, fc2_s,
														  domain_temperature=1.0, class_temperature=10.0)
			source_share_weight = normalize_weight(source_share_weight)

			target_share_weight = get_target_share_weight(domain_prob_discriminator_target_separate, fc2_t,
														  domain_temperature=1.0, class_temperature=10.0)
			w_abs = torch.abs(target_share_weight)

			if epoch > en_start:
				for i in w_abs:
					if i > en_th:
						w_loss += (-1) * i

			target_share_weight = normalize_weight(target_share_weight)

			en_s = get_entropy(fc2_s, 5.0)
			en_t = get_entropy(fc2_t, 5.0)

			index = 0
			for each_label in label:
				weight_list_s[each_label] = weight_list_s[each_label] + source_share_weight[index].item()
				sim_list_s[each_label] = sim_list_s[each_label] + domain_prob_discriminator_source_separate[
					index].item()
				en_list_s[each_label] = en_list_s[each_label] + en_s[index].item()
				sample_num_s[each_label] = sample_num_s[each_label] + 1
				index = index + 1

			index = 0
			for each_label in label_t:
				weight_list_t[each_label] = weight_list_t[each_label] + target_share_weight[index].item()
				sim_list_t[each_label] = sim_list_t[each_label] + domain_prob_discriminator_target_separate[
					index].item()
				en_list_t[each_label] = en_list_t[each_label] + en_t[index].item()
				sample_num_t[each_label] = sample_num_t[each_label] + 1
				index = index + 1

			# Classification loss

			adv_loss = torch.zeros(1, 1).to(output_device)
			adv_loss_separate = torch.zeros(1, 1).to(output_device)

			tmp = source_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source,
																	 torch.ones_like(domain_prob_discriminator_source))
			adv_loss += torch.mean(tmp, dim=0, keepdim=True)
			tmp = target_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_target,
																	 torch.zeros_like(domain_prob_discriminator_target))
			adv_loss += torch.mean(tmp, dim=0, keepdim=True)

			adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_source_separate,
											  torch.ones_like(domain_prob_discriminator_source_separate))
			adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_target_separate,
											  torch.zeros_like(domain_prob_discriminator_target_separate))

			# ============================== cross entropy loss, it receives logits as its inputs
			ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, label)
			ce = torch.mean(ce, dim=0, keepdim=True)
			loss_metric_s = metric_criterion(global_feature1_s, refs1_s)
			loss_metric_t = metric_criterion(global_feature1_t, refs1_t)
			gl_loss = loss_metric_s + loss_metric_t
			loss_c = ce + adv_loss_separate + w_loss + gl_loss

			with OptimizerManager(
					[optimizer_finetune, optimizer_cls, optimizer_discriminator, optimizer_discriminator_separate]):
				loss = adv_loss + loss_c
				loss.backward()

			loss_total += ce.item() * data.size(0)
			loss_adv_total += adv_loss.item() * data.size(0)
			loss_adv_sperate += adv_loss_separate.item() * data.size(0)
			data_total += data.size(0)
			data_t_total += data_t.size(0)
			loss_gl += gl_loss.item() * data.size(0)
			loss_w += w_loss.item()
			writer.add_scalar('adv_loss', adv_loss, global_step)
			writer.add_scalar('ce', ce, global_step)
			writer.add_scalar('adv_loss_separate', adv_loss_separate, global_step)
			writer.add_scalar('loss_c', loss_c, global_step)

			if (batch_idx + 1) % 10 == 0:
				print('Train:{} [{} {}/{}  loss_s: {:.4f} \t loss_adv: {:.4f} \t loss_adv_sperate: {:.4f}\t loss_gl: {:.4f}\t loss_w: {:.4f}  ]'.format(
					epoch, data_total, data_t_total, num_source_train, loss_total / data_total,
																	   loss_adv_total / data_total,
																	   loss_adv_sperate / data_total,
					                                                   loss_gl / data_total,
					                                                   loss_w/data_total
				))

			global_step += 1
		for i in range(21):
			if sample_num_s[i] == 0:
				weight_list_s[i] = 0
				sim_list_s[i] = 0
				en_list_s[i] = 0
			else:
				weight_list_s[i] = weight_list_s[i] / sample_num_s[i]
				sim_list_s[i] = sim_list_s[i] / sample_num_s[i]
				en_list_s[i] = en_list_s[i] / sample_num_s[i]
		for i in range(21):
			if sample_num_t[i] == 0:
				weight_list_t[i] = 0
				sim_list_t[i] = 0
				en_list_t[i] = 0
			else:
				weight_list_t[i] = weight_list_t[i] / sample_num_t[i]
				sim_list_t[i] = sim_list_t[i] / sample_num_t[i]
				en_list_t[i] = en_list_t[i] / sample_num_t[i]

		print('w_list:')
		en_list_s = [round(i, 2) for i in en_list_s]
		en_list_t = [round(i, 2) for i in en_list_t]
		sim_list_s = [round(i, 2) for i in sim_list_s]
		sim_list_t = [round(i, 2) for i in sim_list_t]
		weight_list_s = [round(i, 2) for i in weight_list_s]
		weight_list_t = [round(i, 2) for i in weight_list_t]
		print(weight_list_s)
		print(weight_list_t)
		print('en_list:')
		print(en_list_s)
		print(en_list_t)
		print('sim_list:')
		print(sim_list_s)
		print(sim_list_t)

		# Testing

		with torch.no_grad():
			model.eval()
			class_correct = [0] * 16
			class_num = [0] * 16

			for batch_idx, (data, label,_) in enumerate(target_test_dataloader1):
				data = data.to(device=device)
				label = label.to(device=device).long()
				feature, trans, trans_feat = feature_extractor.forward(data,node=True)
				feature, feature_source, before_softmax, predict_prob = classifier.forward(feature)
				domain_prob = discriminator_separate.forward(feature)
				target_share_weight = get_target_share_weight(domain_prob, before_softmax, domain_temperature=1.0,
															  class_temperature=15.0)
				_, pred = torch.max(predict_prob, 1)

				class_num, class_correct = testtongji(output=target_share_weight, label=label, pred=pred, class_list=source_class,
						   class_num=class_num,
						   class_correct=class_correct, w=-0.1)

			for i in range(16):
				if class_num[i] == 0:
					class_acc[i] = 0
				else:
					class_acc[i] = class_correct[i] / class_num[i]
			print(class_acc)
			pred_acc = np.sum(class_acc) / 6

			data = {
				"feature_extractor": feature_extractor.state_dict(),
				'classifier': classifier.state_dict(),
				'discriminator': discriminator.state_dict() if not isinstance(discriminator, Nonsense) else 1.0,
				'discriminator_separate': discriminator_separate.state_dict(),
			}

			if pred_acc > best_target_test_acc:
				best_target_test_acc = pred_acc
				with open(os.path.join(log_dir, 'best.pkl'), 'wb') as f:
					torch.save(data, f)

			with open(os.path.join(log_dir, 'current.pkl'), 'wb') as f:
				torch.save(data, f)

			print('Target 1:{} [overall_acc: {:.4f}  \t Best Target Acc: {:.4f}]'.format(
				epoch, pred_acc, best_target_test_acc
			))

			writer.add_scalar('accs/target_test_acc', pred_acc, epoch)

		time_pass_e = time.time() - since_e
		print('The {} epoch takes {:.0f}m {:.0f}s'.format(epoch, time_pass_e // 60, time_pass_e % 60))
		print(args)
		print(' ')


if __name__ == '__main__':
	since = time.time()
	main()
	time_pass = since - time.time()
	print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))

