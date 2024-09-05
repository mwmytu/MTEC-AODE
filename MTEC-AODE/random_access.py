import os
import csv
import numpy as np
from fastdtw import fastdtw
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import random

files = {
    'pems03': ['PEMS03/pems03.npz', 'PEMS03/distance.csv'],
    'pems04': ['PEMS04/pems04.npz', 'PEMS04/distance.csv'],
    'pems07': ['PEMS07/pems07.npz', 'PEMS07/distance.csv'],
    'pems08': ['PEMS08/pems08.npz', 'PEMS08/distance.csv'],
    'pemsbay': ['PEMSBAY/pems_bay.npz', 'PEMSBAY/distance.csv'],
    'pemsD7M': ['PeMSD7M/PeMSD7M.npz', 'PeMSD7M/distance.csv'],
    'pemsD7L': ['PeMSD7L/PeMSD7L.npz', 'PeMSD7L/distance.csv']
}
def calculate_gaussian_kernel_similarity(data, sigma):
    num_node = data.shape[1]
    gaussian_similarity_matrix = np.zeros((num_node, num_node))

    for i in range(num_node):
        for j in range(num_node):
            distance = np.linalg.norm(data[:, i] - data[:, j])
            gaussian_similarity_matrix[i, j] = np.exp(-distance ** 2 / (2 * sigma ** 2))
    return gaussian_similarity_matrix

def select_seven_days(data):
    num_days = data.shape[0] // (24 * 12)
    start_day = random.randint(0, num_days - 1)
    selected_data = data[start_day * 24 * 12 : (start_day + 1) * 24 * 12]
    print(data.shape)
    return selected_data

def read_data(args):
    filename = args.filename
    file = files[filename]
    filepath = "./data/"
    if args.remote:
        filepath = '/home/lantu.lqq/ftemp/data/'
    data = np.load(filepath + file[0])['data']
    print(data.shape)
    print(filepath)
    data = select_seven_days(data)
    num_node = data.shape[1]
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]

    random_dtw_matrix = np.random.randint(0, 2, size=(num_node, num_node))
    dist_matrix = random_dtw_matrix

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres1] = 1

    if not os.path.exists(f'data/{filename}_gaussian_similarity.npy'):
        data_mean = np.mean([data[:, :, 0][24 * 12 * i: 24 * 12 * (i + 1)] for i in range(data.shape[0] // (24 * 12))],
                            axis=0)
        data_mean = data_mean.squeeze().T
        args.sigma1 *= 1.6
        gaussian_similarity_matrix = calculate_gaussian_kernel_similarity(data_mean, args.sigma1)
        np.save(f'data/{filename}_gaussian_similarity.npy', gaussian_similarity_matrix)

    gaussian_matrix = np.load(f'data/{filename}_gaussian_similarity.npy')
    print(args.thres2)
    gaussian_matrix[gaussian_matrix < args.thres1] = 0

    if not os.path.exists(f'data/{filename}_spatial_distance.npy'):
        with open(filepath + file[1], 'r') as fp:
            dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
            file = csv.reader(fp)
            for line in file:
                break
            for line in file:
                start = int(line[0])
                end = int(line[1])
                dist_matrix[start][end] = float(line[2])
                dist_matrix[end][start] = float(line[2])
            np.save(f'data/{filename}_spatial_distance.npy', dist_matrix)

    dist_matrix = np.load(f'data/{filename}_spatial_distance.npy')
    std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma2
    sp_matrix = np.exp(- dist_matrix ** 2 / sigma ** 2)
    sp_matrix[sp_matrix < args.thres2] = 0

    print(f'average degree of spatial graph is {np.sum(sp_matrix > 0) / 2 / num_node}')
    print(f'average degree of semantic graph is {np.sum(gaussian_matrix > 0) / 2 / num_node}')
    return torch.from_numpy(data.astype(np.float32)), mean_value, std_value, gaussian_matrix, sp_matrix


def get_normalized_adj(A):
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


class MyDataset(Dataset):
    def __init__(self, data, split_start, split_end, his_length, pred_length):
        split_start = int(split_start)
        split_end = int(split_end)
        self.data = data[split_start: split_end]
        self.his_length = his_length
        self.pred_length = pred_length
    
    def __getitem__(self, index):
        x = self.data[index: index + self.his_length].permute(1, 0, 2)
        y = self.data[index + self.his_length: index + self.his_length + self.pred_length][:, :, 0].permute(1, 0)
        return torch.Tensor(x), torch.Tensor(y)
    def __len__(self):
        return self.data.shape[0] - self.his_length - self.pred_length + 1


def generate_dataset(data, args):

    batch_size = args.batch_size
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    his_length = args.his_length
    pred_length = args.pred_length
    train_dataset = MyDataset(data, 0, data.shape[0] * train_ratio, his_length, pred_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = MyDataset(data, data.shape[0]*train_ratio, data.shape[0]*(train_ratio+valid_ratio), his_length, pred_length)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MyDataset(data, data.shape[0]*(train_ratio+valid_ratio), data.shape[0], his_length, pred_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader

