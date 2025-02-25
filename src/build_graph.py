import os
import pickle
from ast import literal_eval
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix
import math
from cprint import pprint_color
from utils import get_max_item
from typing import Dict, List, Tuple
from param import args
class TargetSubseqs:
    target_subseqs_dict: Dict[int, List[List[int]]]

    def __init__(self, subseqs_path: str, target_subseqs_path: str, subseqs_target_path: str) -> None:
        """

        Args:
            data_path (str): data dir, e.g., ../data
            data_name (str): dataset name, e.g., Beauty
            save_path (str): save dir, e.g., ../data
        """
        self.subseqs_path = subseqs_path
        self.target_subseqs_path = target_subseqs_path
        self.subseqs_target_path = subseqs_target_path


    @staticmethod
    def generate_target_subseqs_dict(subseqs_path, target_subseqs_path, time_subseqs_path) -> [Dict[int, Dict[Tuple[int,...], Tuple[int, List[float]]]],int, int]:
        """Generate the target item for each subsequence, and save to pkl file."""
        train_dic: Dict[int, Dict[Tuple[int,...], Tuple[int, List[float]]]] = {}
        # 初始化最大值和最小值变量
        max_time = float('-inf')  # 初始为负无穷
        min_time = float('inf')  # 初始为正无穷

        with open(subseqs_path, "r", encoding="utf-8") as sub_file,\
                open(time_subseqs_path, "r", encoding="utf-8") as time_file:
            subseq_list = sub_file.readlines()
            time_list =  time_file.readlines()

            for index, (subseq, time_line) in enumerate(zip(subseq_list, time_list)):
                items: List[str] = subseq.split(" ")
                tag_train = int(items[-3])
                train_temp = items[1:-3]
                time = float(time_line.strip().split(" ")[-3]) - float(time_line.strip().split(" ")[-4])

                # 更新最大最小 time
                if time > max_time:
                    max_time = time
                if time < min_time:
                    min_time = time

                if tag_train not in train_dic:
                    train_dic[tag_train] = {}
                subseq_key = tuple(map(int, train_temp))
                if subseq_key not in train_dic[tag_train]:
                    train_dic[tag_train][subseq_key] = (1, [time])
                else:
                    count, time_list = train_dic[tag_train][subseq_key]
                    new_count = count + 1
                    time_list.append(time)
                    train_dic[tag_train][subseq_key] = (new_count, time_list)
        pprint_color(f'>>> Saving target-item specific subsequence set to "{train_dic}"')

        with open(target_subseqs_path, "wb") as fw:
            pickle.dump(train_dic, fw)

        return train_dic, min_time, max_time

    @staticmethod
    def load_target_subseqs_dict(subseqs_path, target_subseqs_path, time_subseqs_path):
        if not target_subseqs_path:
            raise ValueError("invalid data path")
        if not os.path.exists(target_subseqs_path):
            pprint_color("The dict not exist, generating...")
            TargetSubseqs.generate_target_subseqs_dict(subseqs_path, target_subseqs_path, time_subseqs_path)
        with open(target_subseqs_path, "rb") as read_file:
            data_dict = pickle.load(read_file)
        return data_dict

    @staticmethod
    def print_target_subseqs(target_subseqs_dict, target_id):
        subseq_list = target_subseqs_dict[target_id]
        pprint_color(f">>> subseq number: {len(subseq_list)}")
        pprint_color(subseq_list)

    @staticmethod
    def print_subseq_map_info(num_subseqs, subseq_id_map, id_subseq_map):
        pprint_color(f"==>> num_subseqs{' '*9}: {num_subseqs:>6}")
        pprint_color(f"==>> num_hashmap{' '*9}: {len(subseq_id_map):>6}")
        pprint_color(f"==>> duplicate subseq num: {num_subseqs - len(subseq_id_map):>6}")
        pprint_color(f"==>> subseq to id hashmap exapmle: {list(subseq_id_map.items())[:10]}")
        pprint_color(f"==>> id to subseq hashmap exapmle: {list(id_subseq_map.items())[:10]}")


    @staticmethod
    def get_subseq_id_map(subseqs_file: str) -> Tuple[Dict[Tuple[int,...], int], Dict[int, Tuple[int,...]], int, Dict[Tuple[int,...], int]]:
        subseq_id_map: Dict[Tuple[int,...], int] = {} # 一个序列对应的目标项目
        id_subseq_map: Dict[int, Tuple[int,...]] = {} # 一个目标项目对应的序列
        subseq_count_map: Dict[Tuple[int,...], int] = {}  # 记录每个子序列的出现次数
        num_subseqs = 0
        i = 0
        max_count = 0
        with open(subseqs_file, encoding="utf-8") as f:
            for index, line in enumerate(f):
                subseq: Tuple[int,...] = tuple(map(int, line.strip().split(" ")[1:-3])) # 1是用户

                # 记录子序列的出现次数
                if subseq not in subseq_count_map:
                    subseq_count_map[subseq] = 1
                else:
                    subseq_count_map[subseq] += 1
                # 更新最大出现次数
                max_count = max(max_count, subseq_count_map[subseq])

                if subseq not in subseq_id_map:
                    subseq_id_map.setdefault(subseq, i)
                    id_subseq_map.setdefault(i, subseq)
                    i += 1
                num_subseqs = index

        TargetSubseqs.print_subseq_map_info(num_subseqs, subseq_id_map, id_subseq_map)

        return subseq_id_map, id_subseq_map, max_count, subseq_count_map


class Graph:
    def __init__(self, adj_path):
        self.adj_path = adj_path
        if not os.path.exists(self.adj_path):
            raise FileNotFoundError(f'adjacency matrix not found in "{self.adj_path}"')
        self.load_graph()
# 归一化
    def norm_adj(self, mat: sp.csr_matrix) -> sp.coo_matrix:
        degree = np.array(mat.sum(axis=-1))
        d_sqrt_inv = np.reshape(np.power(degree, -0.5), [-1])
        d_sqrt_inv[np.isinf(d_sqrt_inv)] = 0.0
        d_sqrt_inv_mat = sp.diags(d_sqrt_inv)
        return mat.dot(d_sqrt_inv_mat).transpose().dot(d_sqrt_inv_mat).tocoo()

# 成扩展后的对称邻接矩阵
    def get_torch_adj(self, mat: sp.csr_matrix) -> torch.Tensor:
        a = sp.csr_matrix((mat.shape[0], mat.shape[0]))
        b = sp.csr_matrix((mat.shape[1], mat.shape[1]))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = mat + sp.eye(mat.shape[0])

        mat = self.norm_adj(mat)
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse_coo_tensor(idxs, vals, shape).cuda()

    # 加载存储在文件中的稀疏邻接矩阵
    def load_graph(self):
        with open(self.adj_path, "rb") as fs:
            train_matrix = pickle.load(fs)
        if not isinstance(train_matrix, coo_matrix):
            train_matrix = sp.coo_matrix(train_matrix)
        self.train_matrix = train_matrix
        self.torch_A = self.get_torch_adj(train_matrix)


    # 随机丢弃稀疏矩阵中的部分边
    @staticmethod
    def edge_random_dropout(sparse_matrix, dropout_rate=0.1):
        if not 0 <= dropout_rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1.")

        new_sparse_matrix = sparse_matrix.data.copy()
        nonzero_indices = np.where(new_sparse_matrix == 1)[0]
        num_dropout = int(dropout_rate * len(nonzero_indices))

        dropout_indices = np.random.choice(nonzero_indices, num_dropout, replace=False)
        new_sparse_matrix[dropout_indices] = 0
        return sp.coo_matrix(
            (new_sparse_matrix.data, (sparse_matrix.row, sparse_matrix.col)), shape=sparse_matrix.shape
        )

    @staticmethod
    def edge_dropout(sparse_matrix, doupout_num=1, dropout_percent=50):
        """
        Drop the smallest k weight edges for each target_item based on the overall dataset percentile.

        :param sparse_matrix: The sparse matrix to process
        :param topk: The number of smallest edges to drop for each target_item
        :param dropout_percent: The percentage of edges to drop based on overall edge weights
        :return: The new sparse matrix with dropped edges
        """
        # Copy the original sparse matrix's data
        new_sparse_matrix_data = sparse_matrix.data.copy()

        # Get the indices for rows (subseq) and columns (target_item) from the sparse matrix
        rows = sparse_matrix.row
        cols = sparse_matrix.col
        values = sparse_matrix.data

        # Flatten all values to compute overall ranking
        all_values_sorted = np.sort(values)

        # Calculate the value at the dropout_percent percentile from the entire dataset
        total_edges = len(values)
        percentile_index = int(np.floor(dropout_percent / 100 * total_edges))  # How many edges to keep
        threshold_value = all_values_sorted[percentile_index]  # Value at the threshold for top x%

        # For each target_item (column), we want to select the topk smallest weight edges to drop
        unique_target_items = np.unique(cols)

        for target_item in unique_target_items:
            # Get all edges that belong to the current target_item
            target_item_mask = (cols == target_item)
            target_item_values = values[target_item_mask]  # Corresponding edge weights
            target_item_rows = rows[target_item_mask]  # Corresponding row indices

            # If there are more than k edges, drop the k smallest weights
            if len(target_item_values) > doupout_num:
                sorted_indices = np.argsort(target_item_values)  # Sort the edge weights in ascending order
                drop_candidates = sorted_indices[:doupout_num]  # Select the smallest k indices

                # Drop edges only if their weight is smaller than the overall percentile threshold
                for idx in drop_candidates:
                    value_index = np.where((rows == target_item_rows[idx]) & (cols == target_item))[0][0]
                    if target_item_values[idx] <= threshold_value:
                        new_sparse_matrix_data[value_index] = 0


        # Return the updated sparse matrix with dropped edges
        return sp.coo_matrix(
            (new_sparse_matrix_data, (rows, cols)),
            shape=sparse_matrix.shape
        )

    @staticmethod
    def build_graph(
        target_subseqs_dict: Dict[int, Dict[Tuple[int,...], Tuple[int, List[float]]]],  # 包含时间差信息
        subseq_id_map: Dict[Tuple[int,...], int],
        num_items: int,
        num_subseqs: int,
        max_count:int,
        min_time: int,
        max_time: int,
        subseq_count_map: Dict[Tuple[int,...], int]
    ):
        pprint_color(f"==>> num_items: {num_items}")
        pprint_color(f"==>> num_subseqs: {num_subseqs}")
        target_item_list = []
        sub_seq_list = []
        rating_list = []
        min_count = 1
        count_w = args.count_w
        time_w = args.time_w
        for target_item, subseq_dic in target_subseqs_dict.items():
            for subseq, info_dict in subseq_dic.items():
                subseq_tuple = tuple(subseq)
                count, time_list = info_dict
                #0-1化只有ml-1m安排
                #time_list = [(t - min_time) / (max_time - min_time) for t in time_list]
                if subseq_tuple in subseq_id_map:
                    subseq_id= subseq_id_map[subseq_tuple]
                    target_item_list.append(target_item)  # 目标项目ID
                    sub_seq_list.append(subseq_id)  # 子序列映射为ID
                    time = sum(time_list) / len(time_list)
                    # 计算时间权重
                    time_weight = 1 + math.exp(-time/(1/265))
                    count_weight = (count - min_count) / (max_count - min_count) # (0-1)之间
                    weight = count_w * count_weight + time_w * time_weight #（0-1之间）
                    rating_list.append(weight)  # 使用时间差作为权重

        target_item_array = np.array(target_item_list)
        subseq_array = np.array(sub_seq_list)
        rating_array = np.array(rating_list)

        # 将数据保存为 CSV 文件
        data_frame = pd.DataFrame({
            "TargetItem": target_item_array,
            "SubSeq": subseq_array,
            "Rating": rating_array
        })
        data_frame.to_csv("wight.csv", index=False)


        pprint_color(f"==>> count_weight:{count_w}")
        pprint_color(f"==>> time_weight: {time_w}")
        pprint_color(f"==>> time: sum")
        pprint_color(f"==>> max target id: {np.max(target_item_array)}")
        pprint_color(f"==>> max subseq id: {np.max(subseq_array)}")
        return coo_matrix((rating_array, (subseq_array, target_item_array)), (num_subseqs, num_items + 1))


    @staticmethod
    def print_sparse_matrix_info(graph):
        pprint_color(f"==>> graph.nnz: {graph.nnz}")
        pprint_color(f"==>> graph.shape: {graph.shape}")
        pprint_color(f"==>> graph.max(): {graph.max()}")
        pprint_color(f"==>> graph.min(): {graph.min()}")
        pprint_color(f"==>> graph.sum(): {graph.sum()}")
        pprint_color(f"有相同 Target Item 的 Subseq 数: {np.sum(graph.tocsr().sum(axis=1)>1)}")
        pprint_color(f"有相同 Subseq 的 Target Item 数: {np.sum(graph.tocsc().sum(axis=0)>1)}")

    #保存邻接矩阵
    @staticmethod
    def save_sparse_matrix(save_path, graph):
        with open(save_path, "wb") as f:
            pickle.dump(graph, f)
            pprint_color(f">>> save graph to {save_path}")

#
def DS(i_file: str, o_file: str, max_len: int = 50) -> None:
    pprint_color(">>> Using DS to generate subsequence ...")
    with open(i_file, "r+", encoding="utf-8") as fr:
        seq_list = fr.readlines()
    subseq_dict: Dict[str, List] = {}
    max_save_len = max_len + 3
    max_keep_len = max_len + 2
    for data in seq_list:
        u_i, seq_str = data.split(" ", 1)
        seq = seq_str.split(" ")
        seq[-1] = str(literal_eval(seq[-1]))
        subseq_dict.setdefault(u_i, [])
        start = 0
        end = 3
        if len(seq) > max_save_len:
            while start < len(seq) - max_keep_len:
                end = start + 4
                while end < len(seq):
                    if start < 1 and end - start < max_save_len:
                        subseq_dict[u_i].append(seq[start:end])
                        end += 1
                    else:
                        subseq_dict[u_i].append(seq[start : start + max_save_len])
                        break
                start += 1
        else:
            while end < len(seq):
                subseq_dict[u_i].append(seq[start : end + 1])
                end += 1

    with open(o_file, "w+", encoding="utf-8") as fw:
        for u_i, subseq_list in subseq_dict.items():
            for subseq in subseq_list:
                fw.write(f"{u_i} {' '.join(subseq)}\n")
    pprint_color(f">>> DS done, written to {o_file}")


def count_graph(target_item_array, subseq_array, rating_array):
    # 将数据保存为 CSV 文件
    data_frame = pd.DataFrame({
        "TargetItem": target_item_array,
        "SubSeq": subseq_array,
        "Rating": rating_array
    })
    data_frame.to_csv("ml-1m_weight" + ".csv", index=False)


if __name__ == "__main__":
    pprint_color(">>> subsequences and graph generation pipeline")
    pprint_color(">>> Start to generate subsequence and build graph ...")

    force_flag = True
    dataset_list = [
        #"Beauty_2",
        "ml-1m_265",
        #"Sports_2",
        #"Toys_4",
        # "Yelp_1"
        # "Home_1"
        #"ml-1m_1"
    ]
    for dataset in dataset_list:
        data_root = "../data"
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        max_len = 50
        seqs_path = f"../data/{dataset}.txt"
        time_path = f"../data/{dataset}_time.txt"
        train_path = f"{data_root}/{dataset}_train_{max_len}.txt"
        subseqs_path = f"{data_root}/{dataset}_subseq_{max_len}.txt"
        time_subseqs_path = f"{data_root}/{dataset}_subseq_{max_len}_time.txt"
        target_subseqs_dict_path = f"{data_root}/{dataset}_t_{max_len}.pkl"
        time_target_subseqs_dict_path = f"{data_root}/{dataset}_t_{max_len}_time.pkl"
        sparse_matrix_path = f"{data_root}/{dataset}_graph_{max_len}.pkl"
        matrix_path = f"{data_root}/{dataset}_graph_t_{max_len}.pkl"

        if os.path.exists(sparse_matrix_path) and not force_flag:
            pprint_color(f'>>> "{sparse_matrix_path}" exists, skip.')
            continue
        #
        DS(seqs_path, subseqs_path, max_len) # 还是滑动窗口
        DS(time_path, time_subseqs_path, max_len) # 还是滑动窗口

        target_subseqs_dict, min_time, max_time = TargetSubseqs.generate_target_subseqs_dict(subseqs_path, target_subseqs_dict_path, time_subseqs_path)
        subseq_id_map, _, max_count, subseq_count_map= TargetSubseqs.get_subseq_id_map(subseqs_path)
        max_item = get_max_item(seqs_path)
        graph = Graph.build_graph(target_subseqs_dict, subseq_id_map, max_item + 1, len(subseq_id_map), max_count, min_time, max_time, subseq_count_map)
        Graph.save_sparse_matrix(sparse_matrix_path, graph)

        count_graph(graph.col, graph.row, graph.data)
