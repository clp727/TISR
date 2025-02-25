# -*- coding: utf-8 -*-
import math
import warnings
from ast import literal_eval
from typing import Optional, Union, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm
from cprint import pprint_color
from build_graph import Graph
from metric import ndcg_k, recall_at_k
from models import GCN, SASRecModel
from param import args
from utils import EarlyStopping
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def do_train(trainer, valid_rating_matrix, test_rating_matrix):
    pprint_color(">>> Train TISR Start")
    early_stopping = EarlyStopping(args.checkpoint_path, args.latest_path, patience=30)
    for epoch in range(args.epochs):
        args.rating_matrix = valid_rating_matrix
        trainer.train(epoch)
        # * evaluate on NDCG@20
        if args.do_eval:
            scores, _ = trainer.valid(epoch)
            early_stopping(np.array(scores[-1:]), trainer.model, trainer.optim_adam)
            if early_stopping.early_stop:
                pprint_color(">>> Early stopping")
                break

        # * test on while training
        if args.do_test and epoch >= args.min_test_epoch:
            args.rating_matrix = test_rating_matrix
            _, _ = trainer.test(epoch)

    args.train_matrix = test_rating_matrix
    print("---------------Change to test_rating_matrix!-------------------")
    checkpoint = torch.load(args.checkpoint_path)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optim_adam.load_state_dict(checkpoint['optimizer_state_dict'])
    scores, result_info = trainer.test(0)

def do_eval(trainer, test_rating_matrix):
    pprint_color(">>> Test PTSR Start")
    pprint_color(f'>>> Load model from "{args.latest_path}" for test')
    args.rating_matrix = test_rating_matrix
    trainer.load(args.latest_path)
    _, _ = trainer.test(0)


class Trainer:

    def __init__(
        self,
        model: Union[SASRecModel],
        train_dataloader: Optional[DataLoader],
        graph_dataloader: Optional[DataLoader],
        eval_dataloader: Optional[DataLoader],
        test_dataloader: DataLoader,
    ) -> None:
        pprint_color(">>> Initialize Trainer")

        cuda_condition = torch.cuda.is_available() and not args.no_cuda
        self.device: torch.device = torch.device("cuda" if cuda_condition else "cpu")
        torch.set_float32_matmul_precision(args.precision)

        self.model = torch.compile(model) if args.compile else model
        self.graph = Graph(args.graph_path)

        self.gcn = GCN()
        if cuda_condition:
            self.model.cuda()
            self.gcn.cuda()


        self.train_dataloader, self.graph_dataloader, self.eval_dataloader, self.test_dataloader = (
            train_dataloader,
            graph_dataloader,
            eval_dataloader,
            test_dataloader,
        )

        self.optim_adam = AdamW(self.model.parameters(), lr=args.lr_adam, weight_decay=args.weight_decay) # 优化所有参数
        self.scheduler = self.get_scheduler(self.optim_adam)

        # * prepare padding subseq for subseq embedding update
        self.all_subseq = self.get_all_pad_subseq(self.graph_dataloader)
        self.pad_mask = self.all_subseq > 0
        self.num_non_pad = self.pad_mask.sum(dim=1, keepdim=True)

        self.best_scores = {
            "valid": {
                "Epoch": 0,
                "HIT@5": 0.0,
                "HIT@10": 0.0,
                "HIT@20": 0.0,
                "NDCG@5": 0.0,
                "NDCG@10": 0.0,
                "NDCG@20": 0.0,
            },
            "test": {
                "Epoch": 0,
                "HIT@5": 0.0,
                "HIT@10": 0.0,
                "HIT@20": 0.0,
                "NDCG@5": 0.0,
                "NDCG@10": 0.0,
                "NDCG@20": 0.0,
            },
        }



    @staticmethod
    def get_result_log(post_fix):
        log_message = ""
        for key, value in post_fix.items():
            if isinstance(value, float):
                log_message += f" | {key}: {value:.4f}"
            else:
                log_message += f"{key}: [{value:03}]"
        return log_message

    def get_full_sort_score(
        self, epoch: int, answers: np.ndarray, pred_list: np.ndarray, mode
    ) -> Tuple[List[float], str]:
        recall, ndcg = [], []
        for k in [5, 10, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": recall[0],
            "HIT@10": recall[1],
            "HIT@20": recall[2],
            "NDCG@5": ndcg[0],
            "NDCG@10": ndcg[1],
            "NDCG@20": ndcg[2],
        }

        for key, value in post_fix.items():
            if key != "Epoch":
                args.tb.add_scalar(f"{mode}/{key}", value, epoch, new_style=True)

        args.logger.warning(self.get_result_log(post_fix))

        self.get_best_score(post_fix, mode)
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2]], str(post_fix)

    def get_best_score(self, scores, mode):
        improv = {
            "HIT@5": 0,
            "HIT@10": 0,
            "HIT@20": 0,
            "NDCG@5": 0,
            "NDCG@10": 0,
            "NDCG@20": 0,
        }
        self.best_scores[mode]["Epoch"] = scores["Epoch"]
        for key, value in scores.items():
            if key != "Epoch":
                improv[key] = value / (self.best_scores[mode][key] + 1e-12)
                if value > self.best_scores[mode][key]: # 只要有一个提升就更新
                    self.best_scores[mode] = scores
                args.tb.add_scalar(
                    f"best_{mode}/best_{key}",
                    self.best_scores[mode][key],
                    self.best_scores[mode]["Epoch"],
                    new_style=True,
                )
        if mode == "test":
            args.logger.critical(self.get_result_log(self.best_scores[mode]))
            # transfer improv to %
            improv = {k: round((v - 1) * 100, 2) for k, v in improv.items()}
            improv_message = ""
            for key, value in improv.items():
                if isinstance(value, float):
                    improv_message += f" | {key}: {value:.2f}%"
            args.logger.critical(f"v.s. BEST   {improv_message}\n")
        else:
            args.logger.error(self.get_result_log(self.best_scores[mode]))

    def save(self, file_name: str):
        """Save the model to the file_name"""
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name: str):
        """Load the model from the file_name"""
        self.model.load_state_dict(torch.load(file_name))

    @staticmethod
    def get_scheduler(optimizer):
        if args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr)
        elif args.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=args.factor, patience=args.patience, verbose=True
            )
        elif args.scheduler == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=literal_eval(args.milestones), gamma=args.gamma
            )
            pprint_color(f">>> scheduler: {args.scheduler}, milestones: {args.milestones}, gamma: {args.gamma}")
        elif args.scheduler == "warmup+cosine":
            warm_up_with_cosine_lr = lambda epoch: (
                epoch / args.warm_up_epochs
                if epoch <= args.warm_up_epochs
                else 0.5 * (math.cos((epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs) * math.pi) + 1)
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
        elif args.scheduler == "warmup+multistep":
            warm_up_with_multistep_lr = lambda epoch: (
                epoch / args.warm_up_epochs
                if epoch <= args.warm_up_epochs
                else args.gamma ** len([m for m in literal_eval(args.milestones) if m <= epoch])
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
        else:
            raise ValueError("Invalid scheduler")
        return scheduler

    def get_all_pad_subseq(self, gcn_dataloader: DataLoader) -> Tensor:
        all_subseq_ids = []
        all_subseq = []
        for _, (rec_batch) in tqdm(
            enumerate(gcn_dataloader),
            total=len(gcn_dataloader),
            desc=f"{args.save_name} | Device: {args.gpu_id} | get_all_pad_subseq",
            leave=False,
            dynamic_ncols=True,
        ):
            subseq_id, _, subsequence, _, _ = rec_batch
            all_subseq_ids.append(subseq_id)
            all_subseq.append(subsequence)
        all_subseq_ids = torch.cat(all_subseq_ids, dim=0)
        all_subseq = torch.cat(all_subseq, dim=0)

        # * remove duplicate subsequence
        tensor_np = all_subseq_ids.numpy()
        _, indices = np.unique(tensor_np, axis=0, return_index=True)
        sorted_indices = np.sort(indices)
        all_subseq_ids = all_subseq_ids[sorted_indices]
        all_subseq = all_subseq[sorted_indices]
        return all_subseq


    def subseq_embed_update(self, epoch):
        self.model.item_embeddings.cpu()
        self.model.subseq_embeddings.cpu()
        subseq_emb = self.model.item_embeddings(self.all_subseq)
        subseq_emb_avg: Tensor = (
            torch.sum(subseq_emb * self.pad_mask.unsqueeze(-1), dim=1) / self.num_non_pad
        )
        # * accelerate convergence
        self.model.subseq_embeddings.weight.data = (
            subseq_emb_avg if epoch == 0 else (subseq_emb_avg + self.model.subseq_embeddings.weight.data) / 2
        )

        self.model.item_embeddings.to(self.device)
        self.model.subseq_embeddings.to(self.device)


class PTSRTrainer(Trainer):

    def train_epoch(self, epoch, train_dataloader):
        self.model.train()
        if epoch == 0:
            train_matrix = self.graph.edge_random_dropout(self.graph.train_matrix, args.dropout_rate)
            self.graph.torch_A = self.graph.get_torch_adj(train_matrix)
            aug_train_matrix = self.graph.edge_dropout(train_matrix, doupout_num = args.doupout_num, dropout_percent = args.p)
            self.aug_torch_A = self.graph.get_torch_adj(aug_train_matrix)

        rec_avg_loss = 0.0
        batch_num = len(train_dataloader)
        args.tb.add_scalar("train/LR", self.optim_adam.param_groups[0]["lr"], epoch, new_style=True)

        self.subseq_embed_update(epoch)

        for batch_i, (rec_batch) in tqdm(
            enumerate(train_dataloader),
            total=batch_num,
            leave=False,
            desc=f"{args.save_name} | Device: {args.gpu_id} | Rec Training Epoch {epoch}",
            dynamic_ncols=True,
        ):
            # * rec_batch shape: key_name x batch_size x feature_dim
            rec_batch = tuple(t.to(self.device) for t in rec_batch)
            _, _, input_ids, target_pos_1, _ = rec_batch

            _, self.model.all_item_emb= self.gcn(
                self.aug_torch_A, self.model.subseq_embeddings.weight, self.model.item_embeddings.weight
            )

            # * prediction task
            loss = self.model.train_forward(input_ids, target_pos_1[:, -1])
            # self.optim_adagrad.zero_grad()
            self.optim_adam.zero_grad()
            loss.backward()
            # self.optim_adagrad.step()
            self.optim_adam.step()

            rec_avg_loss += loss.item()
            if args.batch_loss:
                args.tb.add_scalar("batch_loss/rec_loss", loss.item(), epoch * batch_num + batch_i, new_style=True)

        self.scheduler.step()
        # * print & write log for each epoch
        # * post_fix: print the average loss of the epoch
        post_fix = {
            "Epoch": epoch,
            "lr_adam": round(self.optim_adam.param_groups[0]["lr"], 6),
            "rec_avg_loss": round(rec_avg_loss / batch_num, 4)
        }

        if (epoch + 1) % args.log_freq == 0:
            loss_message = ""
            for key, value in post_fix.items():
                if "loss" in key:
                    args.tb.add_scalar(f"train/{key}", value, epoch, new_style=True)
                if isinstance(value, float):
                    loss_message += f" | {key}: {value}"
                else:
                    loss_message += f"{key}: [{value:03}]"

            loss_message += f" | Message: {args.save_name}"
            args.logger.info(loss_message)

    def full_test_epoch(self, epoch: int, dataloader: DataLoader, mode):
        with torch.no_grad():
            self.model.eval()
            # * gcn is fixed in the test phase. So it's unnecessary to call gcn() every batch.
            _, self.model.all_item_emb  = self.gcn(
                self.aug_torch_A, self.model.subseq_embeddings.weight, self.model.item_embeddings.weight
            )

            for i, batch in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                leave=False,
                desc=f"{args.save_name} | Device: {args.gpu_id} | Test Epoch {epoch}",
                dynamic_ncols=True,
            ):
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, _, answers = batch
                # * SHAPE: [Batch_size, Seq_len, Hidden_size] -> [256, 50, 64]
                logits: Tensor = self.model(input_ids)  # [BxLxH]
                rating_pred = logits.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()

                # * 将已经有评分的 item 的预测评分设置为 0, 防止推荐已经评分过的 item
                rating_pred[args.rating_matrix[batch_user_index].toarray() > 0] = 0
                ind: np.ndarray = np.argpartition(rating_pred, -20)[:, -20:]

                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

            return self.get_full_sort_score(epoch, answer_list, pred_list, mode)

    def train(self, epoch) -> None:
        assert self.train_dataloader is not None
        args.mode = "train"
        self.train_epoch(epoch, self.train_dataloader)

    def valid(self, epoch) -> Tuple[List[float], str]:
        assert self.eval_dataloader is not None
        args.mode = "valid"
        return self.full_test_epoch(epoch, self.eval_dataloader, "valid")

    def test(self, epoch) -> Tuple[List[float], str]:
        args.mode = "test"
        return self.full_test_epoch(epoch, self.test_dataloader, "test")