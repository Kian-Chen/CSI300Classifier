from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from optimizer.muon import SingleDeviceMuonWithAuxAdam
from utils.tools import compute_model_stats


warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        muon_params = []
        adam_params = []

        for n, p in self.model.named_parameters():
            if p.ndim >= 2:
                if p.requires_grad:
                    muon_params.append(p)
                else:
                    adam_params.append(p)
            else:
                adam_params.append(p)

        param_groups = [
            dict(params=muon_params, lr=self.args.learning_rate * 5, momentum=0.95, use_muon=True),
            dict(params=adam_params, lr=self.args.learning_rate, betas=(0.9, 0.95), eps=1e-8, use_muon=False)
        ]

        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        return optimizer

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x)

                pred = outputs.detach().cpu()
                loss = criterion(pred.view(-1, 2), label.view(-1).cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=-1)   # (N*B, L, 2)
        predictions = torch.argmax(probs, dim=-1).cpu().numpy()  # (N*B, L)
        trues = trues.cpu().numpy()  # (N*B, L)
        accuracy = cal_accuracy(predictions.flatten(), trues.flatten())


        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='test')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate,
                                            cycle_momentum=False)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs.view(-1, 2), label.view(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim,  scheduler, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        batch_x_, _ = next(iter(test_loader))
        batch_x_ = batch_x_.float().to(self.device)
        self.model_stats = compute_model_stats(self.model, batch_x_)


        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x)
                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)   # (N*B, L, 2)
        trues = torch.cat(trues, 0)   # (N*B, L)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds, dim=-1)  # (N*B, L, 2)
        predictions = torch.argmax(probs, dim=-1).cpu().numpy().flatten()
        trues = trues.cpu().numpy().flatten()

        # metrics
        accuracy = accuracy_score(trues, predictions)
        f1 = f1_score(trues, predictions, average='binary')
        precision = precision_score(trues, predictions, average='binary')
        recall = recall_score(trues, predictions, average='binary')
        auc = roc_auc_score(trues, probs[..., 1].cpu().numpy().flatten())

        # logging
        log_path = os.path.join(self.args.log_dir, self.args.log_name)
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(log_path, 'a') as f:
            f.write(setting + "  \n")
            if hasattr(self, 'model_stats'):
                stats = self.model_stats
                f.write(
                    f"params: {stats['Trainable params']}  "
                    f"MACs: {stats['MACs']}  "
                    f"FLOPs: {stats['FLOPs']}  "
                    f"total act: {stats['Total activations']}  "
                    f"peak act: {stats['Peak activations']}\n"
                )
            else:
                f.write("\n")
            f.write(f"Acc: {accuracy:.4f}  F1: {f1:.4f}  Prec: {precision:.4f}  Rec: {recall:.4f}  AUC: {auc:.4f}\n\n")

        print(f"Acc: {accuracy:.4f}  F1: {f1:.4f}  Prec: {precision:.4f}  Rec: {recall:.4f}  AUC: {auc:.4f}")
        return