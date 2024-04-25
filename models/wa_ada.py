import logging
import numpy as np
from tqdm import tqdm
import torch
import copy
import optuna
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base_ada import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8

init_epoch = 100
init_lr = 0.1
init_milestones = [60]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 100
#lrate = 0.1
milestones = [60]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2


class WA_Ada(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.checkpoint_acc = 0
        self.val_acc = None

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def load_data(self, data_manager):
        self._cur_task += 1

        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        val_dataset = data_manager.get_val_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            val_samples_per_class=25
        )

        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )

        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    def inc_train(self, trial, data_manager):

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            self._init_train(data_manager, self.test_loader)
        else:
            self._update_representation(trial, data_manager, self.test_loader, self.val_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        return self.val_acc

    def _init_train(self, data_manager, test_loader):

        self._network.update_fc(self._total_classes)

        #if len(self._multiple_gpus) > 1:
            #self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.to(self._device)
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train"
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        optimizer = optim.SGD(
            self._network.parameters(),
            momentum=0.9,
            lr=init_lr,
            weight_decay=init_weight_decay,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
        )

        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.type(torch.LongTensor).to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

        #if len(self._multiple_gpus) > 1:
            #self._network = self._network.module

    def _update_representation(self, trial, data_manager, test_loader, val_loader):
        lamda = trial.suggest_float('lambda', 0.5, 0.95)
        memory = trial.suggest_int('memory', 40, 50, step=5)
        lrate = trial.suggest_float('lrate', 0.05, 0.1, step=0.01)
        print("Selected memory is: ", memory)
        print("Selected lambda: ", lamda)
        print("Selected lrate: ", lrate)

        self.build_rehearsal_memory(data_manager, memory)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendents=[self._get_memory(temp=True), self._get_memory()]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self._network.update_fc(self._total_classes)
        self._network.to(self._device)
        temp_net = copy.deepcopy(self._network)

        #if len(self._multiple_gpus) > 1:
            #temp_net = nn.DataParallel(temp_net, self._multiple_gpus)

        optimizer = optim.SGD(
            temp_net.parameters(),
            lr=lrate,
            momentum=0.9,
            weight_decay=weight_decay,
        )  # 1e-5

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=lrate_decay
        )

        temp_net.to(self._device)
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            temp_net.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.type(torch.LongTensor).to(self._device)
                logits = temp_net(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                )

                loss = (1-lamda) * loss_clf + lamda * loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            self.val_acc = self._compute_accuracy(temp_net, val_loader)

            trial.report(self.val_acc, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(temp_net, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

        if self.val_acc > self.checkpoint_acc:
            info = 'Better trial is found...'
            logging.info(info)
            torch.save(temp_net.state_dict(), 'network_checkpoint.pth')
            self.checkpoint_acc = self.val_acc
            self.checkpoint_data_memory, self.checkpoint_targets_memory = self._get_memory(temp=True)

        #if len(self._multiple_gpus) > 1:
            #temp_net  = temp_net.module


    def best(self):
        self._network.load_state_dict(torch.load('network_checkpoint.pth'))
        self._network.weight_align(self._total_classes - self._known_classes)
        np.save('saved_data_memory{}.npy'.format(self._known_classes), self.checkpoint_data_memory)
        self._targets_memory = (np.concatenate((self._targets_memory, self.checkpoint_targets_memory))
                                if len(self._targets_memory) != 0
                                else self.checkpoint_targets_memory)
        self._data_memory = (np.concatenate((self._data_memory, self.checkpoint_data_memory))
                             if len(self._data_memory) != 0
                             else self.checkpoint_data_memory)
        self.checkpoint_acc = 0
        self.val_acc = 0

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
