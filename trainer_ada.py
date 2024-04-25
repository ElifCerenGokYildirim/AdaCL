import sys
import logging
import copy
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from utils import factory
from utils.data_manager_ada import DataManager
from utils.toolkit import count_parameters
import os


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top2": []}, {"top1": [], "top2": []}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.load_data(data_manager)
        if task == 0:
            model.inc_train(trial=None, data_manager=data_manager)
        else:
            study = optuna.create_study(sampler=TPESampler(n_startup_trials=5),
                                        pruner=SuccessiveHalvingPruner(min_resource = 10),
                                        direction="maximize")
            study.optimize(lambda trial: model.inc_train(trial, data_manager), n_trials=40)
            for trial in study.trials:
             best_params = study.best_params
            print("Best Hyperparameters")
            for key, value in best_params.items():
             print(f"{key}: {value}")

            model.best()

        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top2"].append(cnn_accy["top2"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top2"].append(nme_accy["top2"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top2 curve: {}".format(cnn_curve["top2"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top2 curve: {}\n".format(nme_curve["top2"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"]) / len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"]) / len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top2"].append(cnn_accy["top2"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top2 curve: {}\n".format(cnn_curve["top2"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
