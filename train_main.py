from validation import epochVal_metrics_test
from options import args_parser
import os
import sys
import logging
import random
import numpy as np
import pandas as pd
import copy
from FedAvg import FedAvg
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from networks.models import DenseNet121
from data import dataset
from local_supervised import SupervisedLocalUpdate
from local_unsupervised import UnsupervisedLocalUpdate
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")


def test(args, epoch, net=None, save_mode_path=None, val=False):
    if net is not None:
        model = net.cuda()
    else:
        checkpoint_path = save_mode_path 
        checkpoint = torch.load(checkpoint_path)
        net = DenseNet121(out_size=args.class_num, mode=args.label_uncertainty, drop_rate=args.drop_rate)
        model = net.cuda()
        model.load_state_dict(checkpoint['state_dict'])

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    
    if val:
        if args.dataset == 'brain':
            val_path =  'data/brain_split/dict_users_val.npy'
        else:
            val_path = 'data/skin_split/dict_users_val.npy'
        
        dict_user = np.load(val_path, allow_pickle=True).item()
        csv_file = args.csv_file_val
    else:
        if args.dataset == 'brain':
            test_path =  'data/brain_split/dict_users_test.npy'
        else:
            test_path = 'data/skin_split/dict_users_test.npy'

        dict_user = np.load(test_path, allow_pickle=True).item()
        csv_file = args.csv_file_test

    client_AUC = []
    client_Acc = []
    client_Sen = []
    client_Spe = []
    client_Pre = []
    client_F1  = []
    for key in dict_user.keys():
        testset = dataset.CSVDataset(root_dir=args.root_path,
                                          csv_file=csv_file,
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))

        dataloader = DataLoader(dataset=dataset.DatasetSplit(testset, dict_user[key]), batch_size=args.batch_size,
                                shuffle=False, num_workers=6, pin_memory=True)
                                
        AUROCs, Accus, Senss, Specs, Preci, F1, loss = epochVal_metrics_test(model, dataloader, thresh=0.4)  
        AUROC_avg = np.array(AUROCs).mean(); client_AUC.append(round(AUROC_avg,6))
        Accus_avg = np.array(Accus).mean();  client_Acc.append(round(Accus_avg,6))
        Senss_avg = np.array(Senss).mean();  client_Sen.append(round(Senss_avg,6))
        Specs_avg = np.array(Specs).mean();  client_Spe.append(round(Specs_avg,6))
        Preci_avg = np.array(Preci).mean();  client_Pre.append(round(Preci_avg,6))
        F1_avg = np.array(F1).mean();        client_F1.append(round(F1_avg,6))
 
    return client_AUC, client_Acc, client_Sen, client_Spe, client_F1, loss


def prepare_data(args, supervised_user_id, unsupervised_user_id):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
    
    trans = transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize,
                                        ])
    sup_trans = dataset.TransformTwice(transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize,
                                        ]))
 
    # Supervise
    train_dataset = dataset.CSVDataset(root_dir=args.root_path,
                                        csv_file=args.csv_file_train,
                                        transform=sup_trans)
    
    if args.dataset =='brain':
        train_path = 'data/brain_split/dict_users_train.npy'
    else:
        train_path = 'data/skin_split/dict_users_train.npy'
    dict_users_train = np.load(train_path, allow_pickle=True).item()
   
    
    
    # Unsupervise 
    client_train_data, client_priors_corr, client_Pi = dataset.load_data(args, train_dataset, [dict_users_train[i] for i in unsupervised_user_id], sub_bank_num_perclient=args.sub_bank_num, clientnum=len(unsupervised_user_id), classnum=args.class_num)
   
    unsup_train_datasets = dict()
    for idx, uid in enumerate(unsupervised_user_id):
        unsup_train_datasets[uid] = dataset.BaseDataset(root_dir=args.root_path,
                                                images=client_train_data[idx]['images'], 
                                                labels=client_train_data[idx]['labels'], 
                                                transform=trans)
       
    return dict_users_train, train_dataset, unsup_train_datasets, client_priors_corr, client_Pi

if __name__ == '__main__':
    args = args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''FL Setings'''
    
    metrics_log = {
        'train_loss':[],
        'val_loss':[],
        'val_auc':[],
        'val_acc':[],
        'val_sen':[],
        'val_spe':[],
        'val_f1':[],
    }
    test_metrics = {
        'test_loss':[],
        'test_auc':[],
        'test_acc':[],
        'test_sen':[],
        'test_spe':[],
        'test_f1':[],
    }
    best_auc = 0
    

    supervised_user_id = []
    unsupervised_user_id = [0,1,2,3,4,5,6,7,8,9]

    flag_create = False
    WARMUP = args.warmup
    EVAL=1
    # num = len(unsupervised_user_id)
    args.class_num = 5 if args.dataset == 'brain' else 7
    args.sub_bank_num = 5 if args.dataset == 'brain' else 7
    snapshot_path = f'./models/{args.dataset}/'
 
    if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)
    if not os.path.exists(f'./logs/{args.dataset}'): os.makedirs(f'./logs/{args.dataset}')
    print('Exp path:', snapshot_path)
    logging.basicConfig(filename=f'./logs/{args.dataset}/log.txt', 
                    level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)


    # Prepare data
    dict_users_train, train_dataset, unsup_train_datasets, client_priors_corr, client_Pi = prepare_data(args, supervised_user_id, unsupervised_user_id)

    # Model    
    net_glob = DenseNet121(out_size=args.class_num, mode=args.label_uncertainty, drop_rate=args.drop_rate)
    net_glob.train()

    # Parameters setup
    w_glob = net_glob.state_dict()

    w_locals = []
    w_sup_last = []
    w_locals_backbone = []
    w_locals_classifier = []
    w_locals_projector = []

    trainer_locals = []
    net_locals = []
    optim_locals = []
    alternate_comm = 'Unsup'

    '''supervised server setup'''
    server_trainer = SupervisedLocalUpdate(args, train_dataset, dict_users_train['server'])
    server_net = copy.deepcopy(net_glob).cuda()
    optimizer = torch.optim.Adam(server_net.parameters(), lr=args.base_lr, 
                                betas=(0.9, 0.999), weight_decay=5e-4)
    server_optim = copy.deepcopy(optimizer.state_dict())

    for i in unsupervised_user_id :
        trainer_locals.append(UnsupervisedLocalUpdate(args, unsup_train_datasets[i], client_Pi[i], client_priors_corr[i]))

    for com_round in range(args.rounds):
        print(f"\n=== Round {com_round} ===")
        loss_locals = []

        if com_round * args.local_ep <= WARMUP:
            w, loss, op = server_trainer.train(args, server_net, server_optim)
            w_glob = copy.deepcopy(w)
            net_glob.load_state_dict(w_glob)
            server_optim = copy.deepcopy(op)
            loss_locals.append(copy.deepcopy(loss))
        else:        
            '''Client training'''
            if  not flag_create : 
                print('Unsupervised clients join')
                for i in unsupervised_user_id : 
                    w_locals.append(copy.deepcopy(w_glob))
                    net_locals.append(copy.deepcopy(net_glob).cuda())
                    optimizer = torch.optim.Adam(net_locals[i].parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
                    optim_locals.append(copy.deepcopy(optimizer.state_dict()))
                flag_create = True
            for idx in unsupervised_user_id :
                local = trainer_locals[idx]
                optimizer = optim_locals[idx]
                w, loss, op = local.train(args, net_locals[idx], optimizer, com_round*args.local_ep, logging)

                w_locals[idx] = copy.deepcopy(w)
                optim_locals[idx] = copy.deepcopy(op)
                loss_locals.append(copy.deepcopy(loss))
            # Aggregation           
            with torch.no_grad():
                w_glob = FedAvg(w_locals)
            net_glob.load_state_dict(w_glob)
            server_net.load_state_dict(w_glob)     

            '''Server training'''
            server_trainer.base_lr = 3e-4
            w, loss, op = server_trainer.train(args, server_net, server_optim)
            # update global model on server
            w_glob = copy.deepcopy(w)
            net_glob.load_state_dict(w_glob)
            server_optim = copy.deepcopy(op)
            loss_locals.append(copy.deepcopy(loss))
            
            '''Broadcast clients models'''
            for i in unsupervised_user_id:
                net_locals[i].load_state_dict(w_glob)
 
        loss_avg = sum(loss_locals) / len(loss_locals)
        metrics_log['train_loss'].append(loss_avg)
        logging.info('Loss Avg {} Round {} LR {} '.format(loss_avg, com_round, args.base_lr))

        # Evaluation and Test
        if com_round % EVAL == 0:
            client_AUC, client_Acc, client_Sen, client_Spe, client_F1, val_loss = test(args, com_round, net_glob, None, True)
            client_AUC_avg, client_Acc_avg, client_Sen_avg, client_Spe_avg, client_F1_avg = np.mean(client_AUC), np.mean(client_Acc), np.mean(client_Sen), np.mean(client_Spe), np.mean(client_F1)
            
            metrics_log['val_auc'].append(client_AUC_avg)
            metrics_log['val_acc'].append(client_Acc_avg)
            metrics_log['val_sen'].append(client_Sen_avg)
            metrics_log['val_spe'].append(client_Spe_avg)
            metrics_log['val_f1'].append(client_F1_avg)
            metrics_log['val_loss'].append(val_loss)
            logging.info("\nVal Epoch: {}".format(com_round))
            logging.info("Val AUC: {:6f}, Acc: {:6f}, Sen: {:6f}, Spe: {:6f}, F1: {:6f}"
                    .format(client_AUC_avg, client_Acc_avg, client_Sen_avg, client_Spe_avg, client_F1_avg))

            # save better model
            if client_AUC_avg > best_auc:
                best_auc = client_AUC_avg
                save_mode_path = os.path.join(snapshot_path, 'best_' + str(com_round) + '.pth')
                torch.save({
                            'state_dict': net_glob.state_dict(),
                            }
                            , save_mode_path)    
                
                client_AUC, client_Acc, client_Sen, client_Spe, client_F1, test_loss = test(args, com_round, None, save_mode_path, False)
                client_AUC_avg, client_Acc_avg, client_Sen_avg, client_Spe_avg, client_F1_avg = np.mean(client_AUC), np.mean(client_Acc), np.mean(client_Sen), np.mean(client_Spe), np.mean(client_F1)
                
                logging.info("\nBest Test Epoch: {}".format(com_round))
                logging.info("Best Test AUC: {:6f}, Acc: {:6f}, Sen: {:6f}, Spe: {:6f}, F1: {:6f}"
                    .format(client_AUC_avg, client_Acc_avg, client_Sen_avg, client_Spe_avg, client_F1_avg))

        # Save every 10 epoch
        if com_round % 10 == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(com_round) + '.pth')
            torch.save({
                        'state_dict': net_glob.state_dict(),
                        }
                        , save_mode_path)
            client_AUC, client_Acc, client_Sen, client_Spe, client_F1, test_loss = test(args, com_round, None, save_mode_path, False)
            logging.info("\nTEST Epoch: {}".format(com_round))
            logging.info("TEST AUC: {:6f}, Acc: {:6f}, Sen: {:6f}, Spe: {:6f}, F1: {:6f}"
                    .format(np.mean(client_AUC), np.mean(client_Acc), np.mean(client_Sen), np.mean(client_Spe), np.mean(client_F1)))
            test_metrics['test_auc'].append(client_AUC_avg)
            test_metrics['test_acc'].append(client_Acc_avg)
            test_metrics['test_sen'].append(client_Sen_avg)
            test_metrics['test_spe'].append(client_Spe_avg)
            test_metrics['test_f1'].append(client_F1_avg)
            test_metrics['test_loss'].append(val_loss)
   
    metrics_pd = pd.DataFrame.from_dict(metrics_log)
    metrics_pd.to_csv(os.path.join(snapshot_path,"val_metrics.csv"))
    metrics_pd = pd.DataFrame.from_dict(test_metrics)
    metrics_pd.to_csv(os.path.join(snapshot_path,"test_metrics.csv"))

            

            
