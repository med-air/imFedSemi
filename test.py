import torch
from networks.models import DenseNet121
from options import args_parser
import numpy as np
from torchvision import transforms
import os
from data import dataset
from  torch.utils.data import DataLoader
import numpy as np
import torch.backends.cudnn as cudnn
import random
from utils.metrics import  compute_metrics_test
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

args = args_parser()
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def epochVal_metrics_test(model, dataLoader, thresh):
    training = model.training
    model.eval()

    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)
    
    gt_study   = {}
    pred_study = {}
    studies    = []

    with torch.no_grad():
        for i, (study, _, image, label) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)
            _, output, _ = model(image)
         
            output = F.softmax(output, dim=1)
            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

          
        
        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)
       
        AUROCs, Accus, Senss, Specs, pre, F1 = compute_metrics_test(gt, pred,  thresh=thresh, competition=True)
    
    model.train(training)
    return AUROCs, Accus, Senss, Specs, pre, F1


def test(args, save_mode_path=None):
    checkpoint_path = save_mode_path 
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net = DenseNet121(out_size=args.class_num, mode=args.label_uncertainty, drop_rate=args.drop_rate)
    model = net.to(device)
    model.load_state_dict(checkpoint['state_dict'])

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_path =  'data/split/dict_users_test.npy'

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
                                
        AUROCs, Accus, Senss, Specs, Preci, F1 = epochVal_metrics_test(model, dataloader, thresh=0.4)  
        AUROC_avg = np.array(AUROCs).mean(); client_AUC.append(round(AUROC_avg,6))
        Accus_avg = np.array(Accus).mean();  client_Acc.append(round(Accus_avg,6))
        Senss_avg = np.array(Senss).mean();  client_Sen.append(round(Senss_avg,6))
        Specs_avg = np.array(Specs).mean();  client_Spe.append(round(Specs_avg,6))
        Preci_avg = np.array(Preci).mean();  client_Pre.append(round(Preci_avg,6))
        F1_avg = np.array(F1).mean();        client_F1.append(round(F1_avg,6))
 
    return client_AUC, client_Acc, client_Sen, client_Spe, client_F1



client_AUC, client_Acc, client_Sen, client_Spe, client_F1 = test(args, args.model_path)
print("TEST AUC: {:6f}, Acc: {:6f}, Sen: {:6f}, Spe: {:6f}, F1: {:6f}"
        .format(np.mean(client_AUC), np.mean(client_Acc), np.mean(client_Sen), np.mean(client_Spe), np.mean(client_F1)))