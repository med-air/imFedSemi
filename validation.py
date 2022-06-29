import numpy as np
import torch
from torch.nn import functional as F

from utils.metrics import  compute_metrics_test
from utils.losses import LabelSmoothingCrossEntropy


def epochVal_metrics_test(model, dataLoader, thresh):
    training = model.training
    model.eval()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    gt_study   = {}
    pred_study = {}
    studies    = []
    losses = []
    loss_fn = LabelSmoothingCrossEntropy()

    with torch.no_grad():
        for i, (study, _, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            _, output, _ = model(image)
         
            output = F.softmax(output, dim=1)
            loss = loss_fn(output , label.long())
            losses.append(loss.item())
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
    loss = np.array(losses).mean()
    return AUROCs, Accus, Senss, Specs, pre,F1, loss
