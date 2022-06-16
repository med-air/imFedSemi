import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.optim
from options import args_parser
import copy
from utils import losses
from data import dataset

args = args_parser()

class SupervisedLocalUpdate(object):
     def __init__(self, args, trainset, idxs):
          
          self.ldr_train = DataLoader(dataset.DatasetSplit(trainset, idxs), batch_size = args.batch_size, shuffle = True, drop_last=True)
          
          self.epoch = 0
          self.iter_num = 0
          self.base_lr = args.base_lr
         
     def train(self, args, net, op_dict):
          net.cuda()
          net.train()
          self.optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
          self.optimizer.load_state_dict(op_dict)
          
          for param_group in self.optimizer.param_groups:
               param_group['lr'] = self.base_lr
          
          loss_fn = losses.LabelSmoothingCrossEntropy()
        # train and update
          epoch_loss = []
          print(' Supervised training')
          for epoch in range(args.local_ep):
               batch_loss = []
               for i, (_,_, (image_batch, ema_image_batch), label_batch) in enumerate(self.ldr_train):
                    image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()

                    ema_inputs = ema_image_batch 
                    inputs = image_batch 

                    _, outputs, _ = net(inputs)
                    
                    _, aug_outputs, _ = net(ema_inputs)
                 
                    loss_classification = loss_fn(outputs , label_batch.long()) + loss_fn(aug_outputs , label_batch.long())
                    
                    loss = loss_classification 

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_loss.append(loss.item())
                    self.iter_num = self.iter_num + 1

               self.epoch = self.epoch + 1
               epoch_loss.append(np.array(batch_loss).mean())
               print(f' Local Loss: {epoch_loss}')
         
          net.cpu()
          net_states = net.state_dict()
          return net_states, sum(epoch_loss) / len(epoch_loss), copy.deepcopy(self.optimizer.state_dict()) 
