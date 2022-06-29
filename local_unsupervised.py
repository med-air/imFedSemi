from torch.utils.data import DataLoader
import copy
import torch
import torch.optim
import torch.nn.functional as F
import torch.nn as nn
from options import args_parser
from utils import losses
from utils.util import get_timestamp, calculate_bank

args = args_parser()

class UnsupervisedLocalUpdate(object):
     def __init__(self, args, dataset, Pi, priors_corr):
          self.dataset = dataset
          self.ldr_train = DataLoader(self.dataset, batch_size = args.batch_size, shuffle = True, drop_last=True)
          self.epoch = 0
          self.iter_num = 0
          self.flag = True
          self.base_lr = 2e-4
          self.Pi = Pi
          self.priors_corr = priors_corr
          self.temp_bank = []
          self.permanent_bank = set()
          self.real_Pi = list(Pi.numpy())
         
     def train(self, args, net, op_dict, epoch, logging):
          net.cuda()
          net.train()
          self.optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
          self.optimizer.load_state_dict(op_dict)
          loss_fun = nn.CrossEntropyLoss()
          
          for param_group in self.optimizer.param_groups:
               param_group['lr'] = self.base_lr
          
          self.epoch = epoch
          epoch_loss = []
          
          print(" Inference priors")
          Pi = torch.zeros_like(self.Pi.float()).cpu().int()
          
          net.eval()
          self.dataset.re_load()
          self.temp_bank = []

          for i, (items, _, image_batch, label_batch) in enumerate(self.ldr_train):
               image_batch, label_batch = image_batch.cuda(), label_batch.cuda() 
               inputs = image_batch 
          
               representations, logits, outputs = net(inputs)

               max_probs, pseudo_labels = torch.max(outputs.cpu(),dim=1)
               # no confidence filter
               items = list(items)
               
               lp_conf = args.hi_lp
               sample_ids = torch.where(max_probs>lp_conf)
               bank_items = []
               for item_id in list(sample_ids[0].cpu().numpy()):
                    bank_items.append(items[item_id]) 
               
               for bank_item in bank_items :           
                    self.permanent_bank.add(bank_item)

               lp_conf = args.lo_lp
               sample_ids = torch.where(max_probs>lp_conf)
               bank_items = []
               for item_id in list(sample_ids[0].cpu().numpy()):
                    bank_items.append(items[item_id]) 
               
               self.temp_bank = self.temp_bank + bank_items

          
          net.eval()
          self.dataset.update(calculate_bank(self.temp_bank, self.permanent_bank))
          
          for i, (items, _, image_batch, label_batch) in enumerate(self.ldr_train):
        
               image_batch, label_batch = image_batch.cuda(), label_batch.cuda() 
               inputs = image_batch 
          
               representations, logits, outputs = net(inputs)

               max_probs, pseudo_labels = torch.max(outputs.cpu(),dim=1)
               label_batch = list(label_batch.int().cpu().numpy())
               for i in range(len(label_batch)):
                    Pi[label_batch[i]][pseudo_labels[i]] += 1
              
               
          
          Pi = Pi.float().cuda()
          Pi = F.normalize(Pi, p=1)
          priors_corr = self.priors_corr.float().cuda()

          print(' Unsupervised training')
          net.train()
          self.dataset.update(self.temp_bank)
    
          for epoch in range(args.local_ep):

               batch_loss = []
               iter_max = len(self.ldr_train)

               for i, (_, _, image_batch, label_batch) in enumerate(self.ldr_train):
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda() 
                    inputs = image_batch 
  
                    representations, logits, outputs = net(inputs, Pi=Pi, priors_corr=priors_corr)
                 
                    loss = loss_fun(outputs, label_batch.long()) 

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    batch_loss.append(loss.item())
                    
                    self.iter_num = self.iter_num + 1
 
               self.epoch = self.epoch + 1
               epoch_loss.append(sum(batch_loss) / len(batch_loss))
               print(f' Local Loss: {epoch_loss}')
          net.cpu()
          net_states = net.state_dict()
          return net_states, sum(epoch_loss) / len(epoch_loss), copy.deepcopy(self.optimizer.state_dict()) 
    
