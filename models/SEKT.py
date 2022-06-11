import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics

class SEKT(Module):

    def __init__(self, num_q, num_s, emb_sizeq, emb_sizes, hidden_size):
        super().__init__()
        self.num_q = num_q
        self.num_s = num_s
        self.emb_sizeq = emb_sizeq
        self.emb_sizes = emb_sizes
        self.hidden_size = hidden_size

        self.interaction_embq = Embedding(self.num_q * 2, self.emb_sizeq)
        self.interaction_embs = Embedding(self.num_s * 2, self.emb_sizes)

        self.lstm_layerq = LSTM(
            self.emb_sizeq, self.hidden_size, batch_first=True
        )
        self.lstm_layers = LSTM(
            self.emb_sizes, self.hidden_size, batch_first=True
        )
        self.out_layerq = Linear(self.hidden_size, self.num_q)
        self.out_layers = Linear(self.hidden_size, self.num_s)
        self.dropout_layer = Dropout()

    def forward(self, q, r, flag):
        if(flag == 'q'):
          x = q + self.num_q * r

          h, _ = self.lstm_layerq(self.interaction_embq(x))
          y = self.out_layerq(h)
          y = self.dropout_layer(y)
          y = torch.sigmoid(y)

          return y

        elif(flag == 's'):
          x = q + self.num_s * r

          h, _ = self.lstm_layers(self.interaction_embs(x))
          y = self.out_layers(h)
          y = self.dropout_layer(y)
          y = torch.sigmoid(y)

          return y

    def train_model(
        self, train_loader, test_loader, num_epochs, opt, ckpt_path
    ):
        aucs = []
        loss_means = []

        max_auc = 0

        for i in range(1, num_epochs + 1):
            loss_mean = []

            for data in train_loader:
                q, s, rq, rs, qshft, sshft, rqshft, rsshft, m = data

                self.train()

                y = self(q.long(), rq.long(), 'q')
                y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                y_s = self(s.long(), rs.long(), 's')
                y_s = (y_s * one_hot(sshft.long(), self.num_s)).sum(-1)
                
                y = torch.masked_select(y, m)
                t = torch.masked_select(rqshft, m)

                opt.zero_grad()

                sen_loss = 0
                count = 0
                for o in range(len(q)):
                  save_sen = [] 
                  for k in range(len(q[0])):
                    if(m[o][k]):
                      if(save_sen.count(s[o][k]) == 0):
                        save_sen.append(s[o][k])

                      elif(save_sen.count(s[o][k]) > 0 and s[o][k] != s[o][k-1]): 

                        sen_loss += pow(2,-save_sen.count(s[o][k])) \
                        * binary_cross_entropy( ( (y_s[o][k] + (y_s[o][k]*pow(2,-save_sen.count(s[o][k])))).clone().detach() if y_s[o][k] + (y_s[o][k]*pow(2,-save_sen.count(s[o][k]))) <=1 else torch.tensor(1.0)) , rsshft[o][k] ) # 지수함수(2)만큼 가중치

                        save_sen.append(s[o][k])
                        count += 1
                        
                    else:
                      break

                sen_loss = sen_loss/count
                loss = binary_cross_entropy(y, t)
                loss = sen_loss + loss
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, s, rq, rs, qshft, sshft, rqshft, rsshft, m = data

                    self.eval()

                    y = self(q.long(), rq.long(),'q')
                    y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                    y = torch.masked_select(y, m).detach().cpu()
                    y_foracc = np.where(y > 0.5, 1 , 0)
                    t = torch.masked_select(rqshft, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=y.numpy()
                    )
                    acc = metrics.accuracy_score(
                        y_true=t.numpy(), y_pred=y_foracc
                    )

                    loss_mean = np.mean(loss_mean)

                    print(
                        "Epoch: {},   AUC: {},    ACC: {},    Loss Mean: {}"
                        .format(i, auc, acc, loss_mean)
                    )

                    if auc > max_auc:
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "model_{}.ckpt".format(auc)
                            )
                        )
                        max_auc = auc

                    aucs.append(auc)
                    loss_means.append(loss_mean)

        return aucs, loss_means
