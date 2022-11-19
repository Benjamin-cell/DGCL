import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os

from collections import defaultdict

from models import DGCL
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, buildMPNN

torch.manual_seed(3407)
np.random.seed(3407)

model_name = 'DGCL'
resume_name = ''

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')
parser.add_argument('--ddi', action='store_true', default=False, help="using ddi")

args = parser.parse_args()
model_name = args.model_name
resume_name = args.resume_path


### 2022-5-14 the distance loss function
def compute_kl_loss(self, p, q ,pad_mask = None):

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

    # keep dropout and forward twice
    logits = model(x)

    logits2 = model(x)

    # cross entropy loss for classifier
    ce_loss = 0.5 * (cross_entropy_loss(logits, label) + cross_entropy_loss(logits2, label))

    kl_loss = compute_kl_loss(logits, logits2)

    # carefully choose hyper-parameters
    loss = ce_loss + 0.1 * kl_loss

## 2022-5-16 Graph Contrastive Learning part
class ContrastiveLoss(nn.Module):
    def __init__(self, margin, device):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.device = device ### (1-1matched)= target

         ##1.0
    def forward(self, output1, output2, target,size_average=True):

        # print(' we are shape') ##torch.Size([20, 100])torch.Size([20, 100])torch.Size([20])
        target = target.to(self.device)
        distances = (output2 - output1).pow(2).sum(1).to(self.device)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        # print(losses)
        # print("i am lossses")
        # if (losses.mean()).item()
        #     losses =  F.kl_div(F.log_softmax(output1, dim=-1), F.softmax(query, dim=-1), reduction='none')
        #     return 100*losses.mean()
        # else:
        return losses.mean() if size_average else losses.sum()
contra_loss = ContrastiveLoss(2.0, device='cuda:0')

## evalution
def eval(model, data_eval, voc_size, epoch):
    # evaluate

    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0

    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for adm_idx, adm in enumerate(input):


            target_output1 = model(input[:adm_idx+1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)


            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)



        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record)

    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))
    dill.dump(obj=smm_record, file=open('../data/gamenet_records.pkl', 'wb'))
    dill.dump(case_study, open(os.path.join('saved', model_name, 'case_study.pkl'), 'wb'))

    # print('avg med', med_cnt / visit_cnt)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    # def poly1_b_cross_entropy_torch( logits, labels, class_number=112, epsilon=1.0):
    #     poly1 = torch.sum(F.one_hot(labels.to(torch.int64), class_number).float() * F.softmax(logits), dim=-1)
    #
    #     loss_bce = F.binary_cross_entropy_with_logits(torch.FloatTensor(logits), torch.FloatTensor(labels))
    #     poly1_ce_loss = (loss_bce + epsilon * (1 - poly1))
    #     return poly1_ce_loss


    ddi_mask_path = 'D:\missing label\data\ddi_mask_H.pkl'
    molecule_path = r'D:\missing label\data\atc3toSMILES.pkl'
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))
    molecule = dill.load(open(molecule_path, 'rb'))


    data_path = r'D:\missing label\data\records_final.pkl'
    voc_path = r'D:\missing label\data\voc_final.pkl'
    ehr_adj_path = 'D:\missing label\data\ehr_adj_final_newone.pkl'
    ddi_adj_path = 'D:\missing label\data\ddi_A_final_new.pkl'
    device = torch.device('cuda')

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    EPOCH = 71
    LR = 0.00015
    TEST = args.eval
    Neg_Loss = args.ddi
    DDI_IN_MEM = args.ddi
    TARGET_DDI = 0.05
    T = 0.5
    decay_weight = 0.85

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    # model = Retain(voc_size, emb_size=64, device=device)



    model = DGCL(voc_size, ehr_adj, ddi_adj, ddi_mask_H, emb_dim=64,
                    device=device, ddi_in_memory=DDI_IN_MEM)


    if TEST:
        model.load_state_dict(torch.load(open(resume_name, 'rb')))
    model.to(device=device)

    print('parameters', get_n_params(model))
    optimizer = Adam(list(model.parameters()), lr=LR)

    if TEST:
        eval(model, data_test, voc_size, 0)
    else:
        history = defaultdict(list)
        best_epoch = 0
        best_ja = 0
        for epoch in range(EPOCH):
            loss_record1 = []
            start_time = time.time()
            model.train()
            prediction_loss_cnt = 0
            neg_loss_cnt = 0
            for step, input in enumerate(data_train):
                for idx, adm in enumerate(input):


                    seq_input = input[:idx+1]
                    
                    loss1_target = np.zeros((1, voc_size[2]))
                   
                    loss1_target[:, adm[2]] = 1
                    loss3_target = np.full((1, voc_size[2]), -1)
                    for idx, item in enumerate(adm[2]):
                        loss3_target[0][idx] = item

                    target_output1 , a,b= model(seq_input)

                    # loss2 = miss_loss(m, torch.FloatTensor(loss1_target).to(device))


                    loss_contra = contra_loss(a,b,torch.FloatTensor(loss1_target).to(device))
                    #
                    #
                    loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
                    loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))

                    ### 2022-5-12 添加新的损失
                    # loss_bce = F.binary_cross_entropy(target_output1, torch.FloatTensor(loss1_target).to(device))
                    # loss0 = poly1_b_cross_entropy_torch( target_output1, torch.FloatTensor(loss1_target))
                    # loss5 = multilabel_categorical_crossentropy(torch.FloatTensor(loss1_target).to(device),target_output1)

                    loss = 1.1*loss1 + 0.02*loss3+ loss_contra
                        # loss = loss2 + loss1 / (loss1 / loss2).detach() + loss3 / (loss3 / loss2).detach()

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record1.append(loss.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))
            # annealing
            T *= decay_weight

            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)

            history['ja'].append(ja)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            torch.save(model.state_dict(), open( os.path.join('saved', model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb'))
            print('')
            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja


        dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

        # test
        torch.save(model.state_dict(), open(
            os.path.join('saved', model_name, 'final.model'), 'wb'))

        print('best_epoch:', best_epoch)


if __name__ == '__main__':
    main()
