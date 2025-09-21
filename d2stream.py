import torch
import torch.nn as nn
import torch.distributed as dist
from utils.distributed import all_reduce
import time, os, subprocess, pandas, tqdm
import torch.nn.functional as F
import numpy as np
from loss_multi import lossAV, lossA, lossV
from d2s.d2sEncoder import d2sEncoder
from subprocess import PIPE
import re


class D2stream(nn.Module):

    def __init__(self):
        super(D2stream, self).__init__()
        self.model = d2sEncoder()
        self.lossAV = lossAV()
        self.lossA = lossA()
        self.lossV = lossV()
        self.criterion = nn.CrossEntropyLoss()

    def talknce_loss(self, features_vis, features_aud, labels):
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to('cuda')
        features_vis = F.normalize(features_vis, dim=1)
        features_aud = F.normalize(features_aud, dim=1)
        similarity_matrix = torch.matmul(features_vis, features_aud.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to('cuda')
        positives = similarity_matrix[mask].view(labels.shape[0], -1)
        negatives = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to('cuda')
        logits = logits / 0.07
        return logits, labels

    def forward(self, audioFeature, visualFeature, labels, masks):
        b, s, t = visualFeature.shape[:3]
        visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
        labels = labels.view(b * s, *labels.shape[2:])
        masks = masks.view(b * s, *masks.shape[2:])
        audioEmbed = self.model.forward_audio_frontend(audioFeature)
        visualEmbed = self.model.forward_visual_frontend(visualFeature)
        audioEmbed = audioEmbed.repeat(s, 1, 1)
        new_labels = labels[0].reshape((-1))
        tri_vis = visualEmbed[0].reshape(-1, 128)
        tri_aud = audioEmbed[0].reshape(-1, 128)
        active_index = np.where(new_labels.cpu() == 1)

        if len(active_index[0]) > 0:
            tri_vis2 = torch.stack([tri_vis[i, :] for i in active_index[0]], dim=0)
            tri_aud2 = torch.stack([tri_aud[j, :] for j in active_index[0]], dim=0)
            nce_label = torch.ones_like(torch.Tensor(active_index[0])).to('cuda')

            logits, labels_nce = self.talknce_loss(tri_vis2, tri_aud2, nce_label)
            nce_loss = self.criterion(logits, labels_nce)
        else:
            nce_loss = torch.tensor(0.0, device=labels.device)

        audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
        outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, b, s)
        outsA = self.model.forward_audio_backend(audioEmbed)
        outsV = self.model.forward_visual_backend(visualEmbed)
        labels = labels.reshape((-1))
        masks = masks.reshape((-1))
        nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels, masks)
        nlossA = self.lossA.forward(outsA, labels, masks)
        nlossV = self.lossV.forward(outsV, labels, masks)
        nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV + 0.3 * nce_loss
        num_frames = masks.sum()
        return nloss, nlossA, nlossV, nlossAV, prec, num_frames, nce_loss


class d2stream(nn.Module):

    def __init__(self, rank=None, device=None):
        super(d2stream, self).__init__()
        self.rank = rank
        if rank != None:
            self.rank = rank
            self.device = device
            self.model = D2stream().to(device)
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[rank],
                                                             output_device=rank,
                                                             find_unused_parameters=False)
            self.optim = torch.optim.Adam(self.model.parameters(), lr=5e-5)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim,
                                                             step_size=1,
                                                             gamma=0.95)
        else:
            self.model = d2sEncoder().cuda()
            self.lossAV = lossAV().cuda()
            self.lossA = lossA().cuda()
            self.lossV = lossV().cuda()

        print(
            time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2fM" %
            (sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader, num_gpus):
        self.model.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lossA, lossV, lossAV, r_nce_loss = 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        loader.sampler.set_epoch(epoch)
        device = self.device

        pbar = enumerate(loader, start=1)
        if self.rank == 0:
            pbar = tqdm.tqdm(pbar, total=loader.__len__())

        for num, (audioFeature, visualFeature, labels, masks) in pbar:

            audioFeature = audioFeature.to(device)
            visualFeature = visualFeature.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            nloss, nlossA, nlossV, nlossAV, prec, num_frames, nce_loss = self.model(
                audioFeature,
                visualFeature,
                labels,
                masks,
            )
            self.optim.zero_grad()
            nloss.backward()
            self.optim.step()
            [nloss, nlossA, nlossV, nlossAV, prec, num_frames, nce_loss] = all_reduce([nloss, nlossA, nlossV, nlossAV, prec, num_frames, nce_loss], average=False)
            top1 += prec.detach().cpu().numpy()
            lossA += nlossA.detach().cpu().numpy()
            lossV += nlossV.detach().cpu().numpy()
            lossAV += nlossAV.detach().cpu().numpy()
            loss += nloss.detach().cpu().numpy()
            index += int(num_frames.detach().cpu().item())
            r_nce_loss += nce_loss.detach().cpu().numpy()
            if self.rank == 0:
                pbar.set_postfix(
                    dict(epoch=epoch,
                         lossA=lossA / (num * num_gpus),
                         lossV=lossV / (num * num_gpus),
                         lossAV=lossAV / (num * num_gpus),
                         nce_loss=r_nce_loss / (num * num_gpus),
                         loss=loss / (num * num_gpus),
                         acc=(top1 / index)))
        dist.barrier()
        return loss / num, lr

    def evaluate_network(self, epoch, loader):
        self.eval()
        predScores = []
        workspace = './save/workspace1'
        os.makedirs(workspace, exist_ok=True)
        evalCsvSave = os.path.join(workspace, "{}_res.csv".format(epoch))
        evalOrig = "../../data/AVADataPath/csv/val_orig.csv"
        for audioFeature, visualFeature, labels, masks in tqdm.tqdm(loader):
            with torch.no_grad():
                audioFeature = audioFeature.cuda()
                visualFeature = visualFeature.cuda()
                labels = labels.cuda()
                masks = masks.cuda()
                b, s, t = visualFeature.shape[0], visualFeature.shape[1], visualFeature.shape[2]
                visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
                labels = labels.view(b * s, *labels.shape[2:])
                masks = masks.view(b * s, *masks.shape[2:])
                audioEmbed = self.model.forward_audio_frontend(audioFeature)
                visualEmbed = self.model.forward_visual_frontend(visualFeature)
                audioEmbed = audioEmbed.repeat(s, 1, 1)
                audioEmbed, visualEmbed = self.model.forward_cross_attention(
                    audioEmbed, visualEmbed)
                outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, b, s)
                labels = labels.reshape((-1))
                masks = masks.reshape((-1))
                outsAV = outsAV.view(b, s, t, -1)[:, 0, :, :].view(b * t, -1)
                labels = labels.view(b, s, t)[:, 0, :].view(b * t).cuda()
                masks = masks.view(b, s, t)[:, 0, :].view(b * t)
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels, masks)
                predScore = predScore[:, 1].detach().cpu().numpy()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = pandas.Series(['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1, inplace=True)
        evalRes.drop(['instance_id'], axis=1, inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O ./utils/get_ava_active_speaker_performance.py -g %s -p %s " % (evalOrig, evalCsvSave)
        process = subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout = process.stdout.decode('utf-8')
        stderr = process.stderr.decode('utf-8')

        if process.returncode != 0:
            print(f"Command execution failed, return code: {process.returncode}")
            mAP = 0
        else:
            match = re.search(r'average precision:\s*([\d\.]+)%', stdout)
            if match:
                mAP = float(match.group(1))
            else:
                print("Overall Average Precision not found in the output")
                mAP = 0
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location='cpu')
        if self.rank != None:
            info = self.load_state_dict(loadedState)
        else:
            new_state = {}

            for k, v in loadedState.items():
                new_state[k.replace("model.module.", "")] = v
            info = self.load_state_dict(new_state, strict=False)
        print(info)
