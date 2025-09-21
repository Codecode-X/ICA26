##
# This code is modified based on https://github.com/SJTUwxz/LoCoNet_ASD.git
##

import os, torch, warnings, glob, json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from utils.tools import *
import torch.multiprocessing as mp
import torch.distributed as dist
from dataLoader_multiperson import train_loader
from d2stream import d2stream
import csv


dataPath = "../../data/AVADataPath/"
ori_file = "../../data/AVADataPath/csv/train_orig.csv"
trianFileName = "../../data/AVADataPath/csv/train_loader.csv"
valFileName = "../../data/AVADataPath/csv/val_loader.csv"
video_root = "../../data/AVADataPath/clips_videos/train"
audio_root = "../../data/AVADataPath/clips_audios/train"

class MyCollator(object):
    def __init__(self,world_size):
        self.world_size = world_size

    def __call__(self, data):
        audiofeatures = [item[0] for item in data]
        visualfeatures = [item[1] for item in data]
        labels = [item[2] for item in data]
        masks = [item[3] for item in data]
        cut_limit = 200
        lengths = torch.tensor([t.shape[1] for t in audiofeatures])
        max_len = max(lengths)
        padded_audio = torch.stack([
            torch.cat([i, i.new_zeros((i.shape[0], max_len - i.shape[1], i.shape[2]))], 1)
            for i in audiofeatures
        ], 0)

        if max_len > cut_limit * 4:
            padded_audio = padded_audio[:, :, :cut_limit * 4, ...]

        lengths = torch.tensor([t.shape[1] for t in visualfeatures])
        max_len = max(lengths)
        padded_video = torch.stack([
            torch.cat(
                [i, i.new_zeros((i.shape[0], max_len - i.shape[1], i.shape[2], i.shape[3]))], 1)
            for i in visualfeatures
        ], 0)
        padded_labels = torch.stack(
            [torch.cat([i, i.new_zeros((i.shape[0], max_len - i.shape[1]))], 1) for i in labels], 0)
        padded_masks = torch.stack(
            [torch.cat([i, i.new_zeros((i.shape[0], max_len - i.shape[1]))], 1) for i in masks], 0)

        if max_len > cut_limit:
            padded_video = padded_video[:, :, :cut_limit, ...]
            padded_labels = padded_labels[:, :, :cut_limit, ...]
            padded_masks = padded_masks[:, :, :cut_limit, ...]

        return padded_audio, padded_video, padded_labels, padded_masks


class DataPrep():
    def __init__(self, world_size, rank,entity_data,ts_to_entity):
        self.world_size = world_size
        self.rank = rank
        self.entity_data = entity_data
        self.ts_to_entity = ts_to_entity

    def train_dataloader(self):
        loader = train_loader(trialFileName="../../data/AVADataPath/csv/train_loader.csv", \
                              audioPath="../../data/AVADataPath/clips_audios/train", \
                              visualPath="../../data/AVADataPath/clips_videos/train", \
                              entity_data=self.entity_data, \
                              ts_to_entity=self.ts_to_entity, \
                              num_speakers=3,
                              )
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            loader, num_replicas=self.world_size, rank=self.rank)
        collator = MyCollator(self.world_size)
        trainLoader = torch.utils.data.DataLoader(loader,
                                                  batch_size=1,
                                                  pin_memory=False,
                                                  num_workers=6,
                                                  collate_fn=collator,
                                                  sampler=train_sampler)
        return trainLoader


def main(gpu, world_size,entity_data,ts_to_entity,NUM_GPUS):
    NUM_GPUS=NUM_GPUS
    rank = gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    make_deterministic(seed=int(20210617))
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:{}".format(gpu))

    warnings.filterwarnings("ignore")

    data = DataPrep(world_size, rank,entity_data,ts_to_entity)

    modelSavePath = "./save/exps/exp_08_13"
    os.makedirs(modelSavePath, exist_ok=True)
    modelfiles = glob.glob('%s/model_0*.model' % modelSavePath)
    modelfiles.sort()
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!" % modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = d2stream(rank, device)
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = d2stream(rank, device)

    while (1):
        loss, lr = s.train_network(epoch=epoch, loader=data.train_dataloader(),num_gpus=NUM_GPUS)

        s.saveParameters(modelSavePath + "/model_%04d.model" % epoch)

        if epoch >= 35:
            quit()

        epoch += 1


if __name__ == '__main__':
    entity_data = {}
    speech_data = {}
    ts_to_entity = {}

    def csv_to_list(csv_path):
        as_list = None
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            as_list = list(reader)
        return as_list

    def postprocess_speech_label(speech_label):
        speech_label = int(speech_label)
        if speech_label == 2:
            speech_label = 0
        return speech_label

    def cache_entity_data(csv_file_path):
        entity_set = set()

        csv_data = csv_to_list(csv_file_path)
        csv_data.pop(0)
        for csv_row in csv_data:
            video_id = csv_row[0]
            entity_id = csv_row[-3]
            timestamp = csv_row[1]
            speech_label = postprocess_speech_label(csv_row[-2])

            if video_id not in entity_data.keys():
                entity_data[video_id] = {}
            if entity_id not in entity_data[video_id].keys():
                entity_data[video_id][entity_id] = {}
                entity_set.add((video_id, entity_id))
            entity_data[video_id][entity_id][timestamp] = speech_label

            if video_id not in speech_data.keys():
                speech_data[video_id] = {}
            if timestamp not in speech_data[video_id].keys():
                speech_data[video_id][timestamp] = speech_label

            new_speech_label = max(
                speech_data[video_id][timestamp], speech_label)
            speech_data[video_id][timestamp] = new_speech_label

        return entity_set


    def entity_list_postprocessing(entity_set, video_root):
        for video_id, entity_id in entity_set.copy():
            exist_entity = os.path.join(video_root, video_id, entity_id)
            if not os.path.exists(exist_entity):
                entity_set.remove((video_id, entity_id))

        for video_id, entity_id in entity_set.copy():
            dir = os.path.join(video_root, video_id, entity_id)
            if len(os.listdir(dir)) != len(entity_data[video_id][entity_id]):
                entity_set.remove((video_id, entity_id))

        entity_list = sorted(list(entity_set))

        for video_id, entity_id in entity_set:
            if video_id not in ts_to_entity.keys():
                ts_to_entity[video_id] = {}

            ent_min_data = entity_data[video_id][entity_id].keys()
            for timestamp in ent_min_data:
                if timestamp not in ts_to_entity[video_id].keys():
                    ts_to_entity[video_id][timestamp] = []
                ts_to_entity[video_id][timestamp].append(entity_id)
        return entity_list


    def clean_entity_data(entity_list, entity_data):
        valid_entities = set(entity_list)
        for video_id in list(entity_data.keys()):
            for entity_id in list(entity_data[video_id].keys()):
                if (video_id, entity_id) not in valid_entities:
                    del entity_data[video_id][entity_id]

            if not entity_data[video_id]:
                del entity_data[video_id]
        return entity_data

    cache_dir = os.path.join(dataPath, "json", "train")
    entity_data_cache_file = os.path.join(cache_dir, "entity_data.json")
    ts_to_entity_cache_file = os.path.join(cache_dir, "ts_to_entity.json")
    speech_data_cache_file = os.path.join(cache_dir, "speech_data.json")
    entity_list_cache_file = os.path.join(cache_dir, "entity_list.json")

    if (os.path.exists(entity_data_cache_file) and
            os.path.exists(ts_to_entity_cache_file) and
            os.path.exists(speech_data_cache_file) and
            os.path.exists(entity_list_cache_file)):
        print("Loading cached data...")
        with open(entity_data_cache_file, 'r') as f:
            entity_data = json.load(f)
        with open(ts_to_entity_cache_file, 'r') as f:
            ts_to_entity = json.load(f)
        with open(speech_data_cache_file, 'r') as f:
            speech_data = json.load(f)
        with open(entity_list_cache_file, 'r') as f:
            entity_list = [tuple(item) for item in json.load(f)]
        print("Cached data loaded successfully.")
    else:
        print("Cached data not found. Generating data...")
        entity_data = {}
        speech_data = {}
        ts_to_entity = {}

        entity_set = cache_entity_data(ori_file)
        entity_list = entity_list_postprocessing(entity_set, video_root)
        entity_data = clean_entity_data(entity_list, entity_data)

        os.makedirs(cache_dir, exist_ok=True)

        print("Saving generated data to cache...")
        with open(entity_data_cache_file, 'w') as f:
            json.dump(entity_data, f)
        with open(ts_to_entity_cache_file, 'w') as f:
            json.dump(ts_to_entity, f)
        with open(speech_data_cache_file, 'w') as f:
            json.dump(speech_data, f)
        with open(entity_list_cache_file, 'w') as f:
            json.dump([list(item) for item in entity_list], f)
        print("Data saved to cache.")

    NUM_GPUS = 4
    world_size = NUM_GPUS
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(random.randint(4000, 8888))
    mp.spawn(main, nprocs=NUM_GPUS, args=(world_size, entity_data, ts_to_entity,NUM_GPUS))
