##
# This code is modified based on https://github.com/SJTUwxz/LoCoNet_ASD.git
##

import os, torch, warnings, glob, json
from utils.tools import *
from dataLoader_multiperson import val_loader
from d2stream import d2stream
import csv

dataPath = "../../data/AVADataPath/"
ori_file = "../../data/AVADataPath/csv/val_orig.csv"
trianFileName = "../../data/AVADataPath/csv/train_loader.csv"
valFileName = "../../data/AVADataPath/csv/val_loader.csv"
video_root = "../../data/AVADataPath/clips_videos/val"
audio_root = "../../data/AVADataPath/clips_audios/val"


class DataPrep():

    def __init__(self,world_size, rank,entity_data,ts_to_entity):
        self.world_size = world_size
        self.rank = rank
        self.entity_data = entity_data
        self.ts_to_entity = ts_to_entity

    def val_dataloader(self):
        loader = val_loader(trialFileName="../../data/AVADataPath/csv/val_loader.csv", \
                            audioPath="../../data/AVADataPath/clips_audios/val", \
                            visualPath="../../data/AVADataPath/clips_videos/val", \
                            entity_data=self.entity_data, \
                            ts_to_entity=self.ts_to_entity, \
                            num_speakers=3,
                            )
        valLoader = torch.utils.data.DataLoader(loader,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=16)
        return valLoader


def main(gpu,world_size,entity_data,ts_to_entity):
    rank = gpu
    warnings.filterwarnings("ignore")
    data = DataPrep(world_size, rank,  entity_data,ts_to_entity)

    s = d2stream()
    model_dir = "./save/exps/exp_08_13"
    mo = glob.glob(os.path.join(model_dir, 'model_*.model'))
    mo.sort()
    modelfiles = mo[20:]
    print("modelfiles:", modelfiles)
    if len(modelfiles) == 0:
        print(f"No model files found in directory {model_dir}.")
        quit()
    print(f"Found {len(modelfiles)} model files in directory {model_dir}. Starting evaluation...")
    for model_path in modelfiles:
        filename = os.path.basename(model_path)
        epoch_str = os.path.splitext(filename)[0].split('_')[-1]
        print(epoch_str)
        epoch = int(epoch_str)
        s.loadParameters(model_path)
        mAP = s.evaluate_network(epoch=epoch, loader=data.val_dataloader())
        print(f"Evaluation result for model {filename} (Epoch {epoch}): mAP = {mAP}")
    print("\nAll model evaluations completed.")


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
        print('Initializing entity list, total entities:', len(entity_set))
        for video_id, entity_id in entity_set.copy():
            exist_entity = os.path.join(video_root, video_id, entity_id)
            if not os.path.exists(exist_entity):
                entity_set.remove((video_id, entity_id))
        print('After filtering out non-downloaded entities, total entities:', len(entity_set))
        for video_id, entity_id in entity_set.copy():
            dir = os.path.join(video_root, video_id, entity_id)
            if len(os.listdir(dir)) != len(entity_data[video_id][entity_id]):
                entity_set.remove((video_id, entity_id))
        print('After filtering out incomplete entities, total entities:', len(entity_set))
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

    cache_dir = os.path.join(dataPath, "json", "val")
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

    gpu = p = 0
    world_size = 6

    main(gpu, world_size, entity_data, ts_to_entity,)
