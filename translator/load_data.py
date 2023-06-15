from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
class TransDataSet(Dataset):
    def __init__(self,data_path) -> None:
        super().__init__()
        with open(data_path) as jf:
            self.data = json.load(jf)
        self.prerocess()
    def prerocess(self):
        self.combine_data = []
        for item in tqdm(self.data,desc="正在处理数据"):
            id = item["id"]
            temp_file = "temp/"+id+'.json'
            if os.path.exists(temp_file):
                continue
            for round,convers in enumerate(item["conversations"]):
                
                for conv_sen in convers["value"].split("\n"):
                    conv_sen = conv_sen.replace("\n<image>","").replace("<image>\n","")
                    self.combine_data.append({"id":id,"round":round,"from":convers["from"],"value":conv_sen})               
    def __len__(self):
        return len(self.combine_data)
    def __getitem__(self, idx):
        return self.combine_data[idx]
        
def self_collfn(batch):
    ids = []
    rounds = []
    froms = []
    values = []
    for item in batch:
        ids.append(item["id"])
        rounds.append(item["round"])
        froms.append(item["from"])
        values.append(item["value"])
    
    return {"ids":ids,"rounds":rounds,"froms":froms,"values":values}

def build_dataloader(dataset,batch_size=32):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=self_collfn
    )
    return dataloader
