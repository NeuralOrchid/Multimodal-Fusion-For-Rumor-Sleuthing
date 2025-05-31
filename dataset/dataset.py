import gdown
from tqdm.notebook import tqdm_notebook as tqdm
import os.path as osp
import zipfile
from typing import Literal
import pandas as pd
import tiktoken
import torch
from torch_geometric.data import Dataset, Data


class TwitterDataset(Dataset):
    def __init__(
            self,
            root='data',
            dataset_name:Literal['twitter15', 'twitter16'] = 'twitter15',
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self.dataset_name = dataset_name
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        base_dir = f"rumor_detection_acl2017/{self.dataset_name}/"
        return [
            base_dir + "tree",
            base_dir + "label.txt",
            base_dir + "source_tweets.txt",
        ]

    @property
    def processed_file_names(self):
        num = 818 if self.dataset_name == 'twitter16' else 1490
        return [f'data_{i}.pt' for i in range(num)]

    def download(self):
        gdown.download(
            id="1oJ1k0ow1398PAHzq2j636ySsbX4jxDpW",
            output=self.raw_dir + "/rumdetect2017.zip",
        )

        with zipfile.ZipFile(f"{self.raw_dir}/rumdetect2017.zip", 'r') as zip_ref:
            zip_ref.extractall(f"{self.raw_dir}")


    def process_graph(self, path):
        graph = []
        features = []
        uids = set()
        root_uid = None
        with open(path) as file:
            for line in file.readlines():
                parent_part, child_part = line.split("->")
                parent_data = [item.strip(" \'\"") for item in parent_part.strip('[\n] ').split(",")]
                parent_uid = parent_data[0]
                child_data = [item.strip(" \'\"") for item in child_part.strip('[\n] ').split(",")]
                child_uid = child_data[0]
                child_delay = child_data[2]

                if parent_uid == 'ROOT':
                    root_uid = child_uid
                else:
                    graph.append([parent_uid, child_uid])
                features.append([child_uid, child_delay])
                uids.add(child_uid)
        uids = sorted(uids)
        for edge in graph:
            edge[0] = uids.index(edge[0])
            edge[1] = uids.index(edge[1])
        for data in features:
            data[0] = uids.index(data[0])
            data[1] = float(data[1])
        root_index = uids.index(root_uid)

        x = torch.zeros((len(features), 1))
        for child_uid, child_delay in features:
            x[child_uid][0] = child_delay

        TD_edge_index = torch.tensor(graph)

        return x, TD_edge_index, root_index


    def process(self):
        enc = tiktoken.get_encoding("cl100k_base")
        CLASSES = ["unverified", "non-rumor", "true", "false",]
        # Reading CSV files
        TREE_DIR, LABELS_PATH, SOURCE_PATH = self.raw_paths
        label_df = pd.read_csv(LABELS_PATH, sep=":", names=['label', 'eid'], header=None)
        source_df = pd.read_csv(SOURCE_PATH, sep="\t", names=['eid', 'source'], header=None)

        EIDs = label_df.eid.tolist()

        for idx, eid in tqdm(enumerate(EIDs)):
            source = source_df.loc[source_df['eid'] == eid]['source'].values[0]
            tokens = torch.zeros((106,))
            for i, token in enumerate(enc.encode(source)):
                tokens[i] = token

            y = torch.tensor(CLASSES.index(label_df.loc[label_df['eid'] == eid]['label'].values[0]))
            x, edge_index, root_index = self.process_graph(TREE_DIR + f"/{eid}.txt")
            data = Data(
                x = x,
                edge_index = torch.transpose(edge_index, 0, 1).to(torch.int64),
                root_index = root_index,
                y = y,
                tokens = tokens.unsqueeze(0),
            )

            # Saving
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data