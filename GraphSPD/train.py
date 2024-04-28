import os
import sys
import time
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from libs.nets.PGCN_noAST import PGCN, PGCNTrain


logsPath = './logs/'
mdlsPath = './models/'

# parameters
_CLANG_  = 1
_NETXARCHT_ = 'PGCN'
_BATCHSIZE_ = 128
dim_features = 20
start_time = time.time() #mark start time

class Logger(object):
    def __init__(self, filename = "log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def RunTime():
    pTime = ' [TIME: ' + str(round((time.time() - start_time), 2)) + ' sec]'
    return pTime

# get dataset
def GetDataset(path=None):
    '''
    Get the dataset from numpy data files.
    :param path: the path used to store numpy dataset.
    :return: dataset - list of torch_geometric.data.Data
    '''

    # check.
    if None == path:
        print('[Error] <GetDataset> The method is missing an argument \'path\'!')
        return [], []

    # contruct the dataset.
    dataset = []
    files = []
    for root, _, filelist in os.walk(path):
        for file in filelist:
            if file == 'representation3.npz':
                # read a numpy graph file.
                graph = np.load(os.path.join(root, file), allow_pickle=True)
                files.append(os.path.join(root, file[:-7]))
                # sparse each element.
                edgeIndex = torch.tensor(graph['edgeIndex'], dtype=torch.long)
                nodeAttr = torch.tensor(graph['nodeAttr'], dtype=torch.float)
                edgeAttr = torch.tensor(graph['edgeAttr'], dtype=torch.float)
                label = torch.tensor(graph['label'], dtype=torch.long)
                # construct an instance of torch_geometric.data.Data.
                data = Data(edge_index=edgeIndex, x=nodeAttr, edge_attr=edgeAttr, y=label)
                # append the Data instance to dataset.
                dataset.append(data)

    if (0 == len(dataset)):
        print(f'[ERROR] Fail to load data from {path}')

    return dataset, files

# main
def main(train_path):
    model = PGCN(num_node_features=dim_features)

    dataset, files = GetDataset(path=train_path)
    dataloader = DataLoader(dataset, batch_size=_BATCHSIZE_, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    epochs = 200
    for epoch in range(epochs):
        model_trained, train_loss, A, P, R = PGCNTrain(model, dataloader, optimizer, criterion)
        print(train_loss)
        print(A, P, R)
    torch.save(model_trained.state_dict(), "models/graphspd_0421")

    return

if __name__ == '__main__':
    logfile = 'test.txt'
    train_path = "/Users/min/Code/SPI/MrSPI/data/1000samples_patchdb/train"
    main(train_path)



