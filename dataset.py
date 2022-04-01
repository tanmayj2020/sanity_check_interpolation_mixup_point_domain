import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random
random.seed(9001)
import torchvision.transforms.functional as F
from my_rasterize import rasterize_Sketch
import numpy as np

class Dataset_TUBerlin(data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode
        coordinate_path = os.path.join(hp.base_dir, 'TU-Berlin/TU_Berlin')

        with open(coordinate_path, 'rb') as fp:
            self.Coordinate = pickle.load(fp)

        total_del = 0
        all_keys = list(self.Coordinate.keys())
        for key in all_keys:
            if len(self.Coordinate[key]) > 300:
                del self.Coordinate[key]
                total_del += 1

        print('Total Number of samples deleted: {}'.format(total_del))

        get_all_classes, all_samples = [], []
        for x in list(self.Coordinate.keys()):
            get_all_classes.append(x.split('/')[0])
            all_samples.append(x)
        get_all_classes = list(set(get_all_classes))
        get_all_classes.sort()
       



        self.num2name, self.name2num = {}, {}
        for num, val in enumerate(get_all_classes):
            self.num2name[num] = val
            self.name2num[val] = num

        self.Train_Sketch, self.Test_Sketch = [], []
        for class_name in get_all_classes:
            per_class_data = np.array([x for x in all_samples if class_name == x.split('/')[0]])
            per_class_Train = per_class_data[random.sample(range(len(per_class_data)), int(len(per_class_data) * hp.splitTrain))]
            per_class_Test = set(per_class_data) - set(per_class_Train)
            self.Train_Sketch.extend(list(per_class_Train))
            self.Test_Sketch.extend(list(per_class_Test))

        print('Total Training Sample {}'.format(len(self.Train_Sketch)))
        print('Total Testing Sample {}'.format(len(self.Test_Sketch)))


        self.train_transform = get_ransform('Train', hp)
        self.test_transform = get_ransform('Test', hp)


    def __getitem__(self, item):

        if self.mode == 'Train':
            sketch_path = self.Train_Sketch[item]

            vector_x = self.Coordinate[sketch_path]
            #sketch_img = rasterize_Sketch(vector_x, self.hp.channels)
            # sketch_img = torch.from_numpy(sketch_img)
            sketch_img = rasterize_Sketch(vector_x)
            sketch_img = Image.fromarray(sketch_img).convert('RGB')

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)


            sketch_img = self.train_transform(sketch_img)
            # sketch_img = sketch_img.float()


            sample = {'sketch_img': sketch_img,
                       'sketch_label': self.name2num[sketch_path.split('/')[0]]}


        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]
            vector_x = self.Coordinate[sketch_path]

            sketch_img = rasterize_Sketch(vector_x)
            # sketch_img = torch.from_numpy(sketch_img)
            sketch_img = Image.fromarray(sketch_img).convert('RGB')
            sketch_img = self.test_transform(sketch_img)
            # sketch_img = sketch_img.float()

            sample = {'sketch_img': sketch_img,
                     'sketch_label': self.name2num[sketch_path.split('/')[0]]}

        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            return len(self.Test_Sketch)


def collate_self(batch):
    batch_mod = {'sketch_img': [],
                 'sketch_label': [],
                 }

    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_label'].append(i_batch['sketch_label'])

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['sketch_label'] = torch.tensor(batch_mod['sketch_label'])

    return batch_mod


def get_dataloader(hp):

    if hp.dataset_name == 'TUBerlin':

        dataset_Train  = Dataset_TUBerlin(hp, mode = 'Train')
        dataset_Test = Dataset_TUBerlin(hp, mode='Test')

        dataset_Train.Test_Sketch = []
        dataset_Test.Train_Sketch = []

    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
                                         num_workers=int(hp.nThreads), collate_fn=collate_self)

    dataloader_Test = data.DataLoader(dataset_Test, batch_size=hp.batchsize, shuffle=False,
                                         num_workers=int(hp.nThreads), collate_fn=collate_self)

    return dataloader_Train, dataloader_Test



def get_ransform(type, hp):
    transform_list = []
    if type == 'Train':
        transform_list.extend([transforms.Resize(256)])
    elif type == 'Test':
        transform_list.extend([transforms.Resize(256)])
    # transform_list.extend(
    #     [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    mean1 = [0.5]*hp.channels
    std1 = [0.5]*hp.channels
    transform_list.extend(
        [
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean1, std=std1)])
    return transforms.Compose(transform_list)
