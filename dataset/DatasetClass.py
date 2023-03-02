import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils
from PIL import Image
import pickle
import os 



class DatasetClass(Dataset):
    def __init__(
        self,
        cur_dir, 
        folder,
        image_size,
        augment_horizontal_flip = False,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
    ):

        super().__init__()

        self.folder = folder
        self.image_size = image_size
        self.base_dir = os.path.join(cur_dir, self.folder)

        self.paths = []
        _img_class_folders = os.listdir(self.base_dir)
        for _img_class_folder in _img_class_folders:
            for _img in os.listdir(os.path.join(self.base_dir, _img_class_folder)):
                if _img.split(".")[-1] in exts:
                    self.paths.append(os.path.join(self.base_dir, _img_class_folder, _img))

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])


    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

    @property
    def get_img_height(self):
        return self.image_size


    @property
    def get_img_width(self):
        return self.image_size