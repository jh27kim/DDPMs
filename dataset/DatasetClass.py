import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils
from PIL import Image


class DatasetClass(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        augment_horizontal_flip = False,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
    ):

        super().__init__()

        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

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
