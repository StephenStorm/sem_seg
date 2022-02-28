import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class TIANCHI(Dataset):
    CLASSES = [
        'origin', 'modified'
    ]

    PALETTE = torch.tensor([
        [0, 0, 0], [255, 255, 255], 
    ])

    def __init__(self, root: str = 'dataset', split: str = 'train', transform = None) -> None:
        super().__init__()
        # stephen add 
        self.thresh = 127
        assert split in ['train', 'val']
        # split = 'training' if split == 'train' else 'validation'
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = -1

        img_path = Path(root) / split / 'img' 
        self.files = list(img_path.glob('*.jpg'))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('img', 'mask').replace('.jpg', '.png')

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)
        # stephen add :
        left_idx = label < self.thresh
        right_idx = label >= self.thresh
        label[left_idx] = 0.0
        label[right_idx] = 1.0

        
        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long() - 1


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(TIANCHI, '/opt/tiger/minist/datasets/tianchi')