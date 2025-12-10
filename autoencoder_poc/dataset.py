import os
import glob
import hydra
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

class SimpleImageDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # まずはRGBで読み込む
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0 # 学習用なのでラベルはダミー

def get_transforms(cfg):
    input_size = tuple(cfg.dataset.input_size)
    
    transform_list = []
    
    # グレースケール変換 (PILレベルまたはTransformレベル)
    # ここではTransformで対応
    if cfg.dataset.channels == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
        
    transform_list.extend([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
    
    return transforms.Compose(transform_list)

def get_dataloaders(cfg):
    transform = get_transforms(cfg)
    
    # --- Train Data Loading ---
    # Hydraを使用している場合、パスを絶対パスに変換（出力ディレクトリに移動している可能性があるため）
    root_dir = hydra.utils.to_absolute_path(cfg.dataset.root_dir)
    train_dir = os.path.join(root_dir, cfg.dataset.train_dir)
    
    # 対応する拡張子
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    train_files = []
    for ext in exts:
        # 大文字小文字両方対応のため
        train_files.extend(glob.glob(os.path.join(train_dir, ext)))
        train_files.extend(glob.glob(os.path.join(train_dir, ext.upper())))
    
    # 重複削除 (念のため)
    train_files = sorted(list(set(train_files)))
    
    if not train_files:
        print(f"Warning: No images found in {train_dir}")
        # エラーにせず、空のリストで進める（テストのみ実行したい場合などを考慮）
        # ただし今回はPoCなのでエラーを出したほうが親切かもしれないが、一旦printのみ。
    
    full_train_dataset = SimpleImageDataset(train_files, transform=transform)
    
    # Validation Split
    if len(full_train_dataset) > 0:
        val_size = int(len(full_train_dataset) * cfg.dataset.validation_ratio)
        train_size = len(full_train_dataset) - val_size
        
        # データが少なすぎてval_sizeが0になるのを防ぐ
        if val_size == 0 and len(full_train_dataset) > 1:
            val_size = 1
            train_size = len(full_train_dataset) - 1
            
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.dataset.batch_size, 
            shuffle=True, 
            num_workers=cfg.dataset.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.dataset.batch_size, 
            shuffle=False, 
            num_workers=cfg.dataset.num_workers
        )
    else:
        train_loader = None
        val_loader = None

    # --- Test Data Loading ---
    test_dir = os.path.join(root_dir, cfg.dataset.test_dir)
    
    if os.path.exists(test_dir):
        # ImageFolderを使うとサブディレクトリ名がクラス名になる
        test_dataset = ImageFolder(test_dir, transform=transform)
        class_names = test_dataset.classes
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            num_workers=cfg.dataset.num_workers
        )
    else:
        print(f"Warning: Test directory {test_dir} not found.")
        test_loader = None
        class_names = []

    return train_loader, val_loader, test_loader, class_names
