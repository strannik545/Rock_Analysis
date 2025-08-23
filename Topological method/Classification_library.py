import torch
import os
#num_threads = os.cpu_count()
#torch.set_num_threads(num_threads)
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split, KFold
# SE-блок
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd 
import torch.nn.functional as F
import copy
import math 



class ECABlock(nn.Module):
    """Efficient Channel Attention Block"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = max(t if t % 2 else t + 1, 3)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.conv(y)
        y = y.transpose(1, 2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class DropPath(nn.Module):
    """Stochastic Depth (DropPath)"""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size, drop_path_rate=0.0):
        super(MBConv, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Инициализация слоев
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            ECABlock(hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path_rate) if self.use_residual else nn.Identity()

    def forward(self, x):
        if self.use_residual:
            return x + self.drop_path(self.conv(x))
        return self.conv(x)

class CustomNet(nn.Module):
    def __init__(self, num_classes=2, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.3, drop_connect_rate=0.2):
        super(CustomNet, self).__init__()
        base_channels = 32
        
        # Stem
        stem_channels = int(base_channels * width_coefficient)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True)
        )
        
        # Конфигурация блоков
        block_config = [
            # in, out, exp, stride, kernel, repeats
            [stem_channels, 16, 1, 1, 3, 1],
            [16, 24, 6, 2, 3, 2],
            [24, 40, 6, 2, 5, 2],
            [40, 80, 6, 2, 3, 3],
            [80, 112, 6, 1, 5, 3],
            [112, 192, 6, 2, 5, 4],
            [192, 320, 6, 1, 3, 1]
        ]
        
        # Построение блоков
        total_blocks = sum([r for *_, r in block_config])
        current_block = 0
        layers = []
        
        for config in block_config:
            in_ch, out_ch, exp_ratio, stride, kernel, repeats = config
            out_ch = int(out_ch * width_coefficient)
            repeats = int(math.ceil(repeats * depth_coefficient))
            
            for i in range(repeats):
                block_stride = stride if i == 0 else 1
                block_in_ch = in_ch if i == 0 else out_ch
                
                # Динамическая вероятность DropPath
                drop_prob = drop_connect_rate * current_block / total_blocks
                
                layers.append(MBConv(
                    block_in_ch, out_ch, exp_ratio, block_stride, 
                    kernel_size=kernel, drop_path_rate=drop_prob
                ))
                current_block += 1
        
        self.blocks = nn.Sequential(*layers)
        
        # Head
        head_in_ch = int(320 * width_coefficient)
        head_hidden_ch = int(1280 * max(width_coefficient, 1.0))
        
        self.head = nn.Sequential(
            nn.Conv2d(head_in_ch, head_hidden_ch, 1, bias=False),
            nn.BatchNorm2d(head_hidden_ch),
            nn.SiLU(inplace=True)
        )
        
        # Классификатор
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_ch, head_hidden_ch // 2),
            nn.BatchNorm1d(head_hidden_ch // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(head_hidden_ch // 2, num_classes)
        )
        
        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



class KidneyStoneDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, strong_transform=None, minor_classes=None):
        self.df = df
        self.transform = transform
        self.strong_transform = strong_transform
        self.minor_classes = minor_classes  # Список миноритарных классов
        self.columns = self.df.columns
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx][self.columns[0]]
        label = self.df.iloc[idx]['label']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            if self.minor_classes and label in self.minor_classes and self.strong_transform:
                image = self.strong_transform(image)
            else:
                image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

    
class Learning_rocks():
    
    def __init__(self, df: pd.DataFrame, random_state = None, shuffle=True, n_splits=5,device = 'cuda', model_name = 'best_model.pth', BATCH_SIZE = 256, num_ep = 20, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2, num_workers = 4, train = 0.7, val = 0.5, minor_classes=None, focal_gamma=2.0, major_aug = True, aug = True):
        self.device = device
        self.shuffle=shuffle
        self.num_ep = num_ep
        self.df = df
        self.width_coefficient=width_coefficient
        self.depth_coefficient=depth_coefficient
        self.dropout_rate=dropout_rate
        self.model_name = model_name
        self.num_workers = num_workers
        self.columns = self.df.columns
        self.minor_classes = minor_classes
        self.random_state = random_state
        self.major_aug = major_aug
        self.n_splits=n_splits
        if aug:
            if major_aug:
                self.train_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.RandomAffine(degrees=0, shear=15),
                    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            else:
                self.train_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomHorizontalFlip(p=0.3),        # Уменьшена вероятность
                    transforms.RandomVerticalFlip(p=0.2),          # Значительно уменьшена вероятность
                    transforms.RandomRotation(10),                 # Уменьшен угол
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Слабое воздействие
                    transforms.RandomAffine(degrees=5, shear=5),   # Уменьшены параметры
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Меньше искажения + вероятность
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        
            
            self.strong_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(45),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomAffine(degrees=30, shear=30),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.7),
                transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
                transforms.GaussianBlur(kernel_size=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            minor_classes = None
            self.train_transform = transforms.Compose([
            transforms.Resize(256),           # Обязательное изменение размера
            transforms.ToTensor(),             # Преобразование в тензор
            transforms.Normalize(             # Нормализация
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
            ])
            self.minor_classes = None
            self.strong_transform = None
            
        self.val_test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.le = LabelEncoder()
        self.df['label'] = self.le.fit_transform(self.df[self.columns[1]])
        #self.df = self.df.drop([self.df.columns[1]])
        self.columns = self.df.columns
        

        # Обработка минорных классов (конвертация строк в числа если нужно)
        if minor_classes is not None:
            if isinstance(minor_classes[0], str):
                # Конвертируем строковые имена классов в числовые метки
                self.minor_classes = self.le.transform(minor_classes)
            else:
                self.minor_classes = minor_classes
        else:
            self.minor_classes = None
            
            
        # === ВСТАВЬТЕ ОТЛАДОЧНУЮ ПЕЧАТЬ ЗДЕСЬ ===
        print(f"Minor classes: {self.minor_classes}")
        if len(self.df) > 0:
            print(f"Sample label: {self.df.iloc[0]['label']}, type: {type(self.df.iloc[0]['label'])}")
        # =========================================
        
        # Разделение данных        

        self.train = train
        
        self.train_df, self.temp_df = train_test_split(self.df, test_size= (1-self.train), stratify=self.df['label'])
        self.val = val
        
        self.val_df, self.test_df = train_test_split(self.temp_df, test_size=(1 - self.val), stratify=self.temp_df['label'])


        # Создание датасетов
        self.train_dataset = KidneyStoneDataset(
            self.train_df, 
            transform=self.train_transform,
            strong_transform=self.strong_transform,
            minor_classes=self.minor_classes
        )
        self.val_dataset = KidneyStoneDataset(
            self.val_df, 
            transform=self.val_test_transform
        )
        self.test_dataset = KidneyStoneDataset(
            self.test_df, 
            transform=self.val_test_transform
        )
        
        
        # Взвешенный семплинг для балансировки классов
        class_counts = self.train_df['label'].value_counts().sort_index().values
        class_weights = 1. / class_counts
        sample_weights = class_weights[self.train_df['label']]
        sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        
        
        self.BATCH_SIZE = BATCH_SIZE
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, num_workers=self.num_workers, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, num_workers=self.num_workers, pin_memory=True)
        

        self.count_of_classes = len(self.le.classes_)#count_of_classes
        
        
        self.model = CustomNet(num_classes=self.count_of_classes, width_coefficient=self.width_coefficient, depth_coefficient=self.depth_coefficient, dropout_rate=self.dropout_rate)
        if self.device == 'cpu':
            self.model = self.model.cpu()
        elif self.device == 'cuda':
            self.model = self.model.cuda()


        
        
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.train_df['label']),
            y=self.train_df['label'].values
        )
        
        
        if device == 'cpu':
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32).cpu()
        elif device == 'cuda':
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32).cuda()

        self.criterion = FocalLoss(alpha=self.class_weights, 
            gamma=focal_gamma
        )            

        #инициализация компонентов
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        #self.scaler = torch.cuda.amp.GradScaler()
        if device == 'cpu':
            self.scaler = torch.amp.GradScaler('cpu')
        elif device == 'cuda':
            #self.scaler = torch.cuda.amp.GradScaler()
            self.scaler = torch.amp.GradScaler('cuda')

        self.best_recall = 0.0
        self.loss_arr_tr = []
        self.acc_arr_tr = []
        self.rec_arr_tr = []
        self.loss_arr_vl = []
        self.acc_arr_vl = []
        self.rec_arr_vl = []
        
    def idx_output(self):
        
        #for idx, class_name in enumerate(self.le.classes_):
        #    print(f"{idx}: {class_name}")
        return {idx: class_name for idx, class_name in enumerate(self.le.classes_)}
    
    def data_info(self):
        print(self.train_df.shape)
        print(self.val_df.shape)
        print(self.test_df.shape)
        
    def calculate_metrics(self, outputs, labels):
        _, preds = torch.max(outputs, 1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        
        # Расчет recall для каждого класса
        recalls = {}
        for cls in range(self.count_of_classes):
            cls_mask = (labels == cls)
            if cls_mask.any():
                recalls[cls] = ((preds[cls_mask] == cls).sum().item() / cls_mask.sum().item())
        
        return {
            'accuracy': correct / total,
            'recalls': recalls,
            'macro_recall': np.mean(list(recalls.values())) if recalls else 0
        }
        

    def clear_memory(self):
        del self.model
        del self.optimizer
        del self.scheduler
        del self.scaler
        del self.train_loader
        del self.val_loader
        import torch, gc
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()


    def learn_model(self):

        for epoch in tqdm(range(self.num_ep), desc="Epochs"):

            self.model.train()
            train_loss = 0.0
            for inputs, labels in self.train_loader:
                if self.device == 'cpu':
                    inputs, labels = inputs, labels
                elif self.device == 'cuda':
                    inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

                    
                self.optimizer.zero_grad()
                if self.device == 'cpu':
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)    
                elif self.device == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                train_loss += loss.item() * inputs.size(0)
            

            self.model.eval()
            val_loss = 0.0
            val_metrics = {'accuracy': 0.0, 'macro_recall': 0.0}
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    if self.device == 'cpu':
                        inputs, labels = inputs, labels
                    elif self.device == 'cuda':
                        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    metrics = self.calculate_metrics(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_metrics['accuracy'] += metrics['accuracy'] * inputs.size(0)
                    val_metrics['macro_recall'] += metrics['macro_recall'] * inputs.size(0)
            
            train_loss = train_loss / len(self.train_dataset)
            val_loss = val_loss / len(self.val_dataset)
            val_accuracy = val_metrics['accuracy'] / len(self.val_dataset)
            val_recall = val_metrics['macro_recall'] / len(self.val_dataset)
            
            self.scheduler.step()

            if val_recall > self.best_recall:
                self.best_recall = val_recall
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_recall': self.best_recall,
                }, f'{self.model_name}')
            
            self.loss_arr_tr.append(train_loss)
            self.loss_arr_vl.append(val_loss)
            self.acc_arr_vl.append(val_accuracy)
            self.rec_arr_vl.append(val_recall)
    
    
    def output_met(self):
        epochs = range(1, len(self.loss_arr_tr) + 1)

        #график потерь (Loss)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.loss_arr_tr, label='Train Loss', color='blue')
        plt.plot(epochs, self.loss_arr_vl, label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        #график точности (Accuracy)
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.acc_arr_vl, label='Validation Accuracy', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()

        #график полноты (Recall)
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.rec_arr_vl, label='Validation Recall', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.title('Validation Recall')
        plt.legend()

        #показать графики
        plt.tight_layout()
        plt.show()
    
    def class_balance(self):
        label_counts = self.df['label'].value_counts()
        label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Распределение меток')
        plt.xlabel('Метки')
        plt.ylabel('Количество')
        plt.xticks(rotation=0)
        plt.show()


    def mean_kfold_validation(self, n_splits):
        """
        Выполняет K-Fold кросс-валидацию
        :param n_splits: Количество фолдов
        :param random_state: Seed для воспроизводимости
        """
        # Инициализация KFold
        kf = KFold(n_splits=n_splits, shuffle= self.shuffle, random_state=self.random_state)
        
        # Подготовка данных для кросс-валидации (объединяем train+val)
        full_df = pd.concat([self.train_df, self.val_df])
        
        # Метрики по фолдам
        fold_metrics = {
            'val_loss': [],
            'val_accuracy': [],
            'val_recall': []
        }
        
        best_models = []  # Сохраним лучшие модели каждого фолда
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(full_df)):
            print(f"\n{'='*30}")
            print(f"Training Fold {fold+1}/{n_splits}")
            print(f"{'='*30}")
            
            # Создаем копии датафреймов для фолда
            train_fold = full_df.iloc[train_idx].copy()
            val_fold = full_df.iloc[val_idx].copy()
            
            # Создаем временный экземпляр класса с данными фолда
            fold_learner = copy.deepcopy(self)
            fold_learner.train_df = train_fold
            fold_learner.val_df = val_fold
            fold_learner.test_df = None  # Отключаем тестовый набор
            
            # Пересчитываем веса классов для нового распределения
            fold_learner.class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(fold_learner.train_df['label']),
                y=fold_learner.train_df['label'].values
            )
            fold_learner.class_weights = torch.tensor(
                fold_learner.class_weights, 
                dtype=torch.float32
            ).to(self.device)
            
            # Инициализируем модель и оптимизатор заново
            fold_learner.model = CustomNet(
                num_classes=fold_learner.count_of_classes,
                width_coefficient=self.width_coefficient,
                depth_coefficient=self.depth_coefficient,
                dropout_rate=self.dropout_rate
            ).to(self.device)
            
            fold_learner.optimizer = AdamW(
                fold_learner.model.parameters(), 
                lr=2e-4, 
                weight_decay=1e-5
            )
            
            # Обучаем на фолде
            fold_learner.learn_model()
            
            # Сохраняем метрики
            fold_metrics['val_loss'].append(fold_learner.loss_arr_vl[-1])
            fold_metrics['val_accuracy'].append(fold_learner.acc_arr_vl[-1])
            fold_metrics['val_recall'].append(fold_learner.rec_arr_vl[-1])
            
            # Сохраняем лучшую модель фолда
            best_models.append(copy.deepcopy(fold_learner.model.state_dict()))
            
            # Очистка памяти
            del fold_learner
            torch.cuda.empty_cache()
        
        # Вывод результатов кросс-валидации
        print("\nK-Fold Validation Results:")
        print(f"{'-'*40}")
        for metric, values in fold_metrics.items():
            print(f"Mean {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        return best_models, fold_metrics

    def kfold_validation_best(self, n_splits=5, random_state=42, save_best_path="best_model.pth"):
        # Инициализация KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        full_df = pd.concat([self.train_df, self.val_df])
        
        best_recall = 0.0
        best_model_state_dict = None
        best_fold = 0
        
        fold_metrics = {
            'val_loss': [],
            'val_accuracy': [],
            'val_recall': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(full_df)):
            print(f"\n{'='*30}")
            print(f"Training Fold {fold+1}/{n_splits}")
            print(f"{'='*30}")
            
            # Создаем копии датафреймов для фолда
            train_fold = full_df.iloc[train_idx].copy()
            val_fold = full_df.iloc[val_idx].copy()
            
            # Создаем временный экземпляр класса
            fold_learner = copy.deepcopy(self)
            fold_learner.train_df = train_fold
            fold_learner.val_df = val_fold
            fold_learner.model_name = f'temp_fold_{fold}.pth'
            
            # Пересчитываем веса классов
            fold_learner.class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(fold_learner.train_df['label']),
                y=fold_learner.train_df['label'].values
            )
            fold_learner.class_weights = torch.tensor(
                fold_learner.class_weights, 
                dtype=torch.float32
            ).to(self.device)
            
            # Инициализируем новую модель
            fold_learner.model = CustomNet(
                num_classes=fold_learner.count_of_classes,
                width_coefficient=self.width_coefficient,
                depth_coefficient=self.depth_coefficient,
                dropout_rate=self.dropout_rate
            ).to(self.device)
            
            fold_learner.optimizer = AdamW(
                fold_learner.model.parameters(), 
                lr=2e-4, 
                weight_decay=1e-5
            )
            
            # Обучаем на фолде
            fold_learner.learn_model()
            
            # Проверяем, является ли текущая модель лучшей
            current_recall = fold_learner.rec_arr_vl[-1]
            fold_metrics['val_loss'].append(fold_learner.loss_arr_vl[-1])
            fold_metrics['val_accuracy'].append(fold_learner.acc_arr_vl[-1])
            fold_metrics['val_recall'].append(current_recall)
            
            if current_recall > best_recall:
                best_recall = current_recall
                best_fold = fold
                
                # Исправление: загрузка с weights_only=False
                checkpoint = torch.load(fold_learner.model_name, weights_only=False)
                best_model_state_dict = checkpoint['model_state_dict']
                
                # Сохраняем лучшую модель
                torch.save({
                    'model_state_dict': best_model_state_dict,
                    'fold': fold,
                    'recall': best_recall
                }, save_best_path)
                print(f"New best model found in fold {fold+1} with recall: {best_recall:.4f}")
            
            # Удаляем временный файл модели
            if os.path.exists(fold_learner.model_name):
                os.remove(fold_learner.model_name)
            
            # Очистка памяти
            del fold_learner
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Вывод результатов
        print("\nK-Fold Validation Results:")
        print(f"{'-'*40}")
        print(f"Best model from fold {best_fold+1} with recall: {best_recall:.4f}")
        for metric, values in fold_metrics.items():
            print(f"Mean {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        return best_model_state_dict, fold_metrics