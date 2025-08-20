from Classification_library import *
from PIL import Image
import torch

class Predict():
    def __init__(self, num_classes=2, device='auto', model_path='best_model.pth'):
        """
        Инициализация класса для предсказаний
        
        :param num_classes: количество классов
        :param device: 'auto', 'cuda' или 'cpu'
        :param model_path: путь к файлу модели
        """
        # Автоматическое определение устройства
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device.lower() in ['gpu', 'cuda']:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.num_classes = num_classes
        self.model = CustomNet(num_classes=self.num_classes)
        self.model_path = model_path
        
        # Загрузка модели с явным указанием weights_only=False
        try:
            checkpoint = torch.load(
                self.model_path, 
                map_location=self.device,
                weights_only=False  # Ключевое исправление!
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Универсальная загрузка модели
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_single_patch(self, img):
        """
        Предсказание для одного изображения
        
        :param img: путь к изображению или объект PIL.Image
        :return: словарь с результатами предсказания
        """
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, Image.Image):
            img = img.convert('RGB')
        else:
            raise ValueError("Unsupported image type. Provide path or PIL.Image")
        
        input_tensor = self.test_transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
        
        return {
            'predicted_class': pred_class.item(),
            'confidence': confidence.item(),
            'probabilities': probs.cpu().numpy()[0]
        }