from Classification_library import *
from PIL import Image
class Predict():
    def __init__(self, num_classes=2, device = 'cuda', model_path = 'best_model.pth'):
        self.num_classes = num_classes
        self.model = CustomNet(num_classes=self.num_classes)
        self.model_path = model_path
        self.checkpoint = torch.load(self.model_path, weights_only=False)

        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.device = device
        if self.device == 'cuda':
            self.model = selfmodel.cuda()
        elif self.device == 'cpu':
            self.model = self.model.cpu()

        #self.img = Image.open(img)


        self.test_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_single_patch(self, img):
        transform = self.test_transform
        if isinstance(img, str):
            path = Image.open(img).convert('RGB')
        elif isinstance(img, Image.Image):
            path = img.convert('RGB')
            

        input_tensor = transform(path).unsqueeze(0)
        
        input_tensor = input_tensor.to('cpu')

        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
        
        return {
            'predicted_class': pred_class.item(),
            'confidence': confidence.item(),
            'probabilities': probs.cpu().numpy()[0]
        }

        

        
