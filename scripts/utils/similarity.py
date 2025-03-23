import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch import nn
import torch.nn.functional as F
class CosineSimilarity(nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()
         # Load pre-trained VGG19 model without the classification head
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg19 = models.vgg19(pretrained=True).to(self.device)
        self.model = nn.Sequential(*list(self.vgg19.children())[:-1])  # Remove classification head
        self.model.eval()
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def load_and_preprocess_image(self,img_path, transform):
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0)
        return img
    def extract_features(self, img_path):
        img_tensor = self.load_and_preprocess_image(img_path, self.transform).to(self.device)
        with torch.no_grad():
            features = self.model(img_tensor)
        return features.cpu().numpy().flatten()
    def calculate_similarity(self,image1_path, image2_path):
        feature1 = self.extract_features(image1_path)
        feature2 = self.extract_features(image2_path)
        similarity = cosine_similarity([feature1], [feature2])
        return similarity[0][0]