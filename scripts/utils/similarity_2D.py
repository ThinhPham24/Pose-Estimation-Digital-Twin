import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
def calculate_vgg19_pytorch_similarity(target_image_path, image_folder):
    """Calculates cosine similarity of a target image with images in a folder using VGG19 features (PyTorch)."""
    # Load pre-trained VGG19 model
    vgg19 = models.vgg19(pretrained=True).features  # Only feature extraction layers
    vgg19.eval()  # Set to evaluation mode
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_features(img_input):
        """Extracts VGG19 features from an image."""
        # img = Image.open(img_path).convert('RGB') #added convert RGB for png with alpha channel.
        if isinstance(img_input, np.ndarray):  # Input is a NumPy array
            try:
                img = Image.fromarray(img_input) #convert numpy array to PIL image.
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            except ValueError as e:
                print(f"Error: Could not convert NumPy array to PIL Image. {e}")
                return None
        elif isinstance(img_input, str):  # Input is a file path
            if not os.path.exists(img_input):
                print(f"Error: Image path '{img_input}' does not exist.")
                return None

            try:
                img = Image.open(img_input).convert('RGB')
            except (OSError, IOError, ValueError, SyntaxError) as e:
                print(f"Error: Could not open image '{img_input}'. {e}")
                return None
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():  # Disable gradient calculation
            features = vgg19(img_tensor)

        features = torch.flatten(features, start_dim=1).numpy() #flatten the features.
        return features

    target_features = extract_features(target_image_path)

    similarity_scores = {}
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            try:
                folder_image_features = extract_features(image_path)
                similarity = cosine_similarity(target_features, folder_image_features)[0][0]
                similarity_scores[filename] = similarity
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return similarity_scores

def load_dinov2_model(device='cuda'):
    # Load the DINOv2 ViT-L/14 model from PyTorch Hub
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.to(device)
    model.load_state_dict(torch.load('/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/checkpoints/dinov2_vitl14_pretrain.pth'))
    model.eval()
    return model

def preprocess_image(img_input, dinov2_size=(224, 224)):
    # Load and preprocess the image
    # img = Image.open(img_path).convert('RGB') #added convert RGB for png with alpha channel.
    if isinstance(img_input, np.ndarray):  # Input is a NumPy array
        try:
            img = Image.fromarray(img_input) #convert numpy array to PIL image.
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except ValueError as e:
            print(f"Error: Could not convert NumPy array to PIL Image. {e}")
            return None
    elif isinstance(img_input, str):  # Input is a file path
        if not os.path.exists(img_input):
            print(f"Error: Image path '{img_input}' does not exist.")
            return None

        try:
            img = Image.open(img_input).convert('RGB')
        except (OSError, IOError, ValueError, SyntaxError) as e:
            print(f"Error: Could not open image '{img_input}'. {e}")
            return None
    transform = T.Compose([
        T.Resize(dinov2_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(dinov2_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

def extract_features(model, img_tensor, device='cuda'):
    # Extract features using the DINOv2 model
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        features = model(img_tensor)
    return features.cpu().squeeze(0)  # Remove batch dimension

def calculate_cosine_similarity(target_feature, features_list):
    # Calculate cosine similarity between the target feature and a list of features
    similarities = []
    for feature in features_list:
        similarity = torch.nn.functional.cosine_similarity(target_feature, feature, dim=0)
        similarities.append(similarity.item())
    return similarities

def calculate_dinov2_pytorch_similarity(target_image_path, image_folder, device='cuda'):
    # Load the model
    model = load_dinov2_model(device)

    # Preprocess and extract features for the target image
    target_img_tensor = preprocess_image(target_image_path)
    target_feature = extract_features(model, target_img_tensor, device)
    similarity_scores = {}
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            try:
                img_tensor = preprocess_image(image_path)
                folder_image_features = extract_features(model, img_tensor, device)
                similarity = cosine_similarity(
                        target_feature.numpy().reshape(1, -1),
                        folder_image_features.numpy().reshape(1, -1)
                    )[0][0]
                similarity_scores[filename] = similarity
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    # print("Similarity scores:", similarity_scores)
    return similarity_scores

# # Example usage:
# target_image = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/images/top_crop.png' #replace with your target image path
# folder_path = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/partial_point_clouds/' #replace with your folder path

# similarity_results = calculate_vgg19_pytorch_similarity(target_image, folder_path)

# # Print similarity scores
# for filename, score in similarity_results.items():
#     print(f"Similarity with {filename}: {score}")

# # Find the most similar image
# if similarity_results:
#     most_similar_image = max(similarity_results, key=similarity_results.get)
#     print(f"\nMost similar image: {most_similar_image} (Similarity: {similarity_results[most_similar_image]})")
# else:
#     print("No images found or errors occurred.")