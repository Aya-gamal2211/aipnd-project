import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    image = preprocess(image).unsqueeze(0)
    return image

def predict(image_path, model, topk=5, category_names=None, gpu=False):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image = process_image(image_path).to(device)
    
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)

    top_p = top_p.cpu().numpy()[0]
    top_class = top_class.cpu().numpy()[0]
    
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_class = [cat_to_name[str(i)] for i in top_class]

    return top_p, top_class

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K classes')
    parser.add_argument('--category_names', type=str, help='Path to category to name JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    top_p, top_class = predict(args.image, model, args.top_k, args.category_names, args.gpu)

    print(f"Top {args.top_k} Classes: {top_class}")
    print(f"Probabilities: {top_p}")
