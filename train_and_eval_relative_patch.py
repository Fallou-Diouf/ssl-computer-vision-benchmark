import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import pandas as pd

# 1. Adaptations Mac / MPS
_original_dataloader = data.DataLoader
def _custom_dataloader(*args, **kwargs):
    kwargs['num_workers'] = 0
    kwargs['pin_memory'] = False
    return _original_dataloader(*args, **kwargs)
data.DataLoader = _custom_dataloader

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

from deepul_helper.tasks.relative_patch_pred import RelativePatchPrediction
from deepul_helper.utils import accuracy

def train_ssl_task(model, train_loader, epochs=5):
    """ Entraîne le modèle sur la tâche prétexte (Position Relative) """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    print(f"--- Début de l'entraînement SSL (Relative Patch) sur {epochs} époques ---")
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            
            metrics, _ = model(images)
            loss = metrics['Loss']
            acc = metrics['Acc1']
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_acc += acc.item()
            
        print(f"Époque [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} | Précision Tâche Prétexte: {running_acc/len(train_loader):.2f}%")

def train_linear_classifier(model, classifier, train_loader, epochs=5):
    """ Entraîne la tête linéaire en gelant le backbone SSL """
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    classifier.train()
    print(f"\n--- Entraînement du Classifieur Linéaire sur {epochs} époques ---")
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                features = model.encode(images)
            
            outputs = classifier(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Époque [{epoch+1}/{epochs}] - Loss Linear: {running_loss/len(train_loader):.4f}")

def evaluate_accuracy(model, classifier, test_loader):
    """ Évalue la précision Top-1 du classifieur """
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            features = model.encode(images)
            outputs = classifier(features)
            acc1, _ = accuracy(outputs, targets, topk=(1, 5))
            correct += acc1.item() * targets.size(0)
            total += targets.size(0)
            
    return round(correct / total, 2)

def main():
    print(f"Utilisation du périphérique : {device}\n")
    
    # Prétraitement CIFAR-10
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    # Datasets CIFAR-10
    cifar_train = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_cifar)
    cifar_test = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_cifar)
    
    train_loader = data.DataLoader(cifar_train, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(cifar_test, batch_size=128, shuffle=False)
    
    # 1. Instanciation du modèle
    model = RelativePatchPrediction(dataset='cifar10', n_classes=10).to(device)
    classifier = model.construct_classifier().to(device)
    
    # 2. Entraînement SSL (Prétexte)
    train_ssl_task(model, train_loader, epochs=5)
    
    # 3. Entraînement du Classifieur Linéaire sur CIFAR-10
    train_linear_classifier(model, classifier, train_loader, epochs=5)
    
    # 4. Évaluation sur CIFAR-10
    cifar_acc = evaluate_accuracy(model, classifier, test_loader)
    print(f"\n[+] Précision Top-1 Test (CIFAR-10) : {cifar_acc}%")
    
    # 5. Évaluation sur FashionMNIST (Transfert)
    transform_fashion = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    fashion_test = torchvision.datasets.FashionMNIST(root='./data/fashion', train=False, download=True, transform=transform_fashion)
    fashion_loader = data.DataLoader(fashion_test, batch_size=128, shuffle=False)
    
    fashion_acc = evaluate_accuracy(model, classifier, fashion_loader)
    print(f"[+] Précision Top-1 Test (FashionMNIST) : {fashion_acc}%")

if __name__ == '__main__':
    main()