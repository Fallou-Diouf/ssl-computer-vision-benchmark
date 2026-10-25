import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Adaptations Mac / MPS
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

def to_device(tensor_or_module, *args, **kwargs):
    return tensor_or_module.to(device)

torch.Tensor.cuda = to_device
torch.nn.Module.cuda = to_device

from deepul_helper.demos import load_model_and_data
from deepul_helper.utils import accuracy

def evaluate_on_dataset(model, linear_classifier, loader):
    correct1 = 0
    with torch.no_grad():
        for images, target in loader:
            if isinstance(images, (tuple, list)):
                images = images[0]
            images = images.to(device)
            target = target.to(device)

            zs = model.encode(images)
            if zs.dim() > 2:
                zs = torch.flatten(zs, 1)

            logits = linear_classifier(zs)
            acc1, _ = accuracy(logits, target, topk=(1, 5))

            correct1 += acc1.item() * logits.shape[0]

    total = len(loader.dataset)
    return round(correct1 / total, 2)

def main():
    models_to_evaluate = ['context_encoder', 'rotation', 'simclr']
    results = []

    print(f"Évaluation globale sur FashionMNIST (Device: {device})...\n")

    for model_name in models_to_evaluate:
        print(f"-> Traitement du modèle : {model_name}...")
        model, linear_classifier, _, _ = load_model_and_data(model_name)

        # Taille cible stricte selon le modèle SSL
        target_size = (128, 128) if model_name == 'context_encoder' else (32, 32)

        # Transformation dédiée et standardisée
        custom_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        train_set = torchvision.datasets.FashionMNIST(root='./data/fashion', train=True, download=True, transform=custom_transform)
        test_set = torchvision.datasets.FashionMNIST(root='./data/fashion', train=False, download=True, transform=custom_transform)

        train_loader = data.DataLoader(train_set, batch_size=128, shuffle=False)
        test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False)

        train_acc = evaluate_on_dataset(model, linear_classifier, train_loader)
        test_acc = evaluate_on_dataset(model, linear_classifier, test_loader)

        results.append({
            'Modèle SSL': model_name,
            'Train Top-1 (%)': train_acc,
            'Test Top-1 (%)': test_acc
        })

    # Tableau comparatif
    df = pd.DataFrame(results)
    print("\n=================== TABLEAU COMPARATIF (FashionMNIST) ===================")
    print(df.to_string(index=False))

    # Graphique
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df['Modèle SSL'], df['Test Top-1 (%)'], color=['#3498db', '#e67e22', '#2ecc71'])
    plt.title('Comparaison des performances de transfert (FashionMNIST)', fontsize=12)
    plt.xlabel('Tâche prétexte SSL', fontsize=10)
    plt.ylabel('Précision Top-1 Test (%)', fontsize=10)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval}%", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('comparaison_fashionmnist.png', dpi=300)
    print(f"\n[+] Graphique sauvegardé sous : 'comparaison_fashionmnist.png'")

if __name__ == '__main__':
    main()