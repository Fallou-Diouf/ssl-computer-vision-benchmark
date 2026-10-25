import torch
import torch.utils.data as data
import matplotlib
matplotlib.use('Agg') # Backend non-interactif pour générer proprement l'image PNG
import matplotlib.pyplot as plt
import pandas as pd
from deepul_helper.demos import load_model_and_data, evaluate_classifier


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

def to_device(tensor_or_module, *args, **kwargs):
    return tensor_or_module.to(device)

torch.Tensor.cuda = to_device
torch.nn.Module.cuda = to_device

def main():
    models_to_evaluate = ['context_encoder', 'rotation', 'simclr']
    results = []

    print(f"Évaluation globale sur CIFAR-10 (Device: {device})...\n")

    for model_name in models_to_evaluate:
        print(f"-> Traitement du modèle : {model_name}...")
        model, linear_classifier, train_loader, test_loader = load_model_and_data(model_name)
        
        train_acc1, _ = evaluate_classifier(model, linear_classifier, train_loader)
        test_acc1, _ = evaluate_classifier(model, linear_classifier, test_loader)
        
        results.append({
            'Modèle SSL': model_name,
            'Train Top-1 (%)': round(train_acc1, 2),
            'Test Top-1 (%)': round(test_acc1, 2)
        })

    # 2. Création et affichage du tableau
    df = pd.DataFrame(results)
    print("\n=================== TABLEAU COMPARATIF (CIFAR-10) ===================")
    print(df.to_string(index=False))

    # 3. Génération du graphique
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df['Modèle SSL'], df['Test Top-1 (%)'], color=['#3498db', '#e67e22', '#2ecc71'])
    plt.title('Comparaison des performances de classification linéaire (CIFAR-10)', fontsize=12)
    plt.xlabel('Tâche prétexte SSL', fontsize=10)
    plt.ylabel('Précision Top-1 Test (%)', fontsize=10)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajout des valeurs au-dessus des barres
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval}%", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path = 'comparaison_cifar10.png'
    plt.savefig(output_path, dpi=300)
    print(f"\n[+] Graphique sauvegardé avec succès sous : '{output_path}'")

if __name__ == '__main__':
    main()