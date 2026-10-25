import matplotlib
# Changer le backend pour permettre l'affichage interactif sur Mac
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data

# Patch DataLoader pour compatibilité macOS / MPS
_original_dataloader = data.DataLoader

def _custom_dataloader(*args, **kwargs):
    kwargs['num_workers'] = 0
    kwargs['pin_memory'] = False
    return _original_dataloader(*args, **kwargs)

data.DataLoader = _custom_dataloader

# Redirection CUDA -> MPS / CPU
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

from deepul_helper.demos import (
    load_model_and_data, 
    evaluate_accuracy, 
    display_nearest_neighbors, 
    show_context_encoder_inpainting
)

def main():
    print(f"Utilisation du périphérique : {device}\n")

    print("==================================================")
    print("DÉMO 1 : Context Encoder (Inpainting)")
    print("==================================================")
    show_context_encoder_inpainting()

    print("==================================================")
    print("DÉMO 2 : Rotation Prediction")
    print("==================================================")
    model_rot, classif_rot, train_loader_rot, test_loader_rot = load_model_and_data('rotation')
    # evaluate_accuracy affiche directement les résultats Top-1 et Top-5
    evaluate_accuracy(model_rot, classif_rot, train_loader_rot, test_loader_rot)

    display_nearest_neighbors('rotation', model_rot, test_loader_rot)

    print("==================================================")
    print("DÉMO 3 : SimCLR (Apprentissage Contrastif)")
    print("==================================================")
    model_sim, classif_sim, train_loader_sim, test_loader_sim = load_model_and_data('simclr')
    evaluate_accuracy(model_sim, classif_sim, train_loader_sim, test_loader_sim)

    display_nearest_neighbors('simclr', model_sim, test_loader_sim)

if __name__ == '__main__':
    main()