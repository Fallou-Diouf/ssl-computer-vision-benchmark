import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePatchPrediction(nn.Module):
    metrics = ['Loss', 'Acc1']
    metrics_fmt = [':.4e', ':6.2f']

    def __init__(self, dataset='cifar10', n_classes=10):
        super().__init__()
        self.dataset = dataset
        self.n_classes = n_classes

        # 1. Extracteur de caractéristiques (Backbone convolutif partagé)
        # Adapté pour des patchs de ~10x10 pixels (découpés depuis une image 32x32)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))  # Réduit à 128 x 2 x 2
        )

        self.latent_dim = 128 * 2 * 2  # 512 caractéristiques par patch

        # 2. Tête de prédiction de position relative (8 positions possibles autour du centre)
        self.classifier_relative = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 8)
        )

    def extract_patch_pairs(self, images):
        """
        Découpe chaque image 32x32 du batch en une grille 3x3 (patchs 10x10)
        et renvoie le patch central ainsi qu'un patch voisin tiré au hasard (0 à 7).
        """
        batch_size, channels, H, W = images.shape
        patch_h, patch_w = H // 3, W // 3  # 10x10 pour CIFAR-10 (32x32)

        # Patch central (requête)
        center_patches = images[:, :, patch_h:2*patch_h, patch_w:2*patch_w]

        # Les 8 coordonnées relatives (haut-gauche, haut, haut-droite, gauche, droite, bas-gauche, bas, bas-droite)
        offsets = [
            (0, 0),             (0, patch_w),             (0, 2 * patch_w),
            (patch_h, 0),                                 (patch_h, 2 * patch_w),
            (2 * patch_h, 0),   (2 * patch_h, patch_w),   (2 * patch_h, 2 * patch_w)
        ]

        # Choix aléatoire d'un voisin par image dans le batch
        targets = torch.randint(0, 8, (batch_size,), device=images.device)
        neighbor_patches = torch.zeros_like(center_patches)

        for i in range(batch_size):
            label = targets[i].item()
            dh, dw = offsets[label]
            neighbor_patches[i] = images[i, :, dh:dh+patch_h, dw:dw+patch_w]

        return center_patches, neighbor_patches, targets

    def forward(self, images):
        # 1. Extraction des couples (patch_centre, patch_voisin)
        center_patches, neighbor_patches, targets = self.extract_patch_pairs(images)

        # 2. Encodage des deux patchs avec le même réseau (poids partagés)
        z_center = torch.flatten(self.encoder(center_patches), 1)
        z_neighbor = torch.flatten(self.encoder(neighbor_patches), 1)

        # 3. Concaténation des deux vecteurs latents
        z_combined = torch.cat([z_center, z_neighbor], dim=1)

        # 4. Prédiction de la position relative
        logits = self.classifier_relative(z_combined)
        loss = F.cross_entropy(logits, targets)

        # Calcul de la précision sur la tâche prétexte
        acc = (logits.argmax(dim=-1) == targets).float().mean() * 100.0

        return dict(Loss=loss, Acc1=acc), z_center

    def encode(self, images):
        """
        Fonction d'encodage utilisée pour évaluer la représentation latente sur la tâche aval.
        """
        return torch.flatten(self.encoder(images), 1)

    def construct_classifier(self):
        """
        Construit le classifieur linéaire pour la tâche aval de classification sur CIFAR-10.
        """
        return nn.Sequential(
            nn.BatchNorm1d(self.latent_dim, affine=False),
            nn.Linear(self.latent_dim, self.n_classes)
        )