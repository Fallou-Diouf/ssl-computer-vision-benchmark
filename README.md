# Benchmark d'Apprentissage Auto-Supervisé (SSL) & Prédiction de Position Relative

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Metal MPS](https://img.shields.io/badge/Accélération-macOS_MPS-000000?style=flat-square&logo=apple&logoColor=white)]()
[![License](https://img.shields.io/badge/Licence-MIT-green.style=flat-square)]()

Étude comparative globale des méthodes fondatrices d'apprentissage auto-supervisé (*Self-Supervised Learning*, SSL) en vision par ordinateur (**Context Encoder**, **Rotation Prediction**, **SimCLR**), incluant une **implémentation complète en PyTorch de la tâche prétexte de prédiction de position relative de patchs** ([Doersch et al., ICCV 2015](https://arxiv.org/abs/1505.05192)).

---

## 🌟 Contributions Clés

- **Implémentation de Relative Patch Location** : Conception intégrale d'une architecture siamoise à poids partagés pour prédire la position spatiale relative parmi 8 patchs voisins (Doersch et al., 2015).
- **Protocole d'Évaluation Uniforme (*Linear Probing*)** : Évaluation standardisée sur CIFAR-10 en gelant les encodeurs pré-entraînés et en entraînant une unique couche linéaire de classification.
- **Analyse de la Transférabilité Hors-Domaine** : Évaluation expérimentale des limites de l'adaptation *zero-shot* sur des données hors-distribution (**FashionMNIST**).
- **Automation du Benchmark** : Scripts modulaires et reproductibles (`compare_cifar10.py`, `compare_custom.py`, `train_and_eval_relative_patch.py`).

---

## 📊 Résultats Expérimentaux

### 1. Classification sur le Domaine d'Origine (CIFAR-10 Top-1)
Évaluation par **Linear Probing** avec extraction de caractéristiques à poids gelés :

| Modèle | Tâche Prétexte | Paradigme de Supervision | Précision Train (%) | Précision Test Top-1 (%) |
| :--- | :--- | :--- | :---: | :---: |
| **SimCLR** | Apprentissage contrastif (NT-Xent) | Information Mutuelle | **90,04%** | **92,84%** |
| **Rotation Prediction** | Classification d'angle ($\{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$) | Transformation Géométrique | **79,49%** | **79,91%** |
| **Relative Patch (Ours)** | Position spatiale relative (8 voisins) | Géométrie Spatiale Locale | **66,18%*** | **63,20%** |
| **Context Encoder** | Inpainting (région masquée $64\times64$) | Reconstruction de Pixels ($L_2$) | **53,24%** | **45,77%** |

*\* Précision sur la tâche prétexte après 5 époques d'entraînement.*

### 2. Transférabilité Hors-Domaine (FashionMNIST)
Évaluation *Zero-Shot* directe (sans ré-entraînement de la tête linéaire) sur des images de vêtements en niveaux de gris $28\times28$ converties sur 3 canaux :

| Modèle | Précision Test FashionMNIST (%) | Analyse |
| :--- | :---: | :--- |
| **SimCLR** | **11,47%** | Effondrement au niveau du hasard (~10%) dû à l'inadaptation de la frontière de décision. |
| **Relative Patch (Ours)** | **10,15%** | Confirme qu'un transfert direct sans ré-entraînement du classifieur linéaire échoue. |
| **Context Encoder** | **8,81%** | Changement de domaine sévère (images naturelles complexes $\to$ silhouettes de vêtements). |
| **Rotation Prediction** | **8,29%** | Démontre la nécessité de fine-tuner la tête linéaire sur les caractéristiques cibles. |

---

## 📂 Arborescence du Dépôt

```text
.
├── deepul_helper/                 # Librairie principale et modules de tâches
│   └── tasks/
│       ├── relative_patch_pred.py # Implémentation maison de Doersch et al. (2015)
│       ├── context_encoder.py     # Baseline Inpainting
│       ├── rotation.py            # Baseline Rotation géométrique
│       └── simclr.py              # Baseline Apprentissage contrastif
├── compare_cifar10.py             # Script de benchmark sur CIFAR-10
├── compare_custom.py              # Script d'évaluation du transfert sur FashionMNIST
├── train_and_eval_relative_patch.py # Entraînement SSL & Linear Probing de notre méthode
├── Compte_Rendu/                  # Code source LaTeX du rapport scientifique de 3 pages
├── environment.yml                # Configuration de l'environnement Conda
└── README.md                      # Documentation du projet