# Projet 1 — ANN + méthode de Taguchi : classification du cancer du sein (WDBC)

Référence : *Artificial Neural Network with Taguchi Method for Robust Classification Model to Improve Classification Accuracy of Breast Cancer* (2021) — https://pmc.ncbi.nlm.nih.gov/articles/PMC7924699/
Données : UCI Breast Cancer Wisconsin (Diagnostic) — 569 observations, 30 variables, 2 classes (357 bénins, 212 malins).

## Contenu du dossier
| Fichier | Rôle |
|---|---|
| `ANN_Taguchi_WDBC.ipynb` | Notebook principal, déjà exécuté (code, explications, tableaux, figures) |
| `ann_taguchi_wdbc.py` | Même code en script Python |
| `data/wdbc.csv` | Données WDBC (colonne `diagnosis` : M = malin, B = bénin + 30 variables) |
| `figures/` | 9 figures (classes, corrélations, courbes d'apprentissage, effets principaux S/N, ANOVA, partitions, matrice de confusion, ROC, comparaison) |
| `results/` | Tous les tableaux de résultats en CSV |
| `requirements.txt` | Bibliothèques nécessaires |

## Exécution
```bash
pip install -r requirements.txt
jupyter notebook ANN_Taguchi_WDBC.ipynb      # ou : python ann_taguchi_wdbc.py
```
Durée : environ 2 minutes sur un PC portable. Lancer depuis ce dossier (chemins relatifs `data/`, `figures/`, `results/`).
Si `data/wdbc.csv` est absent, le notebook télécharge le fichier original sur le site UCI.

## Démarche
1. Exploration et prétraitement (cible : malin = 1, partition stratifiée, StandardScaler ajusté sur l'entraînement).
2. ANN à une couche cachée (scikit-learn `MLPClassifier`, Adam, entropie croisée, arrêt précoce sur la validation).
3. **Taguchi — expérience 1** : tableau orthogonal L9(3⁴) sur neurones cachés (10/15/20), taux d'apprentissage (0,001/0,01/0,05), activation (sigmoïde/tanh/relu), régularisation L2 (0,0001/0,01/0,1). Réponse = 15 accuracies (validation croisée 5-fold × 3) ; S/N « plus grand = meilleur » ; tableau des réponses, effets principaux, ANOVA, expérience de confirmation.
4. **Taguchi — expérience 2** : partitions 50/25/25, 60/20/20, 70/15/15, 80/10/10.
5. Évaluation : 30 exécutions aléatoires, validation croisée 10-fold, comparaison avec 8 autres classifieurs.

## Principaux résultats
Configuration retenue : 20 neurones, sigmoïde, lr = 0,01, alpha = 0,1.

| Indicateur | Article | Ce projet |
|---|---|---|
| Accuracy test (30 exécutions) | 98,8 % | 97,4 % |
| Sensibilité / Spécificité | 99,1 % / 98,4 % | 95,3 % / 98,7 % |
| Accuracy 10-fold CV | 98,7 % | 97,7 % |

À données égales (30 partitions 85/15), l'ANN-Taguchi est le meilleur des 9 modèles comparés (97,9 %), devant la régression logistique (97,8 %) et le SVM RBF (97,5 %). Le plan de Taguchi montre que, sur ce jeu de données, les hyperparamètres testés influencent peu le résultat.
