# -*- coding: utf-8 -*-
# Projet 1 - ANN + méthode de Taguchi sur WDBC (version script du notebook)

import matplotlib
matplotlib.use("Agg")  # mode script : figures enregistrées dans figures/ sans ouvrir de fenêtre

try:
    display
except NameError:
    def display(x):
        print(x)


# # Projet 1 — Réseaux de Neurones Artificiels (ANN) : Classification du cancer du sein
# ### ANN optimisé par la méthode de Taguchi — jeu de données Wisconsin Diagnostic Breast Cancer (WDBC)
# 
# **Article de référence :** *Artificial Neural Network with Taguchi Method for Robust Classification Model to Improve Classification Accuracy of Breast Cancer* (PeerJ Computer Science, 2021) — https://pmc.ncbi.nlm.nih.gov/articles/PMC7924699/
# 
# **Données :** UCI — Breast Cancer Wisconsin (Diagnostic) — https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
# 569 observations, 30 variables explicatives (caractéristiques des noyaux cellulaires), 2 classes : **bénin (B)** et **malin (M)**.
# 
# #### Plan du notebook
# 1. Chargement et exploration des données
# 2. Prétraitement (encodage, partition, normalisation)
# 3. Modèle ANN (perceptron multicouche, arrêt précoce sur la validation)
# 4. **Méthode de Taguchi — Expérience 1** : réglage des hyperparamètres avec un tableau orthogonal L9(3⁴), rapport signal/bruit (S/N), effets principaux, ANOVA, expérience de confirmation
# 5. **Méthode de Taguchi — Expérience 2** : choix de la partition des données (50/25/25, 60/20/20, 70/15/15, 80/10/10)
# 6. Évaluation finale : 30 exécutions aléatoires + validation croisée 10-fold (accuracy, sensibilité, spécificité, F-score, AUC)
# 7. Comparaison avec d'autres classifieurs et avec l'article
# 8. Conclusion

# ## 0. Imports et configuration

import os, copy, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, roc_auc_score,
                             roc_curve, classification_report, log_loss)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

SEED = 42
np.random.seed(SEED)
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
print("OK")

# ## 1. Chargement et exploration des données

DATA_PATH = "data/wdbc.csv"
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    # Secours : fichier brut UCI (id, diagnostic, 30 variables, sans en-tête)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    base = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
            "concavity", "concave points", "symmetry", "fractal dimension"]
    cols = ["id", "diagnosis"] + [f"mean {b}" for b in base] + [f"{b} error" for b in base] + [f"worst {b}" for b in base]
    df = pd.read_csv(url, header=None, names=cols).drop(columns="id")

print("Dimensions :", df.shape)
display(df.head())

print("Valeurs manquantes :", int(df.isna().sum().sum()))
print("\nRépartition des classes :")
print(df["diagnosis"].value_counts().rename({"B": "Bénin (B)", "M": "Malin (M)"}))
display(df.describe().T.head(10))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
counts = df["diagnosis"].value_counts()
axes[0].bar(["Bénin", "Malin"], [counts["B"], counts["M"]], color=["#4C9BE8", "#E8684C"])
for i, v in enumerate([counts["B"], counts["M"]]):
    axes[0].text(i, v + 5, str(v), ha="center")
axes[0].set_title("Répartition des classes")
axes[0].set_ylabel("Nombre d'observations")

sns.kdeplot(data=df, x="worst concave points", hue="diagnosis", fill=True, ax=axes[1],
            palette={"B": "#4C9BE8", "M": "#E8684C"})
axes[1].set_title("Distribution de « worst concave points »")
plt.tight_layout(); plt.savefig("figures/01_classes.png", bbox_inches="tight"); plt.show()

X_all = df.drop(columns="diagnosis")
corr = X_all.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, xticklabels=True, yticklabels=True, cbar_kws={"shrink": .7})
plt.title("Matrice de corrélation des 30 variables")
plt.xticks(fontsize=7); plt.yticks(fontsize=7)
plt.tight_layout(); plt.savefig("figures/02_correlation.png", bbox_inches="tight"); plt.show()

y_tmp = (df["diagnosis"] == "M").astype(int)
top = X_all.corrwith(y_tmp).abs().sort_values(ascending=False).head(10)
print("10 variables les plus corrélées au diagnostic (|r|) :")
print(top.round(3))

# ## 2. Prétraitement
# - Encodage de la cible : **Malin = 1 (classe positive)**, Bénin = 0.
# - Partition stratifiée **entraînement / validation / test** (la validation sert à l'arrêt précoce et à la sélection Taguchi ; le test n'est utilisé qu'à la fin).
# - Normalisation `StandardScaler` **ajustée uniquement sur l'entraînement** (pas de fuite d'information).

X = df.drop(columns="diagnosis").values.astype(float)
y = (df["diagnosis"] == "M").astype(int).values
feature_names = df.drop(columns="diagnosis").columns.tolist()

def split_data(X, y, train=0.70, val=0.15, test=0.15, seed=SEED):
    # Partition stratifiée train/val/test + normalisation ajustée sur train
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, train_size=train, stratify=y, random_state=seed)
    rel_test = test / (val + test)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=seed)
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te), y_tr, y_va, y_te

X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y)
print(f"Entraînement : {X_tr.shape[0]} | Validation : {X_va.shape[0]} | Test : {X_te.shape[0]}")

# ## 3. Modèle ANN
# Comme dans l'article : réseau *feed-forward* à **une couche cachée**, 30 entrées, sortie probabiliste (softmax/sigmoïde), fonction de coût **entropie croisée**, entraînement par descente de gradient, **arrêt précoce** sur l'erreur de validation (on conserve les meilleurs poids).
# 
# La fonction `train_ann` entraîne époque par époque (`partial_fit`) pour suivre les courbes d'apprentissage.

def train_ann(X_tr, y_tr, X_va, y_va, hidden=15, lr=0.01, activation="logistic", alpha=1e-4, batch=32,
              seed=SEED, max_epochs=300, patience=25, return_history=False):
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), activation=activation, solver="adam",
                        learning_rate_init=lr, batch_size=batch, alpha=alpha,
                        max_iter=1, warm_start=True, random_state=seed)
    classes = np.array([0, 1])
    best_loss, best_model, wait = np.inf, None, 0
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(max_epochs):
        clf.partial_fit(X_tr, y_tr, classes=classes)
        p_va = clf.predict_proba(X_va)[:, 1]
        vl = log_loss(y_va, p_va, labels=[0, 1])
        if return_history:
            p_tr = clf.predict_proba(X_tr)[:, 1]
            hist["train_loss"].append(log_loss(y_tr, p_tr, labels=[0, 1]))
            hist["val_loss"].append(vl)
            hist["train_acc"].append(accuracy_score(y_tr, (p_tr >= .5).astype(int)))
            hist["val_acc"].append(accuracy_score(y_va, (p_va >= .5).astype(int)))
        if vl < best_loss - 1e-5:
            best_loss, best_model, wait, best_epoch = vl, copy.deepcopy(clf), 0, epoch + 1
        else:
            wait += 1
            if wait >= patience:
                break
    best_model.best_epoch_ = best_epoch
    return (best_model, hist) if return_history else best_model

def metrics(model, X_, y_):
    proba = model.predict_proba(X_)[:, 1]
    pred = (proba >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_, pred, labels=[0, 1]).ravel()
    return {"accuracy": (tp + tn) / len(y_),
            "sensitivity": tp / (tp + fn),       # rappel classe maligne
            "specificity": tn / (tn + fp),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "f1": f1_score(y_, pred),
            "auc": roc_auc_score(y_, proba)}

def train_with(p, X_tr, y_tr, X_va, y_va, seed=SEED, **kw):
    # Entraîne un ANN avec un dictionnaire de paramètres Taguchi
    return train_ann(X_tr, y_tr, X_va, y_va, hidden=p["A_neurones"], lr=p["B_lr"],
                     activation=p["C_activation"], alpha=p["D_alpha"], seed=seed, **kw)

# Modèle de départ (configuration de l'article : 15 neurones cachés, sigmoïde)
t0 = time.time()
base_model, base_hist = train_ann(X_tr, y_tr, X_va, y_va, hidden=15, lr=0.01, activation="logistic",
                                  batch=32, return_history=True)
print(f"Meilleure époque : {base_model.best_epoch_}  ({time.time()-t0:.1f}s)")
print("Validation :", {k: round(float(v), 4) for k, v in metrics(base_model, X_va, y_va).items()})

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ep = np.arange(1, len(base_hist["train_loss"]) + 1)
ax[0].plot(ep, base_hist["train_loss"], label="Entraînement")
ax[0].plot(ep, base_hist["val_loss"], label="Validation")
ax[0].axvline(base_model.best_epoch_, ls="--", c="gray", label=f"Meilleure époque ({base_model.best_epoch_})")
ax[0].set_title("Entropie croisée"); ax[0].set_xlabel("Époque"); ax[0].legend()
ax[1].plot(ep, base_hist["train_acc"], label="Entraînement")
ax[1].plot(ep, base_hist["val_acc"], label="Validation")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("Époque"); ax[1].legend()
plt.tight_layout(); plt.savefig("figures/03_courbes_apprentissage.png", bbox_inches="tight"); plt.show()

# ## 4. Méthode de Taguchi — Expérience 1 : hyperparamètres de l'ANN
# 
# La méthode de Taguchi remplace le plan factoriel complet (ici 3⁴ = 81 combinaisons) par un **tableau orthogonal** : avec **L9(3⁴)**, 9 essais suffisent pour estimer l'effet principal de 4 facteurs à 3 niveaux, chaque niveau d'un facteur apparaissant le même nombre de fois avec chaque niveau des autres facteurs.
# 
# | Facteur | Niveau 1 | Niveau 2 | Niveau 3 |
# |---|---|---|---|
# | A — Neurones cachés | 10 | 15 | 20 |
# | B — Taux d'apprentissage (Adam) | 0,001 | 0,01 | 0,05 |
# | C — Fonction d'activation | logistic (sigmoïde) | tanh | relu |
# | D — Régularisation L2 (alpha) | 0,0001 | 0,01 | 0,1 |
# 
# L'article fait varier les neurones cachés entre 10 et 20 ; nous ajoutons trois facteurs usuels pour rendre l'étude plus complète (taille de lot fixée à 32).
# 
# **Réponse et robustesse :** l'ensemble de **test (15 %) est mis de côté** et n'intervient jamais dans le réglage. Sur les 85 % restants (« développement »), chaque essai est évalué par **validation croisée stratifiée 5-fold répétée 3 fois** (graines différentes = facteurs de bruit), soit **n = 15 mesures d'accuracy** par essai (chaque fold : ~97 observations, avec 15 % du fold d'entraînement réservés à l'arrêt précoce). On calcule le **rapport S/N « plus c'est grand, mieux c'est »** :
# 
# $$ S/N = -10\,\log_{10}\left(\frac{1}{n}\sum_{i=1}^{n}\frac{1}{y_i^2}\right) $$
# 
# Un S/N élevé signifie une accuracy à la fois **élevée et stable**.

L9 = np.array([[1,1,1,1],[1,2,2,2],[1,3,3,3],
               [2,1,2,3],[2,2,3,1],[2,3,1,2],
               [3,1,3,2],[3,2,1,3],[3,3,2,1]])
levels = {"A_neurones": [10, 15, 20],
          "B_lr": [0.001, 0.01, 0.05],
          "C_activation": ["logistic", "tanh", "relu"],
          "D_alpha": [1e-4, 1e-2, 1e-1]}
factors = list(levels.keys())

# Vérification de l'orthogonalité : chaque paire de colonnes contient les 9 couples une seule fois
for i in range(4):
    for j in range(i + 1, 4):
        assert len(set(zip(L9[:, i], L9[:, j]))) == 9
print("Tableau L9(3^4) orthogonal vérifié.")

def sn_larger_better(y):
    y = np.asarray(y)
    return -10 * np.log10(np.mean(1.0 / y**2))

# Ensemble de test mis de côté une fois pour toutes (15 %)
X_dev, X_hold, y_dev, y_hold = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)

def cv_accuracies(p, n_repeats=3, k=5):
    # Validation croisée k-fold répétée sur l'ensemble de développement -> liste de k*n_repeats accuracies
    accs = []
    for rep in range(n_repeats):
        skf_ = StratifiedKFold(n_splits=k, shuffle=True, random_state=100 + rep)
        for f_i, (tr, va) in enumerate(skf_.split(X_dev, y_dev)):
            Xa, Xs, ya, ys = train_test_split(X_dev[tr], y_dev[tr], test_size=.15, stratify=y_dev[tr],
                                              random_state=10 * rep + f_i)
            sc = StandardScaler().fit(Xa)
            m = train_with(p, sc.transform(Xa), ya, sc.transform(Xs), ys, seed=10 * rep + f_i)
            accs.append(accuracy_score(y_dev[va], m.predict(sc.transform(X_dev[va]))))
    return np.array(accs)

rows = []
t0 = time.time()
for run, lv in enumerate(L9, start=1):
    p = {f: levels[f][l - 1] for f, l in zip(factors, lv)}
    accs = cv_accuracies(p)
    rows.append({"essai": run, **{f: l for f, l in zip(factors, lv)}, **{f + "_val": p[f] for f in factors},
                 "acc_moy": np.mean(accs), "acc_std": np.std(accs), "S/N (dB)": sn_larger_better(accs)})
taguchi1 = pd.DataFrame(rows)
print(f"Durée : {time.time()-t0:.0f}s")
taguchi1.to_csv("results/taguchi_L9_resultats.csv", index=False)
display(taguchi1[["essai"] + [f + "_val" for f in factors] + ["acc_moy", "acc_std", "S/N (dB)"]].round(4))

# ### 4.1 Tableau des réponses (effets principaux) et niveaux optimaux

resp = pd.DataFrame(index=[f"Niveau {k}" for k in (1, 2, 3)])
resp_acc = resp.copy()
for f in factors:
    resp[f] = [taguchi1.loc[taguchi1[f] == k, "S/N (dB)"].mean() for k in (1, 2, 3)]
    resp_acc[f] = [taguchi1.loc[taguchi1[f] == k, "acc_moy"].mean() for k in (1, 2, 3)]
resp.loc["Delta"] = resp.max() - resp.min()
resp.loc["Rang"] = resp.loc["Delta"].rank(ascending=False).astype(int)
print("Tableau des réponses — S/N moyen (dB) :")
display(resp.round(3))

best_levels = {f: int(np.argmax(resp.loc[[f"Niveau {k}" for k in (1, 2, 3)], f].values)) + 1 for f in factors}
best_params = {f: levels[f][best_levels[f] - 1] for f in factors}
print("Niveaux optimaux :", best_levels)
print("Paramètres optimaux :", best_params)
resp.to_csv("results/taguchi_tableau_reponses.csv")

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
titles = ["A : neurones cachés", "B : taux d'apprentissage", "C : activation", "D : régularisation L2"]
grand = taguchi1["S/N (dB)"].mean()
for ax, f, t in zip(axes, factors, titles):
    vals = resp.loc[[f"Niveau {k}" for k in (1, 2, 3)], f].values
    ax.plot([str(v) for v in levels[f]], vals, "o-", lw=2, ms=8, color="#2E6FBA")
    ax.axhline(grand, ls="--", c="gray", lw=1)
    k = np.argmax(vals); ax.plot(k, vals[k], "o", ms=13, mfc="none", mec="#E8684C", mew=2)
    ax.set_title(t)
axes[0].set_ylabel("S/N moyen (dB)")
fig.suptitle("Graphique des effets principaux (S/N) — plus grand = meilleur", y=1.03)
plt.tight_layout(); plt.savefig("figures/04_effets_principaux_SN.png", bbox_inches="tight"); plt.show()

# ### 4.2 ANOVA sur le rapport S/N
# L'ANOVA décompose la variabilité totale du S/N entre les facteurs et donne le **pourcentage de contribution** de chacun. Avec L9(3⁴), les 4 facteurs utilisent les 8 degrés de liberté : on regroupe (*pooling*) le facteur le moins influent dans l'erreur, comme le préconise Taguchi.

sn = taguchi1["S/N (dB)"].values
SS_T = np.sum((sn - sn.mean())**2)
anova = []
for f in factors:
    ss = sum(3 * (taguchi1.loc[taguchi1[f] == k, "S/N (dB)"].mean() - sn.mean())**2 for k in (1, 2, 3))
    anova.append({"Facteur": f, "ddl": 2, "SS": ss})
anova = pd.DataFrame(anova)
pooled = anova.loc[anova["SS"].idxmin(), "Facteur"]
SS_e, df_e = anova.loc[anova["Facteur"] == pooled, "SS"].values[0], 2
anova["MS"] = anova["SS"] / anova["ddl"]
anova["F"] = np.where(anova["Facteur"] == pooled, np.nan, anova["MS"] / (SS_e / df_e))
anova["Contribution (%)"] = 100 * anova["SS"] / SS_T
anova.loc[anova["Facteur"] == pooled, "Facteur"] = pooled + " (regroupé → erreur)"
anova.loc[len(anova)] = ["Total", 8, SS_T, np.nan, np.nan, 100.0]
anova.to_csv("results/taguchi_anova.csv", index=False)
display(anova.round(4))

a = anova.iloc[:-1]
plt.figure(figsize=(7, 3.5))
plt.barh(a["Facteur"], a["Contribution (%)"], color="#2E6FBA")
for i, v in enumerate(a["Contribution (%)"]):
    plt.text(v + 0.5, i, f"{v:.1f} %", va="center")
plt.xlabel("Contribution à la variance du S/N (%)"); plt.title("ANOVA — contribution des facteurs")
plt.gca().invert_yaxis()
plt.tight_layout(); plt.savefig("figures/05_anova_contribution.png", bbox_inches="tight"); plt.show()

# ### 4.3 Expérience de confirmation
# On prédit le S/N à la configuration optimale (modèle additif de Taguchi) puis on le **vérifie expérimentalement**. On compare aussi avec la configuration de départ de l'article (15 neurones, sigmoïde ; lr = 0,01 et alpha = 0,0001).
# 
# $$ \widehat{S/N}_{opt} = \overline{S/N} + \sum_{f}\left(\overline{S/N}_{f,\,opt} - \overline{S/N}\right) $$

pred_sn = grand + sum(resp.loc[f"Niveau {best_levels[f]}", f] - grand for f in factors)

ref_params = {"A_neurones": 15, "B_lr": 0.01, "C_activation": "logistic", "D_alpha": 1e-4}
best_run = taguchi1.loc[taguchi1["S/N (dB)"].idxmax()]
best_run_params = {f: levels[f][int(best_run[f]) - 1] for f in factors}
candidates = {"Optimum prédit (Taguchi)": best_params,
              f"Meilleur essai L9 (n°{int(best_run['essai'])})": best_run_params,
              "Référence (article)": ref_params}
conf_rows = []
for name, p in candidates.items():
    a = cv_accuracies(p)
    conf_rows.append({"Configuration": name, "Paramètres": str(p), "Acc. CV moy.": a.mean(),
                      "Écart-type": a.std(), "S/N observé (dB)": sn_larger_better(a)})
confirm = pd.DataFrame(conf_rows)
final_name = confirm.loc[confirm["S/N observé (dB)"].idxmax(), "Configuration"]
final_params = candidates[final_name]
confirm.to_csv("results/taguchi_confirmation.csv", index=False)
print(f"S/N prédit à l'optimum : {pred_sn:.3f} dB")
print(f"Configuration finale retenue : {final_name} -> {final_params}")
display(confirm.round(4))

# **Interprétation.** Les écarts de S/N entre niveaux (Delta < 0,04 dB, soit moins de 0,5 point d'accuracy) sont du même ordre que la dispersion due aux répétitions, et les rapports F de l'ANOVA restent faibles (F < 3 pour 2 et 2 ddl, alors que le seuil à 5 % est F = 19). Sur WDBC, presque toutes les configurations raisonnables donnent ~97 % : **les hyperparamètres testés ont peu d'influence**. Dans ce cas, l'optimum « additif » prédit par Taguchi n'est pas forcément confirmé ; on retient alors, parmi les candidats, la configuration au **S/N observé le plus élevé** (règle classique quand la confirmation échoue).

# ## 5. Méthode de Taguchi — Expérience 2 : partition des données
# L'article compare 4 partitions entraînement/validation/test avec un plan L4. Nous évaluons ces 4 niveaux avec la configuration finale retenue, 15 répétitions chacune, et retenons la partition au S/N (accuracy de **validation**) le plus élevé.

splits = {"50/25/25": (.50, .25, .25), "60/20/20": (.60, .20, .20),
          "70/15/15": (.70, .15, .15), "80/10/10": (.80, .10, .10)}
rows = []
for name, (a_, b_, c_) in splits.items():
    accs = []
    for r in range(15):
        Xtr_, Xva_, _, ytr_, yva_, _ = split_data(X, y, a_, b_, c_, seed=200 + r)
        m = train_with(final_params, Xtr_, ytr_, Xva_, yva_, seed=200 + r)
        accs.append(accuracy_score(yva_, m.predict(Xva_)))
    rows.append({"Partition": name, "n_val": len(yva_), "Acc. val. moy.": np.mean(accs),
                 "Écart-type": np.std(accs), "S/N (dB)": sn_larger_better(accs)})
part = pd.DataFrame(rows)
best_split = part.loc[part["S/N (dB)"].idxmax(), "Partition"]
part.to_csv("results/taguchi_partitions.csv", index=False)
print("Partition retenue :", best_split)
display(part.round(4))

plt.figure(figsize=(6.5, 3.6))
plt.bar(part["Partition"], part["S/N (dB)"], color=["#E8684C" if p == best_split else "#9DB8D9" for p in part["Partition"]])
plt.ylim(part["S/N (dB)"].min() - 0.3, part["S/N (dB)"].max() + 0.2)
plt.ylabel("S/N (dB)"); plt.title("Effet de la partition des données (S/N)")
plt.tight_layout(); plt.savefig("figures/06_partitions_SN.png", bbox_inches="tight"); plt.show()

# > **Remarque :** avec de petites validations (80/10/10 → 57 observations), l'accuracy varie par pas de ~1,8 point, ce qui rend la comparaison bruitée. Ici 60/20/20 et 70/15/15 sont quasi à égalité ; dans la suite nous utilisons **70/15/15**, retenue par l'article, pour rester comparable.

# ## 6. Évaluation finale du modèle ANN-Taguchi
# ### 6.1 Trente exécutions aléatoires (70/15/15)
# Comme dans l'article, on répète 30 fois (nouvelle partition + nouvelle initialisation) et on rapporte la moyenne et l'écart-type sur l'ensemble de **test**.

res30, cms = [], np.zeros((2, 2), int)
for r in range(30):
    Xtr_, Xva_, Xte_, ytr_, yva_, yte_ = split_data(X, y, .70, .15, .15, seed=1000 + r)
    m = train_with(final_params, Xtr_, ytr_, Xva_, yva_, seed=1000 + r)
    res30.append({"run": r + 1, "train_acc": accuracy_score(ytr_, m.predict(Xtr_)),
                  "val_acc": accuracy_score(yva_, m.predict(Xva_)),
                  **{"test_" + k: v for k, v in metrics(m, Xte_, yte_).items()}})
    cms += confusion_matrix(yte_, m.predict(Xte_), labels=[0, 1])
res30 = pd.DataFrame(res30)
res30.to_csv("results/30_executions.csv", index=False)
summary30 = res30.drop(columns="run").agg(["mean", "std", "min", "max"]).T
display(summary30.round(4))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.heatmap(cms, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[0],
            xticklabels=["Bénin", "Malin"], yticklabels=["Bénin", "Malin"])
ax[0].set_xlabel("Prédit"); ax[0].set_ylabel("Réel")
ax[0].set_title("Matrice de confusion cumulée (30 tests)")
ax[1].boxplot([res30["train_acc"], res30["val_acc"], res30["test_accuracy"]])
ax[1].set_xticks([1, 2, 3]); ax[1].set_xticklabels(["Entraînement", "Validation", "Test"])
ax[1].set_title("Accuracy sur 30 exécutions"); ax[1].set_ylabel("Accuracy")
plt.tight_layout(); plt.savefig("figures/07_confusion_30runs.png", bbox_inches="tight"); plt.show()

# ### 6.2 Modèle final unique : rapport de classification et courbe ROC

final_model, final_hist = train_with(final_params, X_tr, y_tr, X_va, y_va, return_history=True)
print(classification_report(y_te, final_model.predict(X_te), target_names=["Bénin", "Malin"], digits=4))
fpr, tpr, _ = roc_curve(y_te, final_model.predict_proba(X_te)[:, 1])
auc = roc_auc_score(y_te, final_model.predict_proba(X_te)[:, 1])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(fpr, tpr, lw=2, label=f"ANN-Taguchi (AUC = {auc:.4f})"); ax[0].plot([0, 1], [0, 1], "--", c="gray")
ax[0].set_xlabel("Taux de faux positifs"); ax[0].set_ylabel("Taux de vrais positifs (sensibilité)")
ax[0].set_title("Courbe ROC — ensemble de test"); ax[0].legend(loc="lower right")
ep = np.arange(1, len(final_hist["val_loss"]) + 1)
ax[1].plot(ep, final_hist["train_loss"], label="Entraînement"); ax[1].plot(ep, final_hist["val_loss"], label="Validation")
ax[1].axvline(final_model.best_epoch_, ls="--", c="gray", label=f"Arrêt (époque {final_model.best_epoch_})")
ax[1].set_title("Entropie croisée — modèle final"); ax[1].set_xlabel("Époque"); ax[1].legend()
plt.tight_layout(); plt.savefig("figures/08_roc_modele_final.png", bbox_inches="tight"); plt.show()

# ### 6.3 Validation croisée stratifiée 10-fold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
cv_rows = []
for k, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
    # 15 % de la partie entraînement sert de validation pour l'arrêt précoce
    Xtr_, Xva_, ytr_, yva_ = train_test_split(X[tr_idx], y[tr_idx], test_size=.15, stratify=y[tr_idx], random_state=k)
    sc = StandardScaler().fit(Xtr_)
    m = train_with(final_params, sc.transform(Xtr_), ytr_, sc.transform(Xva_), yva_, seed=k)
    cv_rows.append({"fold": k, **metrics(m, sc.transform(X[te_idx]), y[te_idx])})
cv = pd.DataFrame(cv_rows)
cv.loc["Moyenne"] = cv.mean(numeric_only=True); cv.loc["Écart-type"] = cv.iloc[:10].std(numeric_only=True)
cv.loc[["Moyenne", "Écart-type"], "fold"] = np.nan
cv.to_csv("results/validation_croisee_10fold.csv")
display(cv.round(4))

# ## 7. Comparaison avec d'autres classifieurs
# Mêmes 30 partitions **85 % / 15 %** (entraînement / test) pour tous les modèles, normalisation ajustée sur l'entraînement. Pour l'ANN-Taguchi, 15 % de la partie entraînement servent à l'arrêt précoce : chaque modèle dispose donc des mêmes données.

models = {
    "Régression logistique": LogisticRegression(max_iter=5000),
    "SVM linéaire": SVC(kernel="linear", probability=True),
    "SVM RBF": SVC(kernel="rbf", probability=True),
    "k-NN (k=5)": KNeighborsClassifier(5),
    "Arbre de décision": DecisionTreeClassifier(random_state=SEED),
    "Forêt aléatoire": RandomForestClassifier(300, random_state=SEED),
    "Naive Bayes": GaussianNB(),
    "ANN par défaut (100 neurones, sklearn)": MLPClassifier(max_iter=1000, random_state=SEED),
}
comp = {name: [] for name in models}
for r in range(30):
    Xtr_, Xte_, ytr_, yte_ = train_test_split(X, y, test_size=.15, stratify=y, random_state=1000 + r)
    for name, mdl in models.items():
        pipe = make_pipeline(StandardScaler(), mdl).fit(Xtr_, ytr_)
        comp[name].append(accuracy_score(yte_, pipe.predict(Xte_)))
    # ANN-Taguchi : même ensemble d'entraînement (85 %), dont 15 % réservés à l'arrêt précoce
    Xa, Xs, ya, ys = train_test_split(Xtr_, ytr_, test_size=.15, stratify=ytr_, random_state=r)
    sc = StandardScaler().fit(Xa)
    m = train_with(final_params, sc.transform(Xa), ya, sc.transform(Xs), ys, seed=r)
    comp.setdefault("ANN-Taguchi (ce projet)", []).append(accuracy_score(yte_, m.predict(sc.transform(Xte_))))
comp_df = pd.DataFrame({"Accuracy test moy.": {k: np.mean(v) for k, v in comp.items()},
                        "Écart-type": {k: np.std(v) for k, v in comp.items()}}).sort_values("Accuracy test moy.", ascending=False)
comp_df.to_csv("results/comparaison_classifieurs.csv")
display(comp_df.round(4))

plt.figure(figsize=(8, 4.5))
colors = ["#E8684C" if "Taguchi" in n else "#9DB8D9" for n in comp_df.index]
plt.barh(comp_df.index, comp_df["Accuracy test moy."], xerr=comp_df["Écart-type"], color=colors, capsize=3)
plt.xlim(0.88, 1.0); plt.gca().invert_yaxis()
plt.xlabel("Accuracy test moyenne (30 partitions)"); plt.title("Comparaison des classifieurs")
plt.tight_layout(); plt.savefig("figures/09_comparaison_classifieurs.png", bbox_inches="tight"); plt.show()

s = summary30["mean"]
article = pd.DataFrame({
    "Article (2021)": [0.988, 0.991, 0.984, 0.988, 0.987],
    "Ce projet": [s["test_accuracy"], s["test_sensitivity"], s["test_specificity"], s["test_f1"],
                  cv.loc["Moyenne", "accuracy"]]},
    index=["Accuracy test (30 runs)", "Sensibilité", "Spécificité", "F-score", "Accuracy 10-fold CV"])
article.to_csv("results/comparaison_article.csv")
display(article.round(4))

# ## 8. Conclusion
# *(Valeurs obtenues lors de notre exécution ; elles peuvent varier légèrement selon la version de scikit-learn.)*
# 
# **Configuration finale :** 20 neurones cachés, activation sigmoïde (logistic), taux d'apprentissage 0,01, régularisation L2 alpha = 0,1 (essai n°8 du plan L9, confirmé comme meilleur S/N observé).
# 
# | Indicateur | Article (2021) | Ce projet |
# |---|---|---|
# | Accuracy test, 30 exécutions (70/15/15) | 98,8 % | 97,4 % ± 1,6 |
# | Sensibilité / Spécificité | 99,1 % / 98,4 % | 95,3 % / 98,7 % |
# | F-score | 98,8 % | 96,5 % |
# | Accuracy validation croisée 10-fold | 98,7 % | 97,7 % ± 2,2 |
# | AUC (30 exécutions) | — | 0,995 |
# 
# 1. **La méthode de Taguchi** a permis d'étudier 4 facteurs à 3 niveaux avec 9 essais au lieu de 81, en mesurant la robustesse par le rapport S/N. Son enseignement principal ici : sur WDBC, les hyperparamètres testés ont **peu d'influence** (toutes les configurations donnent 96-97 % en validation croisée, effets non significatifs à l'ANOVA). L'activation est le facteur le plus influent (la sigmoïde est la plus robuste, relu la moins robuste) ; une régularisation forte (alpha = 0,1) et 20 neurones sont légèrement favorables.
# 2. **L'optimum additif prédit n'a pas été confirmé** (S/N observé < S/N prédit), signe que les effets sont dans le bruit ou que des interactions existent ; nous avons retenu le meilleur essai observé, comme le prévoit la démarche Taguchi.
# 3. **Partitions :** 60/20/20 obtient le meilleur S/N dans notre expérience, 70/15/15 (choix de l'article) est très proche ; l'écart est dans le bruit.
# 4. **Comparaison équitable** (mêmes 30 partitions 85/15) : l'ANN-Taguchi obtient la meilleure accuracy moyenne (97,9 %) et la plus faible dispersion, devant la régression logistique (97,8 %) et les SVM (97,1-97,5 %). Les écarts entre les meilleurs modèles restent inférieurs à un écart-type.
# 5. **Écart avec l'article (~1 point) :** dans l'article, le nombre de neurones et la partition sont choisis sur des partitions proches de celles de l'évaluation, ce qui peut gonfler le résultat, et le « filtrage du bruit » appliqué aux données n'est pas décrit. Ici, l'ensemble de test n'est jamais utilisé pour le réglage.
# 6. **Point d'attention médical :** la sensibilité (95 %) est plus faible que la spécificité (99 %) : le modèle rate plus de tumeurs malignes qu'il ne produit de fausses alertes. Abaisser le seuil de décision (par ex. 0,3 au lieu de 0,5) augmenterait la sensibilité.
# 
# **Pistes d'amélioration :** plan L18/L27 pour estimer les interactions, sélection de variables (forte colinéarité radius/perimeter/area), ajustement du seuil de décision, ensemble de plusieurs ANN.
