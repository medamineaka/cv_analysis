
import pandas as pd
import random

# Charger ton dataset existant (50k déjà généré)
df_old = pd.read_csv("../../data/referentiels/dataset_marocain.csv")

# Charger les référentiels
ecoles = pd.read_csv("../../data/referentiels/ecoles.csv")["nom"].tolist()
filieres = pd.read_csv("../../data/referentiels/filieres.csv")[["filiere", "diplome", "domaine"]].values.tolist()

# Générer toutes les nouvelles phrases EDUCATION
new_education = []
for filiere, diplome, domaine in filieres:
    ecole = random.choice(ecoles)
    new_education.append((f"{diplome} en {filiere}", "EDUCATION", domaine))
    new_education.append((f"Formation en {filiere}", "EDUCATION", domaine))
    new_education.append((f"{diplome} obtenu au Maroc", "EDUCATION", domaine))
    new_education.append((f"Études supérieures en {filiere}", "EDUCATION", domaine))
    new_education.append((f"{diplome} délivré par {ecole}", "EDUCATION", domaine))

# Sélectionner exactement 10k lignes
target_count = 10000
if len(new_education) >= target_count:
    new_education = random.sample(new_education, target_count)
else:
    new_education = random.choices(new_education, k=target_count)

# Convertir en DataFrame
df_new = pd.DataFrame(new_education, columns=["Text_Block", "Label", "Domain"])

# Fusionner avec l’ancien dataset (50k)
df_final = pd.concat([df_old, df_new], ignore_index=True)

# Sauvegarde
df_final.to_csv("../../data/dataset_marocain.csv", index=False, encoding="utf-8")

print("✅ Dataset enrichi généré : ../../data/dataset_marocain.csv (50k ancien + 10k nouvelles EDUCATION)")
