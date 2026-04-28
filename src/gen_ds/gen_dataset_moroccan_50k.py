import pandas as pd
import random
import os

# Charger les référentiels
ecoles = pd.read_csv("../../data/referentiels/ecoles.csv")["nom"].tolist()
entreprises = pd.read_csv("../../data/referentiels/entreprises.csv")["nom"].tolist()
filieres = pd.read_csv("../../data/referentiels/filieres.csv")[["filiere", "diplome", "domaine"]].values.tolist()
metiers = pd.read_csv("../../data/referentiels/metiers.csv")[["nom", "domaine"]].values.tolist()
villes = pd.read_csv("../../data/referentiels/villes.csv")["nom"].tolist()

dataset = []

# Fonction utilitaire pour équilibrer
def generate_samples(data, target_count):
    if len(data) >= target_count:
        return random.sample(data, target_count)
    else:
        return random.choices(data, k=target_count)

# EDUCATION (CV) - 5 variantes
education_data = []
for ecole in ecoles:
    for filiere, diplome, domaine in filieres:
        education_data.append((f"{diplome} en {filiere} à {ecole}", "EDUCATION", domaine))
        education_data.append((f"Études en {filiere} ({diplome}) à {ecole}", "EDUCATION", domaine))
        education_data.append((f"Diplômé de {ecole} en {filiere}", "EDUCATION", domaine))
        education_data.append((f"Formation en {filiere} obtenue à {ecole}", "EDUCATION", domaine))
        education_data.append((f"{ecole} délivre un {diplome} en {filiere}", "EDUCATION", domaine))
dataset.extend(generate_samples(education_data, 12500))

# EXPERIENCE (CV) - 5 variantes
experience_data = []
for entreprise in entreprises:
    for metier, domaine in metiers:
        ville = random.choice(villes)
        experience_data.append((f"{metier} chez {entreprise} à {ville}", "EXPERIENCE", domaine))
        experience_data.append((f"Expérience en tant que {metier} chez {entreprise}, basé à {ville}", "EXPERIENCE", domaine))
        experience_data.append((f"Travail comme {metier} pour {entreprise} à {ville}", "EXPERIENCE", domaine))
        experience_data.append((f"{entreprise} a employé un {metier} à {ville}", "EXPERIENCE", domaine))
        experience_data.append((f"Poste de {metier} occupé chez {entreprise}, situé à {ville}", "EXPERIENCE", domaine))
dataset.extend(generate_samples(experience_data, 12500))

# REQUIREMENT (Offre) - 5 variantes
requirement_data = []
for entreprise in entreprises:
    for metier, domaine in metiers:
        ville = random.choice(villes)
        requirement_data.append((f"Nous recherchons un {metier} pour {entreprise} à {ville}", "REQUIREMENT", domaine))
        requirement_data.append((f"{entreprise} recrute un {metier} basé à {ville}", "REQUIREMENT", domaine))
        requirement_data.append((f"Poste ouvert : {metier} chez {entreprise}, {ville}", "REQUIREMENT", domaine))
        requirement_data.append((f"{entreprise} souhaite embaucher un {metier} à {ville}", "REQUIREMENT", domaine))
        requirement_data.append((f"Offre d’emploi : {metier} pour {entreprise} à {ville}", "REQUIREMENT", domaine))
dataset.extend(generate_samples(requirement_data, 12500))

# LOCATION - 5 variantes
location_data = []
for ville in villes:
    location_data.append((f"Basé à {ville}, Maroc", "LOCATION", "Global"))
    location_data.append((f"Poste situé à {ville}", "LOCATION", "Global"))
    location_data.append((f"Résidence actuelle : {ville}", "LOCATION", "Global"))
    location_data.append((f"Localisation : {ville}, Maroc", "LOCATION", "Global"))
    location_data.append((f"Travail localisé à {ville}", "LOCATION", "Global"))
dataset.extend(generate_samples(location_data, 12500))

# Sauvegarde en CSV dans data/
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(dataset, columns=["Text_Block", "Label", "Domain"])
df.to_csv("../../data/dataset_marocain_50k.csv", index=False, encoding="utf-8")

print("✅ Dataset marocain équilibré généré et stocké dans data/dataset_marocain.csv avec 50 000 lignes !")
