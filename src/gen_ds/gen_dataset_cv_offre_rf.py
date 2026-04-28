import pandas as pd

# Charger le dataset global
df_all = pd.read_csv("../../data/dataset_global_340k.csv")

# Définir les labels pour CV et Offre
labels_cv = [
    "HEADER","EDUCATION","EXPERIENCE","SKILL","LOCATION","LANGUAGE",
    "INTEREST","CONTACT","PROJECT","ACHIEVEMENT"
]

labels_offre = [
    "HEADER","REQUIREMENT","RESPONSIBILITY","EDUCATION","EXPERIENCE",
    "LOCATION","CONTRACT","SALARY","LANGUAGE","BENEFIT","NOT_IMPORTANT"
]

# Séparer par type de labels
df_cv = df_all[df_all["Label"].isin(labels_cv)]
df_offre = df_all[df_all["Label"].isin(labels_offre)]

# Sauvegarde des sous-datasets
df_cv.to_csv("../../data/dataset_cv.csv", index=False, encoding="utf-8")
df_offre.to_csv("../../data/dataset_offre.csv", index=False, encoding="utf-8")

print("✅ Séparation terminée :")
print("   dataset_cv.csv =", len(df_cv), "lignes")
print("   dataset_offre.csv =", len(df_offre), "lignes")

# Rapport statistique : nombre de lignes par label et par langue
print("\n📊 Statistiques CV :")
print(df_cv.groupby(["Label","Language"]).size())

print("\n📊 Statistiques Offre :")
print(df_offre.groupby(["Label","Language"]).size())