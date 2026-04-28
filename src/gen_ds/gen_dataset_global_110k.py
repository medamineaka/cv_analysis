
import pandas as pd

# Charger les deux datasets
df_maroc = pd.read_csv("../../data/referentiels/dataset_marocain.csv")
df_inter = pd.read_csv("../../data/referentiels/dataset_international.csv")

# Ajouter la colonne Language = "FR" pour tout le dataset marocain
df_maroc["Language"] = "FR"

# Fusionner les deux datasets
df_all = pd.concat([df_maroc, df_inter], ignore_index=True)

# Sauvegarde du dataset fusionné
df_all.to_csv("../../data/dataset_global_110k.csv", index=False, encoding="utf-8")

print("✅ Fusion terminée : dataset_global_110k.csv contient", len(df_all), "lignes")
