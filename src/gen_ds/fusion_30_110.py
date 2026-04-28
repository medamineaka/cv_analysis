import pandas as pd

# Charger les deux datasets
df_global = pd.read_csv("../../data/dataset_global_140k.csv")
df_noise = pd.read_csv("../../data/dataset_global_200k.csv")

# Vérifier colonnes (par ex. 'texte', 'label')
print(df_global.columns)
print(df_noise.columns)


# Fusionner
df_total = pd.concat([df_global, df_noise], ignore_index=True)

# Sauvegarder dataset fusionné
df_total.to_csv("../../data/dataset_global_340k.csv", index=False)

print("✅ Dataset fusionné avec succès :", df_total.shape)
print(df_total['Label'].value_counts())
