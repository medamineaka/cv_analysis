import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def entrainer_nn(dataset_path, model_path, vect_path, enc_path, nom=""):
    print(f"\n🚀 Entraînement modèle {nom}...")

    # ── 1. Chargement
    df = pd.read_csv(dataset_path)[["Text_Block", "Label"]].dropna()
    df = df[df["Text_Block"].str.strip() != ""]
    print(f"\n📊 Distribution labels {nom}:")
    print(df["Label"].value_counts())

    # ── 2. Split
    train, test = train_test_split(
        df, test_size=0.10, random_state=42, stratify=df["Label"]
    )

    # ── 3. TF-IDF → sparse puis sort + toarray par chunks
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        dtype=np.float32
    )
    X_train_sparse = vectorizer.fit_transform(train["Text_Block"])
    X_test_sparse  = vectorizer.transform(test["Text_Block"])

    # ── FIX ERREUR : trier les indices sparse avant conversion
    X_train_sparse.sort_indices()
    X_test_sparse.sort_indices()

    # Conversion dense par chunks → évite 22GB d'un coup
    print("⏳ Conversion dense par chunks...")
    CHUNK = 10000

    def sparse_to_dense_chunks(X_sparse):
        parts = []
        for i in range(0, X_sparse.shape[0], CHUNK):
            chunk = X_sparse[i:i+CHUNK].toarray().astype(np.float32)
            parts.append(chunk)
            print(f"   chunk {i//CHUNK + 1}/{int(np.ceil(X_sparse.shape[0]/CHUNK))}", end="\r")
        return np.vstack(parts)

    X_train = sparse_to_dense_chunks(X_train_sparse)
    X_test  = sparse_to_dense_chunks(X_test_sparse)
    print(f"\n✅ X_train shape: {X_train.shape}")

    # ── 4. Labels
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["Label"])
    y_test  = encoder.transform(test["Label"])
    n_classes   = len(encoder.classes_)
    y_train_cat = to_categorical(y_train, n_classes)
    y_test_cat  = to_categorical(y_test,  n_classes)

    # ── 5. Poids classes
    weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(weights))
    print(f"\n⚖️  Poids: {dict(zip(encoder.classes_, weights.round(2)))}")

    # ── 6. Architecture (Input layer explicite → fix UserWarning)
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(n_classes, activation="softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    model.summary()

    # ── 7. Callbacks
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=3,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=2, verbose=1)
    ]

    # ── 8. Entraînement
    model.fit(
        X_train, y_train_cat,
        epochs=20,
        batch_size=64,
        validation_split=0.1,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # ── 9. Évaluation
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"\n✅ Accuracy {nom}: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # ── 10. Sauvegarde
    model.save(model_path)
    joblib.dump(vectorizer, vect_path)
    joblib.dump(encoder,    enc_path)
    print(f"💾 Modèle {nom} sauvegardé !")
    return acc


# ── CV
entrainer_nn(
    dataset_path="../../data/dataset_cv.csv",
    model_path="../../models/nn_cv.h5",
    vect_path="../../models/vectorizer_cv.pkl",
    enc_path="../../models/encoder_cv.pkl",
    nom="CV"
)

# ── Offres
entrainer_nn(
    dataset_path="../../data/dataset_offre.csv",
    model_path="../../models/nn_offre.h5",
    vect_path="../../models/vectorizer_offre.pkl",
    enc_path="../../models/encoder_offre.pkl",
    nom="Offres"
)