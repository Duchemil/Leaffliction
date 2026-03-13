#!/usr/bin/env python3
"""
Leaf Disease Classifier - Training
------------------------------------
Usage:
    python train.py ./Apple/
    python train.py ./Apple/ --epochs 20 --batch-size 32 --output model.zip

    # Split les donnees en train/test avant d'entrainer
    python train.py ./Apple/ --split 0.2 --split-dir ./split_data/
    python train.py ./Apple/ --split 0.2 --split-dir ./split_data/ --split-only

Arguments:
    src              Repertoire source avec un sous-dossier par classe
    --epochs         Nombre d'epochs (defaut: 15)
    --batch-size     Taille du batch (defaut: 32)
    --img-size       Taille des images en pixels (defaut: 224)
    --val-split      Ratio de validation pendant l'entrainement (defaut: 0.2)
    --output         Nom du zip de sortie (defaut: model.zip)
    --split          Ratio du split test (ex: 0.2 = 20% test, 80% train)
    --split-dir      Repertoire de destination du split (defaut: ./split_data/)
    --split-only     Faire uniquement le split sans entrainer
    --seed           Graine aleatoire pour la reproductibilite (defaut: 42)
    -h               Aide

Le zip contient:
    - model.h5         Modele Keras entraine
    - classes.txt      Noms des classes (une par ligne)
    - metrics.json     Accuracy train / val / test
    - split_info.json  Detail du split si --split est utilise
"""

import sys
import os
import argparse
import json
import zipfile
import shutil
import random
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                    '.JPG', '.JPEG', '.PNG'}


# ─────────────────────────────────────────────
#  Split helpers
# ─────────────────────────────────────────────

def collect_all_images(src_dir):
    """
    Collecte toutes les images par classe depuis src_dir.
    Retourne dict: {class_name: [image_path, ...]}
    """
    dataset = {}
    for name in sorted(os.listdir(src_dir)):
        sub = os.path.join(src_dir, name)
        if not os.path.isdir(sub):
            continue
        imgs = [
            os.path.join(sub, f) for f in sorted(os.listdir(sub))
            if os.path.splitext(f)[1] in IMAGE_EXTENSIONS
        ]
        if imgs:
            dataset[name] = imgs
    return dataset


def split_dataset(src_dir, split_dir, test_ratio=0.2, seed=42):
    """
    Copie les images de src_dir dans split_dir/train/ et split_dir/test/
    en respectant le ratio test_ratio par classe (stratifie).

    Retourne un dict avec les stats du split.
    """
    random.seed(seed)
    np.random.seed(seed)

    dataset = collect_all_images(src_dir)
    if not dataset:
        print(f"Error: aucune image trouvee dans '{src_dir}'")
        sys.exit(1)

    train_dir = os.path.join(split_dir, 'train')
    test_dir  = os.path.join(split_dir, 'test')

    split_info = {
        'src_dir':    src_dir,
        'split_dir':  split_dir,
        'test_ratio': test_ratio,
        'seed':       seed,
        'classes':    {},
        'total_train': 0,
        'total_test':  0,
    }

    print(f"\n[SPLIT] Ratio test: {int(test_ratio*100)}% | "
          f"Ratio train: {int((1-test_ratio)*100)}%")
    print(f"  Source      : {src_dir}")
    print(f"  Destination : {split_dir}")
    print(f"  {'Classe':<35} {'Total':>6} {'Train':>6} {'Test':>6}")
    print(f"  {'-'*55}")

    for cls_name, img_paths in dataset.items():
        # Melanger aleatoirement
        paths = img_paths.copy()
        random.shuffle(paths)

        n_test  = max(1, int(len(paths) * test_ratio))
        n_train = len(paths) - n_test

        test_paths  = paths[:n_test]
        train_paths = paths[n_test:]

        # Creer les dossiers
        cls_train_dir = os.path.join(train_dir, cls_name)
        cls_test_dir  = os.path.join(test_dir,  cls_name)
        os.makedirs(cls_train_dir, exist_ok=True)
        os.makedirs(cls_test_dir,  exist_ok=True)

        # Copier les fichiers
        for p in train_paths:
            shutil.copy2(p, os.path.join(cls_train_dir, os.path.basename(p)))
        for p in test_paths:
            shutil.copy2(p, os.path.join(cls_test_dir,  os.path.basename(p)))

        split_info['classes'][cls_name] = {
            'total': len(paths),
            'train': n_train,
            'test':  n_test,
            'train_files': [os.path.basename(p) for p in train_paths],
            'test_files':  [os.path.basename(p) for p in test_paths],
        }
        split_info['total_train'] += n_train
        split_info['total_test']  += n_test

        print(f"  {cls_name:<35} {len(paths):>6} {n_train:>6} {n_test:>6}")

    total = split_info['total_train'] + split_info['total_test']
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<35} {total:>6} "
          f"{split_info['total_train']:>6} {split_info['total_test']:>6}")

    # Sauvegarder le split_info.json dans split_dir
    info_path = os.path.join(split_dir, 'split_info.json')
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"\n  Split info sauvegarde : {info_path}")

    return split_info, train_dir, test_dir


# ─────────────────────────────────────────────
#  Data helpers
# ─────────────────────────────────────────────

def discover_classes(src_dir):
    """Retourne la liste triee des classes (sous-dossiers avec images)."""
    classes = []
    for name in sorted(os.listdir(src_dir)):
        sub = os.path.join(src_dir, name)
        if os.path.isdir(sub):
            imgs = [f for f in os.listdir(sub)
                    if os.path.splitext(f)[1] in IMAGE_EXTENSIONS]
            if imgs:
                classes.append(name)
    return classes


def build_generators(src_dir, img_size, batch_size, val_split):
    """Construit les generateurs train & validation avec augmentation."""

    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect',
    )

    val_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
    )

    kwargs = dict(
        directory=src_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42,
    )

    train_gen = train_aug.flow_from_directory(subset='training',   **kwargs)
    val_gen   = val_aug.flow_from_directory(  subset='validation', **kwargs)

    return train_gen, val_gen


def evaluate_on_test(model, test_dir, img_size, batch_size):
    """Evalue le modele sur le repertoire de test. Retourne (loss, accuracy)."""
    test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        directory=test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )
    if test_gen.samples == 0:
        return None, None
    loss, acc = model.evaluate(test_gen, verbose=1)
    return loss, acc


# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────

def build_model(num_classes, img_size):
    """MobileNetV2 backbone + tete de classification."""
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
    )
    for layer in base.layers[:-30]:
        layer.trainable = False

    inputs  = tf.keras.Input(shape=(img_size, img_size, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────

def train(src_dir, epochs, batch_size, img_size, val_split, output_zip,
          test_dir=None, split_info=None):

    print(f"\n{'='*55}")
    print(f"  Leaf Disease Classifier - Training")
    print(f"{'='*55}")

    classes = discover_classes(src_dir)
    if not classes:
        print(f"Error: aucun sous-dossier d'images dans '{src_dir}'")
        sys.exit(1)

    print(f"  Classes ({len(classes)}): {', '.join(classes)}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | ImgSize: {img_size}px")
    print(f"  Validation split (pendant training): {int(val_split*100)}%")
    if test_dir:
        print(f"  Repertoire test : {test_dir}")

    # Generators
    print("\n[1/4] Construction des generateurs...")
    train_gen, val_gen = build_generators(src_dir, img_size, batch_size, val_split)
    print(f"  Train samples : {train_gen.samples} | Val samples : {val_gen.samples}")

    if val_gen.samples < 100:
        print(f"  Avertissement : seulement {val_gen.samples} images en validation "
              f"(minimum recommande : 100).")

    # Model
    print("\n[2/4] Construction du modele (MobileNetV2)...")
    model = build_model(len(classes), img_size)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

    # Training
    print(f"\n[3/4] Entrainement ({epochs} epochs max)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_acc = max(history.history.get('val_accuracy', [0]))
    final_acc    = history.history['accuracy'][-1]
    print(f"\n  Train accuracy (finale) : {final_acc*100:.2f}%")
    print(f"  Val   accuracy (meilleure) : {best_val_acc*100:.2f}%")

    # Evaluation sur le jeu de test (si disponible)
    test_acc = None
    if test_dir and os.path.isdir(test_dir):
        print(f"\n  Evaluation sur le jeu de test ({test_dir})...")
        _, test_acc = evaluate_on_test(model, test_dir, img_size, batch_size)
        if test_acc is not None:
            print(f"  Test accuracy : {test_acc*100:.2f}%")
            if test_acc < 0.90:
                print("  Avertissement : test accuracy < 90%.")
            else:
                print("  OK : Test accuracy >= 90%")

    if best_val_acc >= 0.90:
        print("  OK : Val accuracy >= 90%")

    # Sauvegarde
    print(f"\n[4/4] Sauvegarde du modele dans '{output_zip}'...")
    tmp_dir = '/tmp/leaf_model_export'
    os.makedirs(tmp_dir, exist_ok=True)

    model_path        = os.path.join(tmp_dir, 'model.h5')
    classes_path      = os.path.join(tmp_dir, 'classes.txt')
    metrics_path      = os.path.join(tmp_dir, 'metrics.json')
    split_info_path   = os.path.join(tmp_dir, 'split_info.json')

    model.save(model_path)

    with open(classes_path, 'w') as f:
        f.write('\n'.join(classes))

    metrics = {
        'classes': classes,
        'num_classes': len(classes),
        'img_size': img_size,
        'epochs_trained': len(history.history['accuracy']),
        'final_train_accuracy': round(final_acc, 4),
        'best_val_accuracy': round(best_val_acc, 4),
        'test_accuracy': round(test_acc, 4) if test_acc is not None else None,
        'history': {k: [round(v, 4) for v in vals]
                    for k, vals in history.history.items()},
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path,   'model.h5')
        zf.write(classes_path, 'classes.txt')
        zf.write(metrics_path, 'metrics.json')
        if split_info:
            with open(split_info_path, 'w') as f:
                json.dump(split_info, f, indent=2)
            zf.write(split_info_path, 'split_info.json')

    shutil.rmtree(tmp_dir, ignore_errors=True)

    zip_mb = os.path.getsize(output_zip) / 1e6
    print(f"  Sauvegarde : {output_zip} ({zip_mb:.1f} MB)")
    print(f"\n{'='*55}")
    print(f"  Termine. Utilisez predict.py pour classifier.")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Entraine un classifieur de maladies foliaires (MobileNetV2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Entrainement simple
  python train.py ./Apple/

  # Avec split automatique train/test (20%% test)
  python train.py ./Apple/ --split 0.2 --split-dir ./split_data/

  # Faire uniquement le split (sans entrainer)
  python train.py ./Apple/ --split 0.2 --split-dir ./split_data/ --split-only

  # Split + entrainement + options avancees
  python train.py ./Apple/ --split 0.2 --split-dir ./split_data/ --epochs 20 --output model.zip

  # Si le split a deja ete fait, utiliser directement les dossiers
  python train.py ./split_data/train/ --epochs 20
        """
    )
    parser.add_argument('src',
                        help='Repertoire source (sous-dossiers = classes)')

    # Training options
    train_grp = parser.add_argument_group('Options d\'entrainement')
    train_grp.add_argument('--epochs',     type=int,   default=15,
                           help='Nombre d\'epochs (defaut: 15)')
    train_grp.add_argument('--batch-size', type=int,   default=32,
                           help='Taille du batch (defaut: 32)')
    train_grp.add_argument('--img-size',   type=int,   default=224,
                           help='Taille des images en pixels (defaut: 224)')
    train_grp.add_argument('--val-split',  type=float, default=0.2,
                           help='Ratio de validation pendant l\'entrainement (defaut: 0.2)')
    train_grp.add_argument('--output',     type=str,   default='model.zip',
                           help='Nom du fichier zip de sortie (defaut: model.zip)')

    # Split options
    split_grp = parser.add_argument_group('Options de split train/test')
    split_grp.add_argument('--split',      type=float, default=None,
                           metavar='RATIO',
                           help='Ratio du jeu de test, ex: 0.2 = 20%% test / 80%% train')
    split_grp.add_argument('--split-dir',  type=str,   default='./split_data/',
                           metavar='DIR',
                           help='Repertoire de destination du split (defaut: ./split_data/)')
    split_grp.add_argument('--split-only', action='store_true',
                           help='Effectuer uniquement le split, sans entrainer')
    split_grp.add_argument('--seed',       type=int,   default=42,
                           help='Graine aleatoire pour reproductibilite (defaut: 42)')

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.src):
        print(f"Error: '{args.src}' n'est pas un repertoire.")
        sys.exit(1)

    train_dir = args.src
    test_dir  = None
    split_info = None

    # ── Split ──
    if args.split is not None:
        if not (0.0 < args.split < 1.0):
            print("Error: --split doit etre entre 0.0 et 1.0 (ex: 0.2)")
            sys.exit(1)

        split_info, train_dir, test_dir = split_dataset(
            src_dir    = args.src,
            split_dir  = args.split_dir,
            test_ratio = args.split,
            seed       = args.seed,
        )

        if args.split_only:
            print("\nSplit termine (--split-only active, pas d'entrainement).")
            print(f"  Train : {train_dir}")
            print(f"  Test  : {test_dir}")
            sys.exit(0)

    # ── Training ──
    train(
        src_dir    = train_dir,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        img_size   = args.img_size,
        val_split  = args.val_split,
        output_zip = args.output,
        test_dir   = test_dir,
        split_info = split_info,
    )


if __name__ == '__main__':
    main()