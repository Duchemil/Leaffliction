#!/usr/bin/env python3
"""
Leaf Disease Classifier - Training
------------------------------------
Usage:
    python train.py ./Apple/
    python train.py ./Apple/ --epochs 20 --batch-size 32 --output model.zip

Arguments:
    src              Directory with one sub-folder per disease class
    --epochs         Number of training epochs (default: 15)
    --batch-size     Batch size (default: 32)
    --img-size       Input image size (default: 224)
    --val-split      Validation split ratio (default: 0.2)
    --output         Output zip filename (default: model.zip)
    -h               Show this help message

The zip archive contains:
    - model.h5         Trained Keras model
    - classes.txt      Class names (one per line, index = class id)
    - metrics.json     Training accuracy / val_accuracy summary
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
#  Data helpers
# ─────────────────────────────────────────────

def discover_classes(src_dir):
    """Return sorted list of class names (= sub-directory names with images)."""
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
    """Build train & validation ImageDataGenerators with augmentation."""

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


# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────

def build_model(num_classes, img_size):
    """MobileNetV2 backbone + custom classification head."""
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
    )
    # Freeze all but the last 30 layers for fine-tuning
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

def train(src_dir, epochs, batch_size, img_size, val_split, output_zip):
    print(f"\n{'='*55}")
    print(f"  Leaf Disease Classifier - Training")
    print(f"{'='*55}")

    classes = discover_classes(src_dir)
    if not classes:
        print(f"Error: no image sub-directories found in '{src_dir}'")
        sys.exit(1)

    print(f"  Classes ({len(classes)}): {', '.join(classes)}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | ImgSize: {img_size}px")
    print(f"  Validation split: {int(val_split*100)}%")

    # Generators
    print("\n[1/4] Building data generators...")
    train_gen, val_gen = build_generators(src_dir, img_size, batch_size, val_split)
    print(f"  Train samples: {train_gen.samples} | Val samples: {val_gen.samples}")

    if val_gen.samples < 100:
        print(f"  Warning: validation set has only {val_gen.samples} images "
              f"(minimum recommended: 100). Add more images for reliable evaluation.")

    # Model
    print("\n[2/4] Building model (MobileNetV2 + fine-tuning)...")
    model = build_model(len(classes), img_size)
    model.summary(print_fn=lambda x: None)  # silent summary

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

    # Training
    print(f"\n[3/4] Training for up to {epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Metrics summary
    best_val_acc = max(history.history.get('val_accuracy', [0]))
    final_acc    = history.history['accuracy'][-1]
    print(f"\n  Final train accuracy : {final_acc*100:.2f}%")
    print(f"  Best val accuracy    : {best_val_acc*100:.2f}%")

    if best_val_acc < 0.90:
        print("  Note: val accuracy < 90%. Consider more data or more epochs.")
    else:
        print("  ✓ Val accuracy ≥ 90%")

    # Save artifacts to temp dir then zip
    print(f"\n[4/4] Saving model to '{output_zip}'...")
    tmp_dir = '/tmp/leaf_model_export'
    os.makedirs(tmp_dir, exist_ok=True)

    model_path   = os.path.join(tmp_dir, 'model.h5')
    classes_path = os.path.join(tmp_dir, 'classes.txt')
    metrics_path = os.path.join(tmp_dir, 'metrics.json')

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
        'history': {k: [round(v, 4) for v in vals]
                    for k, vals in history.history.items()},
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path,   'model.h5')
        zf.write(classes_path, 'classes.txt')
        zf.write(metrics_path, 'metrics.json')

    shutil.rmtree(tmp_dir, ignore_errors=True)

    zip_mb = os.path.getsize(output_zip) / 1e6
    print(f"  Saved: {output_zip} ({zip_mb:.1f} MB)")
    print(f"\n{'='*55}")
    print(f"  Done. Run predict.py to classify new images.")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Train a leaf disease classifier (MobileNetV2 transfer learning)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py ./Apple/
  python train.py ./Apple/ --epochs 20 --output apple_model.zip
  python train.py ./Leaves/ --epochs 30 --batch-size 16 --val-split 0.2
        """
    )
    parser.add_argument('src',
                        help='Directory with one sub-folder per disease class')
    parser.add_argument('--epochs',      type=int,   default=15,
                        help='Number of training epochs (default: 15)')
    parser.add_argument('--batch-size',  type=int,   default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--img-size',    type=int,   default=224,
                        help='Input image size in pixels (default: 224)')
    parser.add_argument('--val-split',   type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--output',      type=str,   default='model.zip',
                        help='Output zip filename (default: model.zip)')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.src):
        print(f"Error: '{args.src}' is not a directory.")
        sys.exit(1)

    train(
        src_dir    = args.src,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        img_size   = args.img_size,
        val_split  = args.val_split,
        output_zip = args.output,
    )


if __name__ == '__main__':
    main()