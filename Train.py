import sys
import os
import json
import zipfile
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  load_img, img_to_array,
                                                  save_img)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff',
    '.tif', '.JPG', '.JPEG', '.PNG'
}


def collect_all_images(src_dir):
    dataset = {}
    for name in sorted(os.listdir(src_dir)):
        sub = os.path.join(src_dir, name)
        if not os.path.isdir(sub):
            continue
        imgs = []
        for f in sorted(os.listdir(sub)):
            if os.path.splitext(f)[1] in IMAGE_EXTENSIONS:
                imgs.append(os.path.join(sub, f))
        if imgs:
            dataset[name] = imgs
    return dataset


def split_dataset(src_dir, split_dir, test_ratio=0.2, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    dataset = collect_all_images(src_dir)
    if not dataset:
        print(f"Error: no images found in '{src_dir}'")
        exit()

    train_dir = os.path.join(split_dir, 'train')
    test_dir = os.path.join(split_dir, 'test')

    split_info = {
        'src_dir': src_dir,
        'split_dir': split_dir,
        'test_ratio': test_ratio,
        'seed': seed,
        'classes': {},
        'total_train': 0,
        'total_test': 0,
    }

    print(f"\n[SPLIT] Ratio test: {int(test_ratio*100)}% | "
          f"Ratio train: {int((1-test_ratio)*100)}%")
    print(f"  Source      : {src_dir}")
    print(f"  Destination : {split_dir}")
    print(f"  {'Classe':<35} {'Total':>6} {'Train':>6} {'Test':>6}")
    print(f"  {'-'*55}")

    for cls_name, img_paths in dataset.items():
        paths = img_paths.copy()
        random.shuffle(paths)

        n_test = max(1, int(len(paths) * test_ratio))
        n_train = len(paths) - n_test

        test_paths = paths[:n_test]
        train_paths = paths[n_test:]

        cls_train_dir = os.path.join(train_dir, cls_name)
        cls_test_dir = os.path.join(test_dir, cls_name)
        os.makedirs(cls_train_dir, exist_ok=True)
        os.makedirs(cls_test_dir, exist_ok=True)

        for p in train_paths:
            shutil.copy2(p, os.path.join(cls_train_dir, os.path.basename(p)))
        for p in test_paths:
            shutil.copy2(p, os.path.join(cls_test_dir, os.path.basename(p)))

        split_info['classes'][cls_name] = {
            'total': len(paths),
            'train': n_train,
            'test': n_test,
            'train_files': [os.path.basename(p) for p in train_paths],
            'test_files': [os.path.basename(p) for p in test_paths],
        }
        split_info['total_train'] += n_train
        split_info['total_test'] += n_test

        print(f"  {cls_name:<35} {len(paths):>6} {n_train:>6} {n_test:>6}")

    total = split_info['total_train'] + split_info['total_test']
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<35} {total:>6} "
          f"{split_info['total_train']:>6} {split_info['total_test']:>6}")

    info_path = os.path.join(split_dir, 'split_info.json')
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"\n  Split info save : {info_path}")

    return split_info, train_dir, test_dir


def discover_classes(src_dir):
    classes = []
    for name in sorted(os.listdir(src_dir)):
        sub = os.path.join(src_dir, name)
        if not os.path.isdir(sub):
            continue
        imgs = []
        for f in os.listdir(sub):
            if os.path.splitext(f)[1] in IMAGE_EXTENSIONS:
                imgs.append(f)
        if imgs:
            classes.append(name)
    return classes


def build_gen(src_dir, img_size, batch_size, val_split):
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

    train_gen = train_aug.flow_from_directory(subset='training', **kwargs)
    val_gen = val_aug.flow_from_directory(subset='validation', **kwargs)

    return train_gen, val_gen


def save_augmented_samples(src_dir, output_dir, img_size, samples_per_image=5,
                           max_images_per_class=10, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    aug = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect',
    )

    dataset = collect_all_images(src_dir)
    if not dataset:
        print(f"  Warning : no image found in '{src_dir}'")
        return

    total_saved = 0

    print(f"\n[AUG] Save augmented images in : {output_dir}")
    print(f"  {samples_per_image} variants per sources, "
          f"max {max_images_per_class} images/classes")
    print(f"  {'Class':<35} {'Sources':>7} {'Generated':>9}")
    print(f"  {'-'*53}")

    for cls_name, img_paths in dataset.items():
        cls_out = os.path.join(output_dir, cls_name)
        os.makedirs(cls_out, exist_ok=True)

        selected = random.sample(img_paths, min(max_images_per_class,
                                                len(img_paths)))
        n_generated = 0

        for img_path in selected:
            img = load_img(img_path, target_size=(img_size, img_size))
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)

            base_name = os.path.splitext(os.path.basename(img_path))[0]

            # Génération des variantes
            gen = aug.flow(
                arr,
                batch_size=1,
                seed=seed,
            )
            for i in range(samples_per_image):
                aug_arr = next(gen)[0]
                out_path = os.path.join(
                    cls_out, f"aug_{base_name}_{i+1:02d}.jpg"
                )
                save_img(out_path, aug_arr)
                n_generated += 1

        total_saved += n_generated
        print(f"  {cls_name:<35} {len(selected):>7} {n_generated:>9}")

    print(f"  {'-'*53}")
    print(f"  {'TOTAL':<35} {'-':>7} {total_saved:>9}")
    print(f"  Directory : {output_dir}\n")


def evaluate_on_test(model, test_dir, img_size, batch_size):
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


def build_model(num_classes, img_size):
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
    )
    for layer in base.layers[:-30]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def run_training(src_dir, epochs, batch_size, img_size, val_split,
                 output_zip, test_dir=None, split_info=None,
                 aug_samples_per_image=5, aug_max_per_class=10,
                 save_aug=True):

    print(f"\n{'='*55}")
    print("  Leaf Disease Classifier - Training")
    print(f"{'='*55}")

    classes = discover_classes(src_dir)
    if not classes:
        print(f"Error: no image subdirectories in '{src_dir}'")
        exit()

    print(f"  Classes ({len(classes)}): {', '.join(classes)}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | ImgSize: {img_size}px")
    print(f"  Validating split (during training): {int(val_split*100)}%")
    if test_dir:
        print(f"  Test directory : {test_dir}")

    aug_dir = None
    if save_aug:
        aug_dir = os.path.join(os.path.dirname(output_zip) or '.',
                               'augmented_samples')
        save_augmented_samples(
            src_dir=src_dir,
            output_dir=aug_dir,
            img_size=img_size,
            samples_per_image=aug_samples_per_image,
            max_images_per_class=aug_max_per_class,
        )

    print("\n[1/4] Generator construction...")
    train_gen, val_gen = build_gen(src_dir, img_size, batch_size, val_split)
    print(
        f"  Train samples : {train_gen.samples} "
        f"| Val samples : {val_gen.samples}"
    )

    if val_gen.samples < 100:
        print(
            f"  Warning : {val_gen.samples} images validating "
            f"(minimum recommended : 100)."
        )

    print("\n[2/4] Model construction (MobileNetV2)...")
    model = build_model(len(classes), img_size)

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', patience=5,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.3,
            patience=3, min_lr=1e-7, verbose=1
        ),
    ]

    print(f"\n[3/4] Training ({epochs} epochs max)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_acc = max(history.history.get('val_accuracy', [0]))
    final_acc = history.history['accuracy'][-1]
    print(f"\n  Train accuracy (finale)    : {final_acc*100:.2f}%")
    print(f"  Val   accuracy (meilleure) : {best_val_acc*100:.2f}%")

    test_acc = None
    if test_dir and os.path.isdir(test_dir):
        print(f"\n  Evaluating on test set ({test_dir})...")
        _, test_acc = evaluate_on_test(model, test_dir, img_size, batch_size)
        if test_acc is not None:
            print(f"  Test accuracy : {test_acc*100:.2f}%")
            if test_acc < 0.90:
                print("  Warning : test accuracy < 90%.")
            else:
                print("  OK : Test accuracy >= 90%")

    if best_val_acc >= 0.90:
        print("  OK : Val accuracy >= 90%")

    print(f"\n[4/4] Saving model to '{output_zip}'...")
    tmp_dir = '/tmp/leaf_model_export'
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    model_path = os.path.join(tmp_dir, 'model.h5')
    classes_path = os.path.join(tmp_dir, 'classes.txt')
    metrics_path = os.path.join(tmp_dir, 'metrics.json')
    split_info_path = os.path.join(tmp_dir, 'split_info.json')

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
        'history': {
            k: [round(v, 4) for v in vals]
            for k, vals in history.history.items()
        },
        'augmented_samples': {
            'saved': save_aug,
            'directory': aug_dir,
            'samples_per_image': aug_samples_per_image if save_aug else 0,
            'max_images_per_class': aug_max_per_class if save_aug else 0,
        },
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    if split_info:
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=2)

    if os.path.exists(output_zip):
        os.remove(output_zip)

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, 'model.h5')
        zf.write(classes_path, 'classes.txt')
        zf.write(metrics_path, 'metrics.json')
        if split_info:
            with open(split_info_path, 'w') as f:
                json.dump(split_info, f, indent=2)
            zf.write(split_info_path, 'split_info.json')

        if save_aug and aug_dir and os.path.isdir(aug_dir):
            print("  Image added to zip...")
            n_aug_zipped = 0
            abs_output_zip = os.path.abspath(output_zip)
            for cls_name in sorted(os.listdir(aug_dir)):
                cls_path = os.path.join(aug_dir, cls_name)
                if not os.path.isdir(cls_path):
                    continue
                for fname in sorted(os.listdir(cls_path)):
                    fpath = os.path.join(cls_path, fname)
                    if not os.path.isfile(fpath):
                        continue
                    if os.path.abspath(fpath) == abs_output_zip:
                        continue
                    arcname = os.path.join(
                        'augmented_samples', cls_name, fname
                    )
                    zf.write(fpath, arcname)
                    n_aug_zipped += 1
                print(f"  {n_aug_zipped} augmented images added to zip.")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    zip_mb = os.path.getsize(output_zip) / 1e6
    print(f"  Save : {output_zip} ({zip_mb:.1f} MB)")
    print(f"\n{'='*55}")
    print("  Finished. Use Predict.py to classify.")
    print(f"{'='*55}\n")


def parse_args(args):
    src = None
    epochs = 15
    batch_size = 32
    img_size = 224
    val_split = 0.2
    output = 'model.zip'
    split = None
    split_dir = './split_data/'
    split_only = False
    seed = 42

    i = 1
    while i < len(args):
        if args[i] == '--epochs':
            epochs = int(args[i + 1])
            i += 2
        elif args[i] == '--batch-size':
            batch_size = int(args[i + 1])
            i += 2
        elif args[i] == '--img-size':
            img_size = int(args[i + 1])
            i += 2
        elif args[i] == '--val-split':
            val_split = float(args[i + 1])
            i += 2
        elif args[i] == '--output':
            output = args[i + 1]
            i += 2
        elif args[i] == '--split':
            split = float(args[i + 1])
            i += 2
        elif args[i] == '--split-dir':
            split_dir = args[i + 1]
            i += 2
        elif args[i] == '--split-only':
            split_only = True
            i += 1
        elif args[i] == '--seed':
            seed = int(args[i + 1])
            i += 2
        else:
            src = args[i]
            i += 1

    return src, epochs, batch_size, img_size, val_split, output, \
        split, split_dir, split_only, seed


def main(args):
    if len(args) < 2:
        print("Usage: python3 Train.py <src> [options]")
        print("Use -h for help.")
        exit()

    src, epochs, batch_size, img_size, val_split, output, \
        split, split_dir, split_only, seed = parse_args(args)

    if not src or not os.path.isdir(src):
        print(f"Error: '{src}' not a directory.")
        exit()

    train_dir = src
    test_dir = None
    split_info = None

    if split is not None:
        if not (0.0 < split < 1.0):
            print("Error: --split must be between 0.0 and 1.0")
            exit()
        split_info, train_dir, test_dir = split_dataset(
            src_dir=src,
            split_dir=split_dir,
            test_ratio=split,
            seed=seed,
        )
        if split_only:
            print("\nSplit finished (--split-only active, no training).")
            print(f"  Train : {train_dir}")
            print(f"  Test  : {test_dir}")
            exit()

    run_training(
        src_dir=train_dir,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        val_split=val_split,
        output_zip=output,
        test_dir=test_dir,
        split_info=split_info,
    )


if __name__ == '__main__':
    main(sys.argv)
