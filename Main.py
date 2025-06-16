import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.optimizers import Adam
import warnings
import random
import pandas as pd

warnings.filterwarnings('ignore')


class OptimizedCIFAR10Classifier:
    def __init__(self, data_path, train_split=0.8):
        self.data_path = data_path
        self.train_split = train_split
        self.img_size = (96, 96)  # Kompromis między jakością a szybkością
        self.batch_size = 32
        self.num_classes = 10
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
        self.setup_data_generators()

    def setup_data_generators(self, limit_samples=0.1):
        """Przygotowanie generatorów danych z opcją ograniczenia liczby próbek"""
        # Tworzenie listy wszystkich obrazów i etykiet
        image_paths = []
        labels = []

        for class_name in sorted(os.listdir(self.data_path)):
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_path, fname))
                    labels.append(class_name)

        # Tworzymy DataFrame
        df = pd.DataFrame({'filename': image_paths, 'class': labels})

        # Ograniczenie rozmiaru danych, jeśli limit_samples < 1.0
        if 0 < limit_samples < 1.0:
            df = df.sample(frac=limit_samples, random_state=42).reset_index(drop=True)

        # Dzielimy dane na treningowe i walidacyjne
        from sklearn.model_selection import train_test_split
        df_train, df_val = train_test_split(df, stratify=df['class'], train_size=self.train_split, random_state=42)

        # Konfiguracja generatorów
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = train_datagen.flow_from_dataframe(
            df_train,
            x_col='filename',
            y_col='class',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        self.test_generator = test_datagen.flow_from_dataframe(
            df_val,
            x_col='filename',
            y_col='class',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        print(f"Train: {len(df_train)}, Test: {len(df_val)} (limit: {limit_samples * 100:.0f}%)")

    def create_resnet_model(self):
        """Zoptymalizowany ResNet50"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(96, 96, 3)
        )

        # Odblokowanie ostatnich warstw dla lepszego uczenia
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        for layer in base_model.layers[-30:]:
            layer.trainable = True

        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def create_vgg_model(self):
        """Zoptymalizowany VGG16"""
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(96, 96, 3)
        )

        # Odblokowanie ostatnich bloków
        for layer in base_model.layers[:-6]:
            layer.trainable = False
        for layer in base_model.layers[-6:]:
            layer.trainable = True

        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def create_mobilenet_model(self):
        """Zoptymalizowany MobileNetV2"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(96, 96, 3),
            alpha=1.0  # Pełna wersja dla lepszej jakości
        )

        # MobileNet jest mniejszy, więc możemy odblokować więcej warstw
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        for layer in base_model.layers[-20:]:
            layer.trainable = True

        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_model(self, model, model_name, epochs=15):
        """Trenowanie z optymalnymi callbackami"""
        print(f"\n{'=' * 40}")
        print(f"Trenowanie: {model_name}")
        print(f"{'=' * 40}")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=0.00001,
                verbose=1
            )
        ]

        history = model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.test_generator,
            callbacks=callbacks,
            verbose=1
        )

        return model, history

    def evaluate_model(self, model, model_name):
        """Ocena modelu z wszystkimi metrykami"""
        print(f"\nOcena modelu: {model_name}")

        # Predykcje
        y_pred_proba = model.predict(self.test_generator, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = self.test_generator.classes

        # Metryki
        test_loss, test_accuracy = model.evaluate(self.test_generator, verbose=0)
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {report['macro avg']['precision']:.4f}")
        print(f"Recall: {report['macro avg']['recall']:.4f}")
        print(f"F1-Score: {report['macro avg']['f1-score']:.4f}")

        return {
            'model_name': model_name,
            'accuracy': test_accuracy,
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1_score': report['macro avg']['f1-score'],
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

    def plot_confusion_matrix(self, cm, model_name):
        """Wizualizacja macierzy pomyłek"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(f'Macierz pomyłek - {model_name}')
        plt.xlabel('Predykcja')
        plt.ylabel('Rzeczywista wartość')
        plt.tight_layout()
        plt.show()

    def plot_roc_curves(self, results_list):
        """Krzywe ROC dla wszystkich modeli"""
        plt.figure(figsize=(10, 8))

        for result in results_list:
            # Binaryzacja dla multi-class ROC
            y_true_bin = label_binarize(result['y_true'], classes=range(self.num_classes))
            y_pred_proba = result['y_pred_proba']

            # Micro-average ROC
            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)

            plt.plot(fpr_micro, tpr_micro,
                     label=f'{result["model_name"]} (AUC = {roc_auc_micro:.3f})',
                     linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Krzywe ROC - Porównanie modeli')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def compare_models(self, results_list):
        """Porównanie wyników modeli"""
        print(f"\n{'=' * 60}")
        print("PORÓWNANIE MODELI")
        print(f"{'=' * 60}")

        print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 60)

        for result in results_list:
            print(f"{result['model_name']:<15} "
                  f"{result['accuracy']:<10.4f} "
                  f"{result['precision']:<10.4f} "
                  f"{result['recall']:<10.4f} "
                  f"{result['f1_score']:<10.4f}")

        best_model = max(results_list, key=lambda x: x['accuracy'])
        print(f"\n🏆 Najlepszy model: {best_model['model_name']} "
              f"(Accuracy: {best_model['accuracy']:.4f})")

    def run_analysis(self, epochs=15):
        """Główna analiza wszystkich modeli"""
        models = {
            'ResNet50': self.create_resnet_model,
            'VGG16': self.create_vgg_model,
            'MobileNetV2': self.create_mobilenet_model
        }

        results = []

        for model_name, model_func in models.items():
            print(f"\n{'#' * 50}")
            print(f"ANALIZA: {model_name}")
            print(f"{'#' * 50}")

            # Tworzenie i trenowanie
            model = model_func()
            trained_model, _ = self.train_model(model, model_name, epochs)

            # Ocena
            result = self.evaluate_model(trained_model, model_name)
            results.append(result)

            # Macierz pomyłek
            self.plot_confusion_matrix(result['confusion_matrix'], model_name)

            # Czyszczenie pamięci
            del model, trained_model
            tf.keras.backend.clear_session()

        # Porównanie i krzywe ROC
        self.compare_models(results)
        self.plot_roc_curves(results)

        return results


def main():
    """Główna funkcja programu"""
    print("=" * 50)
    print("KLASYFIKATOR CIFAR-10 (ResNet/VGG/MobileNet)")
    print("=" * 50)

    # Pobieranie parametrów
    data_path = "archive/cifar10/data"
    if not os.path.exists(data_path):
        print(f"❌ Błąd: Nie znaleziono {data_path}")
        return

    while True:
        try:
            train_percent = float(input("Procent danych treningowych (np. 80): "))
            if 0 < train_percent < 100:
                train_split = train_percent / 100
                break
            else:
                print("Podaj wartość między 0 a 100!")
        except ValueError:
            print("Podaj poprawną liczbę!")

    print("\n💡 Zalecenia:")
    print("• 10-15 epok dla dobrych wyników")
    print("• Minimum 8 epok aby uniknąć klasyfikacji wszystkiego jako jedna klasa")

    while True:
        try:
            epochs = int(input("Liczba epok (zalecane 12): "))
            if epochs > 0:
                break
            else:
                print("Liczba epok musi być > 0!")
        except ValueError:
            print("Podaj poprawną liczbę!")

    # Uruchomienie analizy
    print(f"\n🚀 Start analizy: {train_percent:.0f}% train, {epochs} epok")

    classifier = OptimizedCIFAR10Classifier(data_path, train_split)
    results = classifier.run_analysis(epochs)

    # Zapisanie wyników
    with open('wyniki_klasyfikacji.txt', 'w', encoding='utf-8') as f:
        f.write("WYNIKI KLASYFIKACJI CIFAR-10\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Podział: {train_percent:.0f}% train / {100 - train_percent:.0f}% test\n")
        f.write(f"Epoki: {epochs}\n\n")

        for result in results:
            f.write(f"{result['model_name']}:\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall: {result['recall']:.4f}\n")
            f.write(f"  F1-Score: {result['f1_score']:.4f}\n\n")

    print("\n✅ Analiza zakończona!")
    print("📄 Wyniki zapisane w 'wyniki_klasyfikacji.txt'")


if __name__ == "__main__":
    main()