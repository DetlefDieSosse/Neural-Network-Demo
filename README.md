# Neural-Network-Demo

In diesem Repo zeigen wir euch ein mit [Python](https://www.python.org/) und [TensorFlow](https://www.tensorflow.org/) erstelltes künstliches neuronales Netz zur Klassifikation von handgeschriebenen Zahlen.

Zum Einstieg empfehlen wir [TensorFlow Playground](https://playground.tensorflow.org/) um die Struktur von neuronalen Netzen besser zu verstehen.

## Einrichtung

Achtung: Nutzt Python 3.12, da 3.13 noch nicht mit tensorflow funktioniert.

1. **Repository klonen**:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Virtuelle Umgebung erstellen**:
    ```sh
    python -m venv venv
    ```

3. **Virtuelle Umgebung aktivieren**:
    - Unter Windows:
        ```sh
        .\venv\Scripts\activate
        ```
    - Unter macOS und Linux:
        ```sh
        source venv/bin/activate
        ```

4. **Benötigte Pakete installieren**:
    ```sh
    pip install -r requirements.txt
    ```

## Modell trainieren

Um das neuronale Netzwerk zu trainieren, führe folgenden Befehl aus:
```sh
python src/neuralNetwork/train_model.py
```
Dies trainiert das Modell mit dem MNIST-Datensatz und speichert das trainierte Modell in `mnist_model.keras`.

## Bild vorhersagen

Um die Ziffer in einem Bild vorherzusagen, platziere das Bild im Verzeichnis `src/image/` und benenne es `img.png`. Führe dann folgenden Befehl aus:
```sh
python src/neuralNetwork/predict_image.py
```
Dies lädt das Bild, verarbeitet es vor und verwendet das trainierte Modell, um die Ziffer vorherzusagen.

## Beispielbild anzeigen

Um ein Beispielbild aus dem MNIST-Datensatz anzuzeigen, führe folgenden Befehl aus:
```sh
python src/neuralNetwork/display_sample_image.py
```
Dies zeigt ein zufälliges Beispielbild aus dem Trainingsdatensatz zusammen mit seinem Label an.

## Fehlerbehebung

- **Importfehler**: Stelle sicher, dass die virtuelle Umgebung aktiviert ist und alle benötigten Pakete installiert sind.
- **Modellgenauigkeit**: Wenn das Modell nicht genau genug ist, versuche die Hyperparameter zu optimieren oder die Modellarchitektur in `src/neuralNetwork/train_model.py` zu ändern.

Für weitere Unterstützung siehe die [TensorFlow-Dokumentation](https://www.tensorflow.org/).