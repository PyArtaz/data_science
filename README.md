# MVP - Image Classification

First MVP of a 2D CNN with custom architecture and a pretrained model that was adapted using transfer learning:

- Models are instantiated and trained in function train.py
- prediction accuracy is evaluated in predict.py  


## Installation:

1. Klone/Forke dieses Repository
2. Richte ein eigenes Repository auf github/gitlab ein. Darüber könnt ihr später die Abgaben eurer Modelle machen.
3. Python Virtual-Environment anlegen (z.B. mit Anaconda oder in PyCharm)
4. "requirements.txt" mit `pip install -r requirements.txt` ausführen und die notwendigen Pakete installieren
5. Speichere den nach labels separierten Bilder-Datensatz im Projekt-Subordner `/datasets`
6. Separiere den Datensatz in Training, Validation und Test-Daten mit der Funktion: `train_val_test_split.py`
7. Training des Modells: `cnn_training.py`
8. Evaluation des Modells: `cnn_evaluation.py`
