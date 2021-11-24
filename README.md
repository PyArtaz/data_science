# MVP - Image Classification

First MVP of a 2D CNN with custom architecture and a pretrained model that was adapted using transfer learning:

- Models are instantiated and trained in function train.py
- prediction accuracy is evaluated in predict.py  


Additionally, repository contains two spelling correction scripts:

- pyspellchecker: german spelling correction for simgle words
- EasyMNT.py: Translation of sentences into 150+ languages

## Installation:

1. Klone/Forke dieses Repository
2. Richte ein eigenes Repository auf github/gitlab ein. Darüber könnt ihr später die Abgaben eurer Modelle machen.
3. Python Virtual-Environment anlegen (z.B. mit Anaconda oder in PyCharm)
4. "requirements.txt" mit `pip install -r requirements.txt` ausführen und die notwendigen Pakete installieren
5. Speichere den nach labels separierten Bilder-Datensatz im Projekt-Subordner `datasets/images`
6. Training des Modells: `train.py`
7. Evaluation des Modells: `predict.py`
