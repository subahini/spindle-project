#  Sleep Spindle Detection using CNN
*A Master's thesis project by Subahini Nadarajh (University of Basel)*

---

## Project Overview

This project focuses on detecting **sleep spindles** in EEG recordings using a **Convolutional Neural Network (CNN)**. The goal is to build a model that learns to identify spindles directly from raw EEG signals — without relying on pre-existing models for labeling.

The project uses data stored on the **DBE lakeFS server** and processes it into a clean format for model training and evaluation. The work includes signal preprocessing, label alignment, and training a CNN with temporal windows of EEG data.


Getting Started

 1. Clone the Repository

```bash
git clone https://github.com/yourusername/spindle-project.git
cd spindle-project
```

2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3.Install Dependencies

```bash
pip install -r requirements.txt
```

---
## Configuration

Edit the file `config/pipeline.yaml` to point to the EEG file and label file you want to work with.

---

##  Downloading Data

### Download EEG from lakeFS:
```bash
python code/download_raw.py
```

### Download Spindle Labels:
```bash
python code/download_labels.py
```

---

## Preprocessing

Segments the EEG into overlapping windows and labels each based on spindle presence (from JSON):

```bash
python code/preprocess.py
```

You’ll get:
- `data/X_windows.npy`: preprocessed EEG windows (e.g., shape `[N, 6, 512]`)
- `data/y_labels.npy`: corresponding binary labels (0 = no spindle, 1 = spindle)

---

## Training the CNN

Train the model using:

```bash
python code/train_cnn.py
```

The model uses a lightweight 1D CNN that:
- Takes EEG from 6 channels
- Detects temporal patterns
- Outputs a probability of spindle presence


## Author

**Subahini Nadarajh**  
subahini.nadarajh@stud.unibas.ch

Examiner & Supervisor: Prof. Dr. Volker Roth
Supervisor: Florentin Bieder
Project Collaborators: Prof. Alex Datta and Dr. Martina Studer (UKBB)


Faculty of Science, University of Basel
Department of Mathematics and Computer Science


In collaboration with the Department of Biomedical Engineering
CIAN Group – Center for medical Image Analysis & Navigation
and University Children’s Hospital Basel (UKBB)


---

## Acknowledgements

- Data courtesy of the DBE sleep spindle research team  
- lakeFS for large-scale data versioning  
- MNE and PyTorch libraries  

