# Linguistic Similarity Analysis of Major vs. Minor Characters in Homer's Epics

## Research Question

**Claim**: Major speaking characters in the Iliad (Achilles, Hector, Agamemnon, etc.) exhibit greater linguistic similarity to each other than minor characters. This methodology should generalize to the Odyssey.

---

## 📁 Project Structure

```
NLP-Final-Project/
├── data/                          # Source texts (Iliad & Odyssey)
│   ├── Homer_Iliad_book.txt
│   └── Homer_Odyssey_book.txt
├── final_notebook/
│   └── Final_notebook.ipynb       # Main analysis notebook
├── notebooks resources/           # Reference notebooks
│   ├── 1_AppliedNLP_Session2_Bi_Trigrams.ipynb
│   ├── 2_AppliedNLP_Session2_POS_Patterns.ipynb
│   ├── 2_AppliedNLP_Session4_Character_Centered_Topic_Drift.ipynb
│   └── 1_AppliedNLP_Session5_RAG.ipynb
├── results/
│   ├── figures/                   # Generated visualizations
│   └── tables/                    # Generated CSV tables
├── slides/                        # Presentation materials
├── .gitignore
├── README.md                      # This file
├── report.md                      # Detailed methodology and results
└── requirements.txt               # Python dependencies
```

---

## 🚀 Environment Setup

### Prerequisites
- Python 3.10+ recommended
- ~2GB disk space for dependencies

### Setup Instructions

#### Windows (PowerShell)
```powershell
# Navigate to project directory
cd C:\path\to\NLP-Final-Project

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

#### Windows (Git Bash)
```bash
# Navigate to project directory
cd /path/to/NLP-Final-Project

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/Scripts/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

#### macOS / Linux
```bash
# Navigate to project directory
cd /path/to/NLP-Final-Project

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Deactivate Environment
```bash
deactivate
```

---

## 📓 Running the Analysis

1. Ensure the virtual environment is activated
2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `final_notebook/Final_notebook.ipynb`
4. Run all cells (Kernel → Restart & Run All)
5. Results will be saved to `results/figures/` and `results/tables/`

---

## 📊 Analysis Overview

The notebook performs:

1. **Speech Extraction**: Regex-based dialogue attribution with multi-paragraph handling
2. **Character Classification**: Major/minor based on speaking frequency and volume
3. **Multi-Dimensional Linguistic Analysis**:
   - Lexical patterns (TF-IDF, vocabulary, n-grams)
   - Syntactic patterns (POS tag distributions)
   - Semantic similarity (sentence embeddings)
   - Stylometric features (sentence length, pronouns, etc.)
4. **Clustering & Visualization**: Heatmaps, dendrograms, PCA plots
5. **Hypothesis Testing**: Statistical comparison of group similarities

---

## 📄 Documentation

- **`report.md`**: Comprehensive methodology explanation, results interpretation, and conclusions
- **`final_notebook/Final_notebook.ipynb`**: Executable analysis with inline documentation

---

## 📈 Key Results

Results are saved to:
- `results/figures/` - All visualizations (PNG)
- `results/tables/` - Data tables (CSV)

See `report.md` for detailed interpretation.

---

## 🛠 Troubleshooting

### Common Issues

1. **spaCy model not found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **CUDA/GPU issues with PyTorch**:
   - The analysis runs on CPU by default
   - sentence-transformers will use GPU if available

3. **Memory issues**:
   - Close other applications
   - Restart the Jupyter kernel if needed

4. **PowerShell execution policy error**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

---

## 📚 References

- Homer. *The Iliad*. Translated by Samuel Butler. Project Gutenberg.
- Homer. *The Odyssey*. Translated by Samuel Butler. Project Gutenberg.
- Reference notebooks adapted from Applied NLP course materials.

---

## 👥 Contributors

NLP Final Project Team
