import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from models.deep.encoding import BiologicalMismatchEncoder
from models.deep.neural_scm import NeuralSCM

# Regole biochimiche per le mutazioni
TRANSITIONS = {'A': 'G', 'G': 'A', 'C': 'T', 'T': 'C'}
# Per la trasversione, ne scegliamo una rappresentativa (purina <-> pirimidina)
TRANSVERSIONS = {'A': 'C', 'C': 'A', 'G': 'C', 'T': 'A'}
# Il Wobble classico RNA-DNA mimato
WOBBLES = {'G': 'T', 'T': 'G'} 

def mutate_target(guide_seq: str, pos: int, mut_type: str):
    """
    Data una sequenza guida reale, muta il TARGET in base al tipo di mismatch.
    Restituisce il target mutato, o None se la mutazione non è biochimicamente applicabile.
    """
    target = list(guide_seq)
    base = target[pos]
    
    if mut_type == "wobble":
        if base in WOBBLES:
            target[pos] = WOBBLES[base]
        else:
            return None # Impossibile fare wobble su A o C secondo le nostre regole
    elif mut_type == "transition":
        target[pos] = TRANSITIONS[base]
    elif mut_type == "transversion":
        target[pos] = TRANSVERSIONS[base]
        
    return "".join(target)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = Path("experiments/results/Exp06_TypedMLP_HardPrior/neural_scm.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato in {model_path}")

    # Caricamento del modello causale (Run 10)
    encoder = BiologicalMismatchEncoder()
    model = NeuralSCM(encoder=encoder, architecture="typed_mlp", hidden_dim=8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Pool di guide REALI (veri positivi terapeutici standard con GC 45-55%)
    # Sostituiscile con le tue top guide estratte dal dataset CHANGE-seq
    real_guides = [
        "GAGTCCGAGCAGAAGAAGAA", # EMX1
        "GGTGAGTGAGTGTGTGCGTG", # VEGFA
        "GGCCCAGACTGAGCACGTGA", # HEK293
        "GACGGAATCTCCTGTATCTA", 
        "CACCGCGTAAGCAGTCCTCA",
        "TGGACCTTACGGCTACACTG"
    ]

    positions = list(range(20))
    mismatch_types = ["wobble", "transition", "transversion"]
    
    # Struttura dati per accumulare le penalità: dict[tipo][posizione] = [valori...]
    raw_results = {m: {p: [] for p in positions} for m in mismatch_types}

    with torch.no_grad():
        for guide in real_guides:
            # Calcolo baseline (Match perfetto ON-TARGET)
            out_base = model([guide], [guide])
            logit_base = out_base["logit"].item()

            for pos in positions:
                for m_type in mismatch_types:
                    mutated_target = mutate_target(guide, pos, m_type)
                    
                    # Se la mutazione è applicabile a questa base
                    if mutated_target is not None:
                        out = model([guide], [mutated_target])
                        logit_mut = out["logit"].item()
                        
                        # Penalità causale isolata
                        penalty = logit_base - logit_mut
                        raw_results[m_type][pos].append(penalty)

    # Calcolo della media delle penalità per ogni posizione
    avg_results = {m_type: [] for m_type in mismatch_types}
    for m_type in mismatch_types:
        for pos in positions:
            penalties = raw_results[m_type][pos]
            if len(penalties) > 0:
                avg_results[m_type].append(np.mean(penalties))
            else:
                # Se per caso nessuna guida aveva la lettera giusta per quella posizione
                avg_results[m_type].append(0.0) 

    # --- Plotting Termodinamico ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    colors = {"wobble": "blue", "transition": "orange", "transversion": "red"}

    for m_type in mismatch_types:
        plt.plot(np.array(positions) + 1, avg_results[m_type], 
                 marker='o', linewidth=2.5, color=colors[m_type], 
                 label=m_type.capitalize())

    # Zone Cas9
    plt.axvspan(1, 8, color='gray', alpha=0.1, label='Distal / Non-Seed')
    plt.axvspan(9, 16, color='yellow', alpha=0.1, label='Seed')
    plt.axvspan(17, 20, color='red', alpha=0.1, label='PAM-Proximal')

    plt.title("Impatto Termodinamico Causale (Media su Guide Reali)", fontsize=16, fontweight='bold')
    plt.xlabel("Posizione del Mismatch (1=Distale, 20=PAM)", fontsize=12)
    plt.ylabel("Penalità Causale Media (Δ Logit)", fontsize=12)
    plt.xticks(range(1, 21))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    out_dir = Path("explainability/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "thermodynamic_profile_real_guides.png"
    plt.savefig(out_file, dpi=300)
    print(f"\nGrafo basato su guide reali salvato in: {out_file}")

if __name__ == "__main__":
    main()