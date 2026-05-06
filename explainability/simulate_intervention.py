import torch
import numpy as np
from pathlib import Path
from models.deep.encoding import BiologicalMismatchEncoder
from models.deep.neural_scm import NeuralSCM

def calculate_prob(logit: float) -> float:
    return torch.sigmoid(torch.tensor(logit)).item() * 100

def logit_from_prob(prob_pct: float) -> float:
    """Ricava il logit dalla probabilità, evitando infiniti"""
    p = np.clip(prob_pct / 100.0, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

def reads_to_prob(reads: int, max_reads: int, method: str = "log") -> float:
    """
    Converte i read counts in probabilità mitigando i bias di amplificazione PCR.
    """
    # Protezione matematica di base
    reads = max(0, reads)
    max_reads = max(reads, max_reads) # Il max deve essere almeno uguale alle reads

    if method == "linear":
        p = reads / max_reads
    elif method == "log":
        # log1p è l'equivalente di np.log(x + 1), previene errori con lo 0
        p = np.log1p(reads) / np.log1p(max_reads)
    else:
        raise ValueError(f"Metodo {method} non supportato.")
        
    # Scaliamo e cappiamo al 99% per evitare problemi con l'infinito nel calcolo dei logit
    return min(p * 100.0, 99.0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inizializzazione Oracolo Causale su device: {device}\n")
    
    model_path = Path("experiments/results/Exp06_TypedMLP_HardPrior/neural_scm.pt")
    encoder = BiologicalMismatchEncoder()
    model = NeuralSCM(encoder=encoder, architecture="typed_mlp", hidden_dim=8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ==========================================
    # DATI FATTUALI (Inserisci qui i tuoi veri dati!)
    # ==========================================
    guide_wt =   "GTCCCTAGTGGCCCCACTGT" 
    off_target = "ATATTTAGTGGCTCCACTGTGGG"
    
    # I TUOI DATI REALI DAL DATASET CHANGE-SEQ
    off_target_reads = 3130  # Quante reads ha fatto questo off-target?
    on_target_reads = 4614  # Quante reads ha fatto l'on-target (o il max dell'esperimento)?
    
    y_obs_prob = reads_to_prob(off_target_reads, on_target_reads, method="log")
    y_obs_logit = logit_from_prob(y_obs_prob)

    print("--- 1. ABDUCTION (Calcolo del Rumore di Saggio U) ---")
    with torch.no_grad():
        out_factual = model([guide_wt], [off_target])
        y_pred_logit = out_factual['logit'].item()
        
        # U è l'esatta differenza tra la chimica predetta e i read reali
        U = y_obs_logit - y_pred_logit
        
        print(f"Dati Lab (CHANGE-seq): {off_target_reads} reads (Efficienza: {y_obs_prob:.1f}%)")
        print(f"Predizione Termodinamica SCM: {calculate_prob(y_pred_logit):.1f}%")
        print(f"-> Rumore Esogeno Inferito (U): {U:+.3f}\n")

    print("--- 2. INTERVENTO (Do-Calculus: Troncamento della Guida) ---")
    # Invece di intervenire su un nodo, facciamo un intervento reale sulla sequenza (tru-gRNA)
    # ma tenendo conto di U!
    guide_tru = "NN" + guide_wt[2:]
    
    with torch.no_grad():
        out_intervened = model([guide_tru], [off_target])
        y_do_logit = out_intervened['logit'].item()
        
        print(f"Azione: do(Guida = Troncata di 2 nucleotidi al 5')")
        print(f"Impatto Termodinamico Puro: {calculate_prob(y_do_logit):.1f}%\n")

    print("--- 3. PREDICATO CONTROFATTUALE COMPLETO ---")
    # Step 3: Sommiamo la nuova termodinamica al vecchio rumore U
    y_counterfactual_logit = y_do_logit + U
    y_counterfactual_prob = calculate_prob(y_counterfactual_logit)
    
    print(f"Se in quello specifico esperimento avessimo usato la guida troncata,")
    print(f"i {off_target_reads} read sarebbero diventati un'efficienza del:")
    print(f"-> {y_counterfactual_prob:.1f}%")

    print("\n--- 4. INTERVENTO B (Do-Calculus: Rescue Mutation nel Seed) ---")
    # L'off-target è molto forte (alto U). Il troncamento distale potrebbe non bastare.
    # Proviamo a ingegnerizzare una mutazione severa (Trasversione) nel Seed.
    # La guida WT ha una 'C' in posizione 15 (indice 14). La mutiamo in 'A'.
    guide_list = list(guide_wt)
    guide_list[14] = "A"  
    guide_mut = "".join(guide_list)
    
    with torch.no_grad():
        out_intervened_mut = model([guide_mut], [off_target])
        y_do_mut_logit = out_intervened_mut['logit'].item()
        
        print(f"Azione: do(Guida = Mutazione mirata in pos 15, C -> A)")
        print(f"Impatto Termodinamico Puro: {calculate_prob(y_do_mut_logit):.1f}%\n")

    print("--- 5. PREDICATO CONTROFATTUALE (Intervento B) ---")
    # Sommiamo la nuova termodinamica della Rescue Mutation al vecchio rumore U
    y_cf_mut_logit = y_do_mut_logit + U
    y_cf_mut_prob = calculate_prob(y_cf_mut_logit)
    
    print(f"Se in quello specifico esperimento avessimo usato la guida con Rescue Mutation,")
    print(f"l'efficienza di questo grave off-target sarebbe diventata:")
    print(f"-> {y_cf_mut_prob:.1f}%")

    print("\n--- 6. TRADE-OFF CLINICO (Impatto sull'On-Target) ---")
    # L'On-Target reale è la Riga 6 del dataset
    on_target = "GTCCCTAGTGGCCCCACTGTGGG"
    true_on_target_reads = 4614  # Le reads effettive della Riga 6

    # A. Abduzione per l'On-Target (Calcolo di U_on)
    # Usiamo on_target_reads (540, definito in cima) come denominatore per la normalizzazione
    y_obs_on_prob = reads_to_prob(true_on_target_reads, on_target_reads, method="log")
    y_obs_on_logit = logit_from_prob(y_obs_on_prob)

    with torch.no_grad():
        out_on_factual = model([guide_wt], [on_target])
        y_pred_on_logit = out_on_factual['logit'].item()
        U_on = y_obs_on_logit - y_pred_on_logit

        print(f"Abduzione On-Target: {true_on_target_reads} reads (Efficienza: {y_obs_on_prob:.1f}%)")
        print(f"Predizione Termodinamica (Match perfetto): {calculate_prob(y_pred_on_logit):.1f}%")
        print(f"-> Rumore Esogeno On-Target (U_on): {U_on:+.3f}\n")

        # B. Applichiamo la STESSA Rescue Mutation (guide_mut) all'On-Target
        out_on_mut = model([guide_mut], [on_target])
        y_do_on_logit = out_on_mut['logit'].item()

        # C. Controfattuale On-Target
        y_cf_on_logit = y_do_on_logit + U_on
        y_cf_on_prob = calculate_prob(y_cf_on_logit)

        print("--- VERDETTO FINALE DELL'ORACOLO ---")
        print(f"Efficienza On-Target (Terapia): {y_obs_on_prob:.1f}%  --->  {y_cf_on_prob:.1f}%")
        print(f"Rischio Off-Target (Pericolo):  {y_obs_prob:.1f}%  --->  {y_cf_mut_prob:.1f}%\n")

        # Regola decisionale clinica razionale: 
        # La terapia deve restare decente (>50%) e l'off-target deve essere soppresso (<20%)
        if y_cf_on_prob > 50.0 and y_cf_mut_prob < 20.0:
            print("[APPROVATA] La mutazione salva il paziente mantenendo l'efficacia terapeutica.")
        else:
            print("[RIGETTATA] Trade-off sfavorevole. La mutazione distrugge l'efficacia o non spegne il rischio.")
            print("Azione consigliata: Cambiare la regione genomica bersaglio.")

if __name__ == "__main__":
    main()