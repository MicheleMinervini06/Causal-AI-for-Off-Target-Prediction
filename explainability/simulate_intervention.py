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
    """Converte i read counts in probabilità mitigando i bias di amplificazione PCR."""
    reads = max(0, reads)
    max_reads = max(reads, max_reads)
    if method == "linear":
        p = reads / max_reads
    elif method == "log":
        p = np.log1p(reads) / np.log1p(max_reads)
    else:
        raise ValueError(f"Metodo {method} non supportato.")
    return min(p * 100.0, 99.0)

def compute_gc_context(sgRNA: str, target: str, device: torch.device) -> torch.Tensor:
    """
    Calcola al volo il tensore di contesto GC richiesto dal DAG (nodo esogeno).
    Le features sono: [gc_sgRNA, gc_offtarget, concept_gc_delta]
    """
    gc_sg = sum(1 for c in sgRNA if c in 'GC') / len(sgRNA)
    gc_tg = sum(1 for c in target if c in 'GC') / len(target)
    delta = gc_sg - gc_tg
    return torch.tensor([[gc_sg, gc_tg, delta]], dtype=torch.float32, device=device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inizializzazione Oracolo Causale su device: {device}\n")
    
    # 1. Puntiamo al nuovo modello di Stato dell'Arte
    model_path = Path("experiments/results/Exp15_Positional_ExtendedOneCycle/neural_scm.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato in {model_path}")

    state_dict = torch.load(model_path, map_location=device)

    context_dim = 0
    if "context_net.0.weight" in state_dict:
        context_dim = state_dict["context_net.0.weight"].shape[1]

    encoder = BiologicalMismatchEncoder()
    model = NeuralSCM(encoder=encoder, architecture="positional_mlp", hidden_dim=8, context_dim=context_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ==========================================
    # DATI FATTUALI (Esempio dal Dataset)
    # ==========================================
    # guide_wt =   "GTCCCTAGTGGCCCCACTGT" 
    # off_target = "ATATTTAGTGGCTCCACTGTGGG"
    guide_wt =   "GTCCCTAGTGGCCCCACTGT" 
    off_target = "GTCCCCAAAGCCCCCACTGTGGG"
    
    # off_target_reads = 3130  
    # on_target_reads = 4614  
    off_target_reads = 193    # Le reads effettive dell'off-target
    on_target_reads = 51571   # Il massimo dell'esperimento (riga 1491) 
    
    y_obs_prob = reads_to_prob(off_target_reads, on_target_reads, method="log")
    y_obs_logit = logit_from_prob(y_obs_prob)

    print("--- 1. ABDUCTION (Calcolo del Rumore Cellulare U) ---")
    with torch.no_grad():
        ctx_factual = compute_gc_context(guide_wt, off_target, device)
        # 3. Passiamo il contesto esogeno al modello
        out_factual = model([guide_wt], [off_target], context_features=ctx_factual)
        y_pred_logit = out_factual['logit'].item()
        
        U = y_obs_logit - y_pred_logit
        
        print(f"Dati Lab (Fattuali): {off_target_reads} reads (Efficienza: {y_obs_prob:.1f}%)")
        print(f"Predizione Termodinamica (White-Box): {calculate_prob(y_pred_logit):.1f}%")
        print(f"-> Rumore Esogeno Inferito (U): {U:+.3f}\n")

    print("--- 2. INTERVENTO (Do-Calculus: Troncamento della Guida) ---")
    guide_tru = "NN" + guide_wt[2:]
    
    with torch.no_grad():
        # Ricalcoliamo il contesto! Il troncamento potrebbe aver abbassato il GC%
        ctx_tru = compute_gc_context(guide_tru, off_target, device)
        out_intervened = model([guide_tru], [off_target], context_features=ctx_tru)
        y_do_logit = out_intervened['logit'].item()
        
        print(f"Azione: do(Guida = Troncata di 2 nucleotidi al 5')")
        print(f"Impatto Termodinamico Puro post-intervento: {calculate_prob(y_do_logit):.1f}%\n")

    print("--- 3. PREDICATO CONTROFATTUALE COMPLETO ---")
    y_counterfactual_logit = y_do_logit + U
    y_counterfactual_prob = calculate_prob(y_counterfactual_logit)
    
    print(f"Se in quello specifico esperimento avessimo usato la guida troncata,")
    print(f"i {off_target_reads} read sarebbero diventati un'efficienza del:")
    print(f"-> {y_counterfactual_prob:.1f}%")

    print("\n--- 4. INTERVENTO B (Do-Calculus: Rescue Mutation in pos 15) ---")
    # Mutiamo la posizione 14 (che corrisponde all'indice 14 del Python array, 15esima base)
    guide_list = list(guide_wt)
    guide_list[14] = "A"  
    guide_mut = "".join(guide_list)
    
    with torch.no_grad():
        ctx_mut = compute_gc_context(guide_mut, off_target, device)
        out_intervened_mut = model([guide_mut], [off_target], context_features=ctx_mut)
        y_do_mut_logit = out_intervened_mut['logit'].item()
        
        print(f"Azione: do(Guida = Mutazione mirata in pos 15, C -> A)")
        print(f"Impatto Termodinamico Puro post-intervento: {calculate_prob(y_do_mut_logit):.1f}%\n")

    print("--- 5. PREDICATO CONTROFATTUALE (Intervento B) ---")
    y_cf_mut_logit = y_do_mut_logit + U
    y_cf_mut_prob = calculate_prob(y_cf_mut_logit)
    
    print(f"Se avessimo usato la guida con Rescue Mutation contro questo off-target,")
    print(f"l'efficienza sarebbe diventata:")
    print(f"-> {y_cf_mut_prob:.1f}%")

    print("\n--- 6. TRADE-OFF CLINICO CAUSALE (Impatto sull'On-Target) ---")
    #on_target = "GTCCCTAGTGGCCCCACTGTGGG"
    #true_on_target_reads = 4614  

    on_target = "GTCCCTAGTGGCCCCACTGTGGG"
    true_on_target_reads = 51571

    # Abduzione On-Target
    y_obs_on_prob = reads_to_prob(true_on_target_reads, on_target_reads, method="log")
    y_obs_on_logit = logit_from_prob(y_obs_on_prob)

    with torch.no_grad():
        ctx_on = compute_gc_context(guide_wt, on_target, device)
        out_on_factual = model([guide_wt], [on_target], context_features=ctx_on)
        y_pred_on_logit = out_on_factual['logit'].item()
        U_on = y_obs_on_logit - y_pred_on_logit

        print(f"Abduzione On-Target: {true_on_target_reads} reads (Efficienza: {y_obs_on_prob:.1f}%)")
        print(f"Predizione Termodinamica (Match perfetto): {calculate_prob(y_pred_on_logit):.1f}%")
        print(f"-> Rumore Esogeno On-Target (U_on): {U_on:+.3f}\n")

        # Controfattuale On-Target (Cosa succede alla Terapia se applichiamo la mutazione di salvataggio?)
        ctx_on_mut = compute_gc_context(guide_mut, on_target, device)
        out_on_mut = model([guide_mut], [on_target], context_features=ctx_on_mut)
        y_do_on_logit = out_on_mut['logit'].item()

        y_cf_on_logit = y_do_on_logit + U_on
        y_cf_on_prob = calculate_prob(y_cf_on_logit)

        print("--- VERDETTO FINALE DELL'ORACOLO ---")
        print(f"Efficienza On-Target (Terapia): {y_obs_on_prob:.1f}%  --->  {y_cf_on_prob:.1f}%")
        print(f"Rischio Off-Target (Pericolo):  {y_obs_prob:.1f}%  --->  {y_cf_mut_prob:.1f}%\n")

        if y_cf_on_prob > 50.0 and y_cf_mut_prob < 20.0:
            print("[APPROVATA] La mutazione salva il paziente mantenendo l'efficacia terapeutica.")
        else:
            print("[RIGETTATA] Trade-off sfavorevole. La mutazione distrugge l'efficacia o non spegne il rischio.")


if __name__ == "__main__":
    main()