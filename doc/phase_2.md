# Fase 2 

## Obiettivi
La fase 2 risponde a tre domande precise, in ordine:
Obiettivo A — Formalizzare l'SCM con equazioni strutturali stimate dai dati, non assunte. Verificare che le implicazioni causali del DAG (indipendenze condizionali) siano presenti in CHANGE-seq.
Obiettivo B — Rispondere a query interventionali P(activity∣do(X=x))P(\text{activity} \mid do(X=x))
P(activity∣do(X=x)) usando l'adjustment formula derivata dal DAG. Dimostrare che le risposte interventionali differiscono da quelle associazionali — questo è il punto in cui mostri che XGBoost sbaglia per ragioni strutturali.
Obiettivo C — Costruire il dataset sintetico di interventi controllati che servirà come training signal per il Neural SCM in fase 3.

## File da implementare e responsabilità

dag/scm.py

Dataclass StructuralEquation che rappresenta un singolo arco causale con la sua forma funzionale (moltiplicativa per PAM, sigmoidale per seed)
Classe CRISPRCausalModel che contiene tutte le equazioni strutturali del DAG
Metodo fit(df) che stima i parametri α, β, γ, δ dalle equazioni sui dati osservati (MLE)
Metodo predict(pair) che propaga in avanti attraverso le equazioni strutturali
Metodo sample_exogenous(pair, observed_activity) che inferisce i rumori UU
U dai dati osservati — questo è l'abduction step per i controfattuali


dag/do_calculus.py

Funzione backdoor_adjustment(dag, treatment, outcome) che identifica il set di aggiustamento corretto dato il DAG
Funzione do_query(scm, df, intervention: dict) che risponde a P(Y∣do(X=x))P(Y \mid do(X=x))
P(Y∣do(X=x)) applicando l'adjustment formula
Funzione compare_observational_vs_interventional(scm, df, intervention) che calcola la differenza tra risposta associazionale e interventionale — questo è il risultato principale della fase 2
Funzione build_intervention_dataset(df, interventions: list[dict]) che genera il dataset sintetico di coppie con interventi controllati, da usare come training signal in fase 3


dag/independence_tests.py

Funzione test_conditional_independence(df, X, Y, Z) che testa se X⊥Y∣ZX \perp Y \mid Z
X⊥Y∣Z nei dati usando il test di correlazione parziale di Spearman
Funzione validate_dag_implications(dag, df) che estrae tutte le indipendenze condizionali implicite dal DAG e le testa sistematicamente sui dati — questo è il test di misspecification del DAG
Restituisce un report con quali indipendenze sono rispettate e quali violate, con p-value corretto per test multipli (Benjamini-Hochberg)


evaluation/ccs.py

Funzione causal_consistency_score(model, df, interventions) che per ogni esempio del test set genera varianti sintetiche con interventi controllati (es. PAM NGG→NAA, mismatch pos1 0→1) e misura la percentuale di casi in cui la predizione del modello cambia nella direzione causalmente attesa
Funzione pam_gate_consistency(model, df) — caso specifico: mutare il PAM da canonico a non canonico deve ridurre lo score di almeno il 50% in ogni caso
Funzione seed_intervention_consistency(model, df) — aggiungere un mismatch in posizione 1 deve sempre ridurre lo score rispetto alla coppia senza quel mismatch
Restituisce un valore in [0,1] dove 1 = perfettamente consistente con i vincoli causali


experiments/exp_02_scm/config.yaml

Parametri per la stima delle equazioni strutturali
Lista degli interventi da testare
Dataset sintetico: numero di varianti per esempio, posizioni da mutare, PAM alternativi


experiments/exp_02_scm/run.py

Step 1: carica i dati e il modello XGBoost dalla fase 1
Step 2: testa le indipendenze condizionali del DAG sui dati — se più del 30% fallisce, il DAG va rivisto prima di procedere
Step 3: stima i parametri dell'SCM con CRISPRCausalModel.fit()
Step 4: confronta predizioni associazionali vs interventionali su un set di query biologicamente motivate
Step 5: calcola CCS sul modello XGBoost baseline — questo è il punto di riferimento per la fase 3
Step 6: genera e salva il dataset sintetico di interventi per il training del Neural SCM


## Output attesi della fase 2
experiments/results/exp_02_scm/
├── dag_independence_tests.csv     ← quali implicazioni causali reggono nei dati
├── scm_parameters.json           ← α, β, γ, δ stimati
├── observational_vs_do.csv       ← differenza P(Y|X) vs P(Y|do(X))
├── ccs_baseline.json             ← CCS del modello XGBoost (riferimento per fase 3)
└── intervention_dataset.parquet  ← dataset sintetico per Neural SCM

## TODO Tecnici
1. Raffinare `backdoor_adjustment` in `dag/do_calculus.py`.
	Stato attuale: usa una strategia conservativa basata su antenati comuni e esclusione dei discendenti del treatment.
	Perche va bene ora: con il DAG corrente e piccolo e sufficiente per le query principali della fase 2.
	Upgrade previsto: aggiungere un controllo formale via d-separation / backdoor path blocking per casi piu complessi.

2. Raffinare il confronto osservazionale in `do_query`.
	Stato attuale: `P(Y|X=x)` e stimata con matching esatto sui valori di intervento (`np.isclose` per numerici).
	Perche va bene ora: mantiene una baseline semplice e trasparente per mostrare il gap con `P(Y|do(X=x))`.
	Upgrade previsto: introdurre binning/nearest-neighbors o smoothing quando `n_observational` e troppo piccolo.

## La connessione con la fase 3
Il risultato più importante della fase 2 non è la performance predittiva — è il gap tra risposte associazionali e interventionali. Se P(Y∣PAM=NAG)≠P(Y∣do(PAM=NAG))P(Y \mid \text{PAM}=\text{NAG}) \neq P(Y \mid do(\text{PAM}=\text{NAG}))
P(Y∣PAM=NAG)=P(Y∣do(PAM=NAG)) in modo sistematico, hai dimostrato empiricamente che i modelli associazionali (XGBoost, CCLMoff) non possono rispondere alle domande clinicamente rilevanti. Il Neural SCM in fase 3 viene motivato esattamente da questo gap.