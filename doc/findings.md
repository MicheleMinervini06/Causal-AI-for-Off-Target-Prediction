# Findings

Documento di ricerca per tracciare osservazioni empiriche, anomalie e decisioni metodologiche emerse durante gli esperimenti. Aggiornare ad ogni fase.

---

## Fase 1 — Baseline XGBoost/CatBoost con feature DAG

### F1 — Generalizzazione cross-assay: calo netto di AUPRC con AUROC alto

**Esperimento:** valutazione within-dataset (CHANGE-seq) e cross-assay (GUIDE-seq).

**Risultato:**

| Modello | AUPRC within | AUPRC cross | Delta | AUROC cross |
|---|---|---|---|---|
| XGBoost | 0.4393 | 0.1853 | -57% | 0.966 |
| CatBoost | 0.4916 | 0.2744 | -44% | 0.974 |

**Interpretazione:** il modello sa ordinare correttamente i positivi su GUIDE-seq (AUROC alto) ma è mal calibrato — il threshold ottimizzato su CHANGE-seq non trasferisce. Il problema non è la rappresentazione ma la distribuzione degli score tra i due assay. Questo è un segnale di overfitting al protocollo sperimentale di CHANGE-seq, non al meccanismo biologico sottostante.

**Implicazione per la tesi:** motiva l'uso di un modello con vincoli causali strutturali che apprendano il meccanismo e non le correlazioni specifiche dell'assay.

---

### Aggiornamento — Risultati con nuovo split (exp_01)

Analisi risultati exp_01 con nuovo split

Metriche predittive
- XGBoost:  AUPRC within=0.144  cross=0.270
- CatBoost: AUPRC within=0.183  cross=0.231

AUPRC within è sceso rispetto al run precedente (0.44/0.49) — atteso. Il test set ora è più rappresentativo e bilanciato. Il dato interessante è che cross-assay è migliorato per XGBoost (0.185→0.270) — il modello generalizza meglio con split corretto.

DAG Validation — nuovo finding
- node_C_seed_extension → label:  ρ=+0.011  atteso NEGATIVO  ✗ FAIL
- node_D_non_seed       → label:  ρ=+0.071  atteso NEGATIVO  ✗ FAIL

Con il nuovo split ora falliscono sia node_C che node_D — prima falliva solo node_D. Con il vecchio split le guide ricche di positivi nel val mascheravano questo pattern.

Ablation — confermato
- CatBoost no_energy_full:      0.2055  ← migliore
- CatBoost no_aggregate_energy: 0.2019
- CatBoost full_dag:            0.1829

Il pattern precedente regge: le feature energetiche aggregate peggiorano la performance.

Feature importance — segnale chiave
- mismatch_count:  0.134  ← feature più importante
- pam_score:       0.109
- profile_pos_20:  0.076  ← posizione PAM-proximal
- gc_sgRNA:        0.069
- node_B_proximal: 0.015  ← nodo DAG poco usato
- node_C_seed:     0.015

XGBoost usa principalmente feature aggregate (mismatch_count, mismatch_rate) e il profilo posizionale, non i nodi DAG strutturati. Questo è coerente con R2=0.0 nel CCS — il modello non ha imparato l'effetto causale posizione-specifico.

### F2 — Feature energetiche aggregate peggiorano la performance

**Esperimento:** ablation study con varianti `no_aggregate_energy`, `no_energy_full`, `full_dag`.

**Risultato:**

| Variante | XGBoost AUPRC | CatBoost AUPRC |
|---|---|---|
| no_energy_full | 0.4584 | 0.5310 |
| no_aggregate_energy | 0.4489 | 0.5332 |
| full_dag | 0.4393 | 0.4916 |

**Interpretazione:** le feature energetiche aggregate (`mean_energy_penalty`, `total_energy_penalty`, `concept_energy`) peggiorano la performance rispetto alla loro rimozione. I nodi energetici nodali (`node_B_proximal`, `node_C_seed_extension`, `node_D_non_seed`) non aggiungono informazione significativa rispetto alle feature di conteggio — la differenza tra `no_aggregate_energy` e `no_energy_full` è < 0.01 per entrambi i modelli.

**Ipotesi:** i pesi energetici (wobble=0.4, transition=0.75, transversion=1.0) non sono calibrati ottimalmente sui dati CHANGE-seq. Aggiungono rumore perché sono collineari con `mismatch_count` ma con una pesatura arbitraria.

**Decisione:** usare `no_aggregate_energy` come configurazione base per tutti gli esperimenti successivi. I pesi energetici andranno stimati dai dati nell'SCM in fase 2 invece di essere assunti a priori.

---

### F3 — PAM solo non è informativo come feature

**Esperimento:** ablation `pam_only`.

**Risultato:** AUPRC = 0.04 (quasi casuale) per entrambi i modelli.

**Interpretazione:** quasi tutti i target nel dataset CHANGE-seq hanno PAM canonico NGG — il PAM da solo non discrimina perché non c'è variabilità sufficiente. Il PAM è invece un gate causale forte (biologicamente il primo checkpoint del meccanismo Cas9), ma la sua rilevanza emerge solo in interazione con le altre feature, non marginalmente.

**Implicazione per il DAG:** il PAM va modellato come gate moltiplicativo nell'SCM (fase 2), non come feature additiva come fa attualmente XGBoost.

---

### F4 — node_D_non_seed fallisce il test esterno: confounding nel DAG

**Esperimento:** validazione DAG, test esterno `node_D_non_seed → label`.

**Risultato:**
```
node_D_non_seed → label:  ρ = +0.028  atteso NEGATIVO  ✗ FAIL
```

**Interpretazione:** la correlazione marginale di `node_D` con la label è positiva, contrariamente all'ipotesi causale (più energia in non-seed = meno attività). Il motivo è confounding strutturale: i target con alta energia in non-seed ma bassa in seed tendono ad essere off-target attivi perché la seed è intatta. L'effetto di `node_D` non è diretto ma mediato e condizionato a `node_B` e `node_C`.

**Revisione DAG da valutare (da testare con independence tests in fase 2):**

- *Opzione A:* rimuovere l'arco diretto `node_D → activity` e modellare `node_D` come modificatore di `full_hybridization`:
  ```
  Prima:  node_D → activity  (arco diretto)
  Dopo:   node_D → full_hybridization → activity  (mediato)
  ```
- *Opzione B:* aggiungere un arco di interazione `node_B × node_D` — l'effetto di `node_D` è negativo solo quando `node_B` è basso (seed intatta).

**Da fare in fase 2:** testare con `dag/independence_tests.py` quale delle due opzioni è supportata dalle indipendenze condizionali nei dati.

**Aggiornamento dopo Fase 2:** l'ipotesi di confounding biologico resta plausibile, ma non e' sufficiente a spiegare tutte le dipendenze anomale osservate nei test CI globali. Parte del segnale e' verosimilmente dovuto a bias di costruzione del dataset (vedi F6.2).

---

### F5 — Correlazioni esterne del DAG molto basse ma nella direzione attesa

**Esperimento:** validazione DAG, test esterni verso `label`.

**Risultato:**
```
node_A_pam            → label:  ρ = +0.044  ✓
node_B_proximal       → label:  ρ = -0.038  ✓
node_C_seed_extension → label:  ρ = -0.049  ✓
mismatch_count        → label:  ρ = -0.040  ✓
```

**Interpretazione:** le correlazioni sono nella direzione biologicamente attesa ma hanno magnitudine molto bassa. Questo è atteso: le relazioni sono non lineari, il dataset è fortemente sbilanciato (41x), e le correlazioni marginali di Spearman sottostimano le relazioni condizionali. Non invalida il DAG — indica che le relazioni causali emergono solo condizionando sugli altri nodi, non marginalmente.

---

## Fase 2 — SCM parametrico + do-calculus + independence tests

### F6.1 — Test di indipendenza condizionale: alta violazione delle implicazioni del DAG

**Esperimento:** `exp_02_scm` con DAG potato (`node_A_pam`, `node_B_proximal`, `node_C_seed_extension`, `pam_score`, `mismatch_rate`, `label`) e test CI multipli con correzione FWER.

**Risultato:**

- `ci_failure_rate = 0.8889` (8/9 test respingono H0)
- esempio coerente col DAG: `label ⟂ node_A_pam | pam_score` con `p = 0.703`
- molte altre indipendenze teoriche non reggono sui dati osservati

**Interpretazione:** il DAG semplificato non descrive completamente la struttura statistica osservata. Tuttavia, il risultato non va letto automaticamente come "DAG biologicamente sbagliato": la pipeline negativa introduce dipendenze spurie (vedi F6.2).

---

### F6.2 — Dipendenza PAM-mismatch: probabile artefatto di costruzione dei negativi (Cas-OFFinder)

**Osservazione chiave:** nei test CI emerge una dipendenza forte tra variabili derivate da porzioni diverse della sequenza (PAM vs mismatch rate/profili mismatch), non attesa dal meccanismo causale biologico puro.

**Ipotesi supportata:** i negativi generati con Cas-OFFinder (fino a 6 mismatch, campionati con regole non condizionate sul PAM biologico) introducono una correlazione artificiosa tra PAM e mismatch nei negativi. Quindi una quota rilevante delle violazioni CI e' un **dataset bias**, non una misspecification meccanicistica del DAG.

**Implicazione metodologica:** separare la valutazione in:

- coerenza biologica del DAG/SCM
- robustezza rispetto al processo di negative sampling

e usare test stratificati per assay/source o reweighting dei negativi prima di concludere sulla struttura causale.

---

### F6.3 — SCM parametrico: parametri con segni biologicamente plausibili

**Risultato (fit train):**

- `pam_alpha = 1.00`
- `activity_delta_pam = +4.46`
- `activity_eta_proximal = -6.83`
- `activity_theta_seed = -13.33`

**Interpretazione:** il gate PAM e il costo mismatch (soprattutto seed-extension) sono coerenti con il meccanismo atteso. Il problema principale non e' il segno dei coefficienti, ma il mismatch tra modello causale ideale e distribuzione empirica dei campioni (bias + possibile eterogeneita' assay).

---

### F6.4 — CCS baseline (3 regole): coerenza causale ancora insufficiente

**Esperimento:** CCS sul baseline XGBoost con `mode = 3_rules`.

**Risultato:**

- `R1_PAM_Ablation = 1.0`
- `R2_Pos1_Mismatch = 0.0`
- `R3_Heal_Seed = 0.9375`
- `CCS_Overall = 0.0`

**Interpretazione:** anche con metrica ridotta (3 regole) il baseline fallisce la coerenza causale complessiva per violazione netta di R2. Quindi il passaggio a 3 regole non cambia la conclusione qualitativa: il baseline non e' causalmente consistente in modo globale.

---

### F6.5 — Dataset interventivo: fissata inconsistenza interna

**Problema precedente:** `build_intervention_dataset` applicava i valori di intervento senza propagare le variabili a valle, producendo blocchi con feature intervenute ma label non aggiornate.

**Fix implementato:** propagazione opzionale tramite SCM (`scm` opzionale) durante la costruzione del dataset sintetico.

**Esito:** il dataset interventivo ora include `activity_probability` coerente con l'intervento; `label` e' allineata alla soglia su probabilita' nel blocco sintetico.

---

## Todo — Da investigare nelle fasi successive

- [x] **Fase 2:** testare indipendenza condizionale con `dag/independence_tests.py` (eseguito; outcome fortemente influenzato da bias dei negativi da validare con analisi stratificate)
- [x] **Fase 2:** stimare i pesi energetici/strutturali dai dati nell'SCM parametrico (versione potato)
- [x] **Fase 2:** implementare PAM come gate moltiplicativo nell'SCM
- [x] **Fase 2:** calcolare CCS sul baseline XGBoost come punto di riferimento
- [ ] **Fase 2-bis:** quantificare esplicitamente il bias Cas-OFFinder (analisi separata positivi/negativi e per assay)
- [ ] **Fase 2-bis:** ripetere CI tests con campionamento negativo controllato o reweighting
- [ ] **Fase 3:** verificare se il Neural SCM risolve il calo cross-assay (F1)

### F6.6 — Confronto: default DAG vs variante con arco mismatch_rate → label

La modifica del DAG che aggiunge l'arco diretto `mismatch_rate → label` non porta benefici rilevanti sugli esiti sperimentali. I segnali principali sono i seguenti:

- Il CI failure rate scende da 0.889 a 0.875 — un miglioramento di 1.4 punti percentuali su un failure rate dell'87%. Non è statisticamente o praticament rilevante.
- Il delta osservazionale vs interventale cambia di circa 0.0003 — compatibile con rumore statistico.
- L'aspetto più notevole è che il fit per la variante assegna `activity_eta_mismatch ≈ -30`, ovvero un peso estremamente grande a `mismatch_rate`.

Questa grandezza dei coefficienti è coerente con un problema di collinearità: `mismatch_rate` è una funzione di `node_B_proximal` e `node_C_seed_extension`, quindi l'aggiunta dell'arco introduce ridondanza informativa che il fitting compensa con coefficienti estremi invece di migliorare la rappresentazione causalmente.

Conclusione: la variante non migliora la qualità causale o predittiva degli esperimenti. Il DAG originale (senza l'arco diretto `mismatch_rate → label`) è preferibile per parsimonia: cattura l'informazione necessaria tramite i nodi esistenti e evita ridondanze che portano a stime instabili. Non portare questa modifica in Fase 3; il risultato è comunque un finding utile che conferma come i segnali evidenziati dai test di indipendenza siano riconducibili a informazioni ridondanti nei nodi già presenti, non alla necessità di un arco esplicito.

---

## Fase 3 — Neural SCM: deep complesso vs bypass lineare

### F7 — La complessità deep induce overfitting severo; il bypass lineare semplificato migliora la generalizzazione

**Esperimento:** `Exp04_LinearBypass_HardPrior` (encoder biologico + bypass lineare delle regioni mismatch, training con Focal Loss, OneCycleLR, 5 epoche).

**Setup rilevante:**

- split: train=2,925,972 / val=842,659 / test=582,999
- imbalance train: `pos_weight = 46.3`
- device: CUDA
- encoder: `BiologicalMismatchEncoder (embed_dim=12)`
- combinatore strutturale con pochi gradi di libertà (assetto semplificato):
  - `w_proximal`, `w_seed`, `w_nonseed`, `bias` + gate PAM

**Risultati Exp04:**

| Split | AUPRC | AUROC | F1 |
|---|---:|---:|---:|
| CHANGE-seq train | 0.2362 | 0.8834 | 0.0005 |
| CHANGE-seq val | 0.0469 | 0.7853 | 0.0014 |
| CHANGE-seq test | 0.0878 | 0.7982 | 0.0009 |
| GUIDE-seq cross-assay | 0.1250 | 0.8722 | 0.0094 |

**Causal Consistency:**

- `Neural CCS_Overall = 0.8333` (decisamente migliore rispetto ai baseline non causali)

**Confronto con i run deep precedenti (prima del bypass lineare):**

- nei run con maggiore componente deep si osservava pattern di overfitting severo, con train AUPRC molto alto e validation quasi collassata (ordine di grandezza ~`0.89` train vs ~`0.002` val)
- con il modello semplificato il gap train/val resta presente ma si riduce in modo sostanziale; la validazione torna su valori non degeneri (`~0.047`)

**Interpretazione:**

- la porzione deep ad alta capacità nella catena causale tendeva a catturare correlazioni spurie assay-specifiche
- il ritorno a una forma più parsimoniosa (bypass lineare con hard priors) riduce la varianza del modello e migliora la robustezza out-of-sample
- il miglioramento di CCS suggerisce che la semplificazione non solo aiuta la generalizzazione predittiva, ma preserva meglio la coerenza causale desiderata

**Implicazione per la tesi:**

- in questo dominio, una parametrizzazione causale semplice e ben vincolata è preferibile a una componente deep più espressiva ma instabile
- la complessità architetturale va introdotta solo se produce un guadagno netto e stabile su validation/cross-assay, non solo su train
