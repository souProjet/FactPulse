# FactPulse — MVP Fact-Checking en Temps Réel (Texte)

## 🚀 Description
FactPulse est un MVP de vérification automatique des faits pour navigateur.  
Il détecte les **affirmations factuelles** dans le texte des pages web (articles, réseaux sociaux, descriptions YouTube) et vérifie uniquement **si nécessaire**.  
Le verdict est rendu en moins de 2 secondes, avec sources et justification, pour fournir une **alerte rapide et fiable** à l’utilisateur.

---

## ✅ Fonctionnalités du MVP V1
- Détection de claims factuels dans le texte  
- Ignorer les opinions et les phrases non vérifiables  
- Vérification rapide via **base locale de faits connus**  
- Vérification avancée via **RAG local + LLM** (source-grounded)  
- Verdict simple : `TRUE / FALSE / NOT_VERIFIABLE`  
- Affichage clair côté utilisateur via **badge et popup**  
- Mesure et log des performances à chaque étape  

---

## 🏗 Architecture
```
[Browser Extension]
│
▼
[FastAPI Backend]
│
▼
Claim Detection → Fast Local Lookup → RAG Verification (Phi-3)
│
▼
Verdict JSON + Sources
```

---

## ⚙️ Stack Technique

| Composant | Technologie |
|-----------|------------|
| Backend | Python 3.11, FastAPI |
| GPU / IA | PyTorch + CUDA, Phi-3 (quantisé), sentence-transformers, FAISS |
| Frontend Extension | Vanilla JS (Chrome/Firefox) |
| Base de données | JSON / SQLite pour fast path, FAISS pour embeddings |
| Cache / Performance | Redis (optionnel) |

---

## ⏱ Objectifs de Performance

| Étape pipeline | Latence cible |
|----------------|---------------|
| Extraction texte | < 20 ms |
| Claim detection | < 100 ms |
| Fast local lookup | < 300 ms |
| RAG / LLM | < 1.5 s |
| **Total** | < 2 s |

---

## 📊 Benchmarks

- Précision Claim Detection : ≥ 85%  
- Faux positifs : < 10%  
- Couverture claims viraux : ≥ 70%  
- Mesurable via `scripts/benchmark.py` sur dataset sample

---

## ⚠️ Limitations

- Texte uniquement (pas audio / vidéo)  
- Pas 100% de certitude — le MVP détecte le probable faux / douteux  
- Optimisé pour vitesse et fiabilité, pas pour exhaustivité  

---

## 🔮 Roadmap Futur

- Audio transcription (podcasts, vidéos)  
- Vidéo et contenu multimédia  
- Fact-checking en live (streams, débats)  
- Application mobile / extension universelle  

---

## 🧪 Installation & Setup rapide

1. Installer Python 3.11 et CUDA (RTX 5060)  
2. Installer dépendances :
```bash
pip install fastapi uvicorn torch torchvision torchaudio sentence-transformers faiss-cpu
```

3. Lancer backend FastAPI :

```
uvicorn backend.api:app --reload
```

4. Charger l’extension navigateur (`extension/`)

5. Tester avec le `dataset data/claims_detection.jsonl` et `data/fact_check_benchmark.json`

6. Lancer benchmarks :
```
python scripts/benchmark.py
```

## 📖 Usage

- Ouvrir n’importe quelle page web

- L’extension analyse le texte et envoie au backend

- Badge couleur :

    - 🟢 Rien → tout OK

    - 🟠 Vérification en cours

    - 🔴 Douteux / faux

- Popup → détails + sources

## 🔐 Règle d’or

>FactPulse n’est pas un juge absolu de la vérité.
Il s’agit d’un radar rapide de contenus suspects, transparent et sourcé.

## 📂 Structure du Repo
```
/factpulse
├── backend/
│   ├── api.py
│   ├── claim_detector/
│   ├── fact_checker/
│   ├── rag/
│   └── benchmarks/
├── models/
│   ├── claim_model/
│   ├── embedding_model/
│   └── llm/
├── data/
│   ├── claims_detection.jsonl
│   └── fact_check_benchmark.json
├── extension/
│   ├── content.js
│   ├── popup.html
│   └── popup.js
└── scripts/
    ├── build_index.py
    ├── benchmark.py
    └── load_data.py
```

## 📌 Notes

- Tous les pipelines mesurent la latence et l’utilisation GPU

- Timeout global = 2 secondes

- Le LLM Phi-3 est quantisé pour tourner sur RTX 5060

- Fast path prioritaire pour claims connus pour réduire la latence