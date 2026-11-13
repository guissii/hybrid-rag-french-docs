# Jeu d'évaluation

`questions.jsonl` : une question par ligne (JSON). Le fichier fourni est un **gabarit non
étiqueté** (25 lignes `TODO`) : `make eval` refuse de produire des chiffres tant qu'il n'est
pas complété.

```json
{"id": "q01", "question": "Quel est le délai de paiement ?",
 "relevant": [
   {"doc": "ccag_travaux_2021.pdf", "pages": [31, 32]},
   {"doc": "reglement.docx", "evidence": "extrait copié tel quel depuis un seul paragraphe"}
 ]}
```

- `doc` : nom du fichier dans `data/raw/` (tel qu'il apparaît dans `data/processed/`).
- `pages` : pages (PDF) contenant la réponse. Un chunk est pertinent s'il partage une page.
- `evidence` : passage verbatim (DOCX n'ont pas de pages ; utilisable aussi pour les PDF).
  Il doit tenir dans **un seul** paragraphe/tableau extrait par Docling. Un chunk est
  pertinent s'il couvre cet élément.
- Les étiquettes ne dépendent pas du chunking : un même fichier sert à toutes les stratégies.
- Au moins 20 questions, toutes étiquetées. Rédigez-les sans copier les mots exacts du
  passage, sinon BM25 est avantagé artificiellement.
- Chaque `doc`/`page`/`evidence` est validé contre le corpus extrait avant tout calcul.
