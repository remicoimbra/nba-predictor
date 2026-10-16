---
name: pushage
description: Clôture de session — met à jour CLAUDE.md avec ce qui a été fait, commite puis pousse sur GitHub.
disable-model-invocation: true
---

# Clôture de session : CLAUDE.md, commit, push

Objectif : que le `CLAUDE.md` décrive fidèlement l'état du projet à la fin de
cette session, pour que la prochaine session (qui ne verra que ce fichier)
reparte sans rien redécouvrir.

## 1. Faire l'inventaire de la session

- `git status`, puis `git log --oneline` depuis le premier commit de la
  session, et `git diff` sur ce qui n'est pas encore commité.
- Ne décrire que ce qui a **réellement** été fait et vérifié. Une chose
  commencée mais pas terminée est notée comme telle, sans l'embellir.

## 2. Mettre à jour CLAUDE.md

Modifier les sections existantes plutôt que d'empiler un journal en fin de
fichier :

- **§6 (titre)** : date d'état du jour.
- **§6.1 Décisions actées** : une ligne par décision technique nouvelle, avec
  sa **raison** (c'est la raison qui évite de la remettre en cause plus tard).
- **§6.2 / §6.3 / §6.4** : seulement si le modèle, l'arborescence ou les
  routes ont changé.
- **§6.5 Commandes courantes** : nouvelles commandes, comptes de test, URL.
- **§6.6 État d'avancement** : réécrire les blocs « Fait », « Reste des
  squelettes » et « Prochaine étape ». Retirer ce qui est devenu faux.
- **§8 / §8.1** : cocher les points tranchés ; ajouter les questions
  ouvertes apparues pendant la session, ou l'arbitrage provisoire retenu.

Rester concis : le fichier est relu à chaque session. Ne pas y recopier ce que
le code ou l'historique Git disent déjà.

## 3. Commiter

- Un commit par sujet s'il reste du travail non commité, puis un commit
  dédié à `CLAUDE.md`.
- Message en français : un titre court à l'indicatif (« Ajoute… »,
  « Met à jour… »), une ligne vide, puis le détail par section ou par
  sujet.
- Ne jamais commiter de secret (`.env.local`, clés Stripe) ni les dossiers
  ignorés (`public/uploads/`, `var/`, `vendor/`).

## 4. Pousser sur GitHub

- Vérifier `git remote -v` et `gh auth status`.
- S'il n'y a pas de remote ou que `gh` n'est pas connecté : **s'arrêter et
  demander** (URL du dépôt existant, ou création d'un dépôt **privé** ;
  connexion via `! gh auth login`). Ne jamais créer un dépôt public sans
  accord explicite : c'est le code d'une cliente.
- Pousser la branche courante (`git push -u origin <branche>` au premier
  push). Jamais de `--force`.

## 5. Compte rendu

Terminer par un résumé court : ce qui a été ajouté au `CLAUDE.md`, les
commits poussés (hash + titre), et ce qui n'a pas pu être fait.
