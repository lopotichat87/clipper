# Clipper : générateur de clips en local

Tu colles un lien YouTube (ou tu importes une vidéo) et Clipper :
1. télécharge la vidéo (jusqu'en 1440p pour des clips nets),
2. la transcrit mot par mot (audio nettoyé, filtre anti-hallucinations, vocabulaire personnalisé),
3. choisit les meilleurs passages (IA locale Ollama si elle est lancée, sinon un score automatique),
4. recadre en vertical en suivant la personne qui parle,
5. ajoute les sous-titres animés (6 styles, entièrement réglables),
6. exporte en haute qualité (x264, son normalisé à -14 LUFS),
7. écrit une description et des hashtags adaptés à TikTok, YouTube Shorts et Instagram,
8. peut publier automatiquement chaque jour sur YouTube, TikTok et Instagram.

Tout tourne sur ton Mac, sans abonnement ni quota de génération.

## Utilisation
| Fichier | Rôle |
|---|---|
| `lancer.command` | Démarre Clipper en arrière-plan et ouvre l'interface. Tu peux ensuite fermer le terminal et le navigateur. |
| `arreter.command` | Arrête Clipper. |
| `demarrage-auto.command` | Lance Clipper automatiquement à chaque ouverture de session (nécessaire pour publier chaque jour sans y penser). Relance-le pour désactiver. |

L'interface se trouve sur http://127.0.0.1:7860. Les clips sont rangés dans `clips/`, et le journal du serveur dans `logs/server.log`.

## Publication automatique
Page **Automatisation** : chaque réseau a un guide pas à pas pour créer tes identifiants d'API (une seule fois), puis un bouton « Connecter mon compte ».
- **YouTube :** projet Google Cloud + YouTube Data API v3. Pense à passer l'écran de consentement en production, sinon la connexion expire au bout de 7 jours.
- **TikTok :** app sur developers.tiktok.com (Login Kit + Content Posting API). Tant que l'app n'a pas passé l'audit de TikTok, les vidéos sont publiées en **privé**.
- **Instagram :** compte professionnel relié à une Page Facebook + app Meta (Facebook Login for Business). En mode Développement, tu peux publier sur ton propre compte.

Les identifiants et les jetons restent sur ce Mac, dans `config/` (accès limité à ton compte).
Le Mac doit être allumé avec Clipper lancé : un créneau manqué pendant la veille est rattrapé dans la journée.

## Améliorer la sélection et les textes avec une IA locale (optionnel)
```bash
brew install ollama
```
```bash
ollama serve
```
```bash
ollama pull qwen2.5:3b
```
Avec 8 Go de RAM, reste sur un modèle 3B. Il faut environ 2 Go de disque.

## Espace disque
- `travail/` contient les vidéos sources et les transcriptions, gardées pour pouvoir retoucher les clips. Tu peux le vider pour récupérer de la place.
- La transcription « Maximale » télécharge un modèle de 1,6 Go la première fois.

N'utilise l'outil que sur des vidéos dont tu as les droits.
