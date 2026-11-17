"""Textes de publication par réseau : description courte + hashtags adaptés à chaque plateforme.

Avec Ollama : l'IA écrit les textes. Sans IA : mots-clés du passage (comparés au reste de la vidéo)
+ hashtags de découverte propres à chaque réseau + description construite sur l'accroche du clip.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter

NETWORKS = ("tiktok", "youtube", "instagram")

# Limites et usages de chaque plateforme
RULES = {
    "tiktok":    {"n_tags": 5, "desc_max": 150},
    "youtube":   {"n_tags": 3, "desc_max": 300, "title_max": 95},
    "instagram": {"n_tags": 8, "desc_max": 200},
}

DISCOVERY = {
    "fr": {
        "tiktok": ["#pourtoi", "#fyp", "#viral"],
        "youtube": ["#shorts"],
        "instagram": ["#reels", "#reelsfrance", "#explore", "#pourtoi"],
    },
    "en": {
        "tiktok": ["#fyp", "#foryou", "#viral"],
        "youtube": ["#shorts"],
        "instagram": ["#reels", "#explore", "#reelsviral"],
    },
}

CTA = {
    "fr": {
        "tiktok": ["Abonne-toi pour la suite 👀", "T'en penses quoi ? Dis-le en commentaire 👇", "Partage à quelqu'un qui doit voir ça"],
        "youtube": ["Abonne-toi pour plus de moments comme ça.", "Tu es d'accord ? Dis-le en commentaire."],
        "instagram": ["Enregistre ce reel pour plus tard 📌", "Envoie-le à quelqu'un qui doit l'entendre", "Ton avis en commentaire 👇"],
    },
    "en": {
        "tiktok": ["Follow for part 2 👀", "What do you think? Comment below 👇", "Send this to someone who needs it"],
        "youtube": ["Subscribe for more moments like this.", "Agree? Let me know in the comments."],
        "instagram": ["Save this for later 📌", "Send this to someone who needs to hear it", "Thoughts? 👇"],
    },
}

STOP = set("""
a à ai aie ainsi alors après au aucun aussi autre aux avais avait avant avec avoir bah ben bien bon c ça car ce ceci cela celle celles celui cependant ces cet cette ceux chaque chez ci comme comment d dans de des deux dit dire doit donc dont du elle elles en encore est et étaient était étant été être eu eux fait faire fais faut fois font gens genre ici il ils j je juste l la là le les leur leurs lui m ma mais me même mes moi moins mon n ne ni non nos notre nous on ont ou où oui par parce pas peu peut peux plus pour pourquoi quand que quel quelle quelque quelques qu qui quoi rien s sa sans se ses si sinon soit son sont sous suis sur t ta te tes toi ton tous tout toute toutes très trop tu un une vais vas va vers veux voilà vos votre vous vraiment y euh hein voila truc trucs choses chose fois mec ouais okay ok enfin bref déjà alors puis après avant
the a an and or but so because if then than that this these those there here is are was were be been being have has had do does did done to of in on at for with from by as it its it's i you he she we they me him her us them my your his our their what which who whom why how when where not no yes yeah just really very like get got go going gonna want know think say said thing things people one some any all more most much many also well right now okay oh um uh
""".split())


VERB_ENDINGS = ("er", "ez", "ais", "ait", "aient", "ons", "ent", "ais", "iez", "erai", "erait", "é", "ée", "és", "ant",
                "ing", "ed")
STOP |= set("""regarde regarder rappelle raconte raconter parle parler sais savais savoir chacun chacune
veut voulu voir vois vu venir viens vient prendre prends pris mettre mets mis aller allez allons dois devoir
pense penser crois croire trouve trouver passe passer donne donner arrive arriver commence laisse laisser
montre montrer demande demander tiens tenir appelle reste rester sens sentir connais connaître ouvre
quelqu'un toujours jamais beaucoup partout maintenant aujourd'hui demain hier vraiment complètement
divertissant youtube vidéo video vidéos abonnez abonne like likes chaîne""".split())


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _tag(word: str) -> str:
    w = _strip_accents(word.lower())
    w = re.sub(r"[^a-z0-9]", "", w)
    return f"#{w}" if len(w) >= 3 else ""


def keywords(clip_text: str, full_text: str, k: int = 8) -> list[str]:
    """Mots les plus caractéristiques du passage par rapport au reste de la vidéo (type TF-IDF)."""
    tok = lambda t: [w for w in re.findall(r"[a-zA-Zàâäçéèêëîïôöûùüÿœæ'-]{4,}", t.lower())
                     if w.strip("'-") not in STOP and not w.startswith(("j'", "l'", "d'", "qu'", "c'", "n'", "s'"))]
    local = Counter(tok(clip_text))
    glob = Counter(tok(full_text)) if full_text else Counter()
    total_g = sum(glob.values()) or 1
    def verbish(w: str) -> float:  # les noms font de meilleurs hashtags que les verbes
        return 0.35 if w.endswith(VERB_ENDINGS) else 1.0
    scored = {w: c * (1.0 / (1 + glob.get(w, 0) / total_g * 400)) * (1 + 0.15 * len(w)) * verbish(w)
              for w, c in local.items()}
    out, seen = [], set()
    for w in sorted(scored, key=lambda w: -scored[w]):
        t = _tag(w)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= k:
            break
    return out


def _first_sentence(text: str, max_len: int) -> str:
    s = re.split(r"(?<=[.!?…])\s", text.strip())[0].strip()
    if len(s) > max_len:
        s = s[: max_len - 1].rsplit(" ", 1)[0] + "…"
    return s


def heuristic_copy(title: str, clip_text: str, full_text: str, lang: str, seed: int = 0) -> dict:
    lang = "fr" if lang == "fr" else "en" if lang == "en" else ("fr" if lang is None else "en")
    kws = keywords(clip_text, full_text)
    out = {}
    for net in NETWORKS:
        r = RULES[net]
        disc = DISCOVERY[lang][net]
        topic = kws[: max(1, r["n_tags"] - len(disc))]
        tags = (topic + [d for d in disc if d not in topic])[: r["n_tags"]] if net != "youtube" else (disc + topic)[: r["n_tags"]]
        cta = CTA[lang][net][seed % len(CTA[lang][net])]
        hook = title.strip() if title else _first_sentence(clip_text, 90)
        if net == "youtube":
            sentences = [x.strip() for x in re.split(r"(?<=[.!?…])\s", clip_text.strip()) if x.strip()]
            quote = next((x for x in sentences if x.lower() != hook.lower() and len(x) > 25), "")
            quote = quote if len(quote) <= 160 else quote[:159].rsplit(" ", 1)[0] + "…"
            desc = f"{quote}\n\n{cta}" if quote else cta
            out[net] = {"title": (hook[: r["title_max"] - 8] + " #shorts").strip(), "description": desc[: r["desc_max"]],
                        "hashtags": tags}
        else:
            desc = f"{hook} {cta}"
            out[net] = {"description": desc[: r["desc_max"]], "hashtags": tags}
    return out


LLM_PROMPT = """Tu es social media manager expert en contenus courts viraux.
Voici la transcription d'un clip vidéo (langue : {lang}). Titre provisoire : « {title} ».

{text}

Écris les textes de publication pour chaque réseau, dans la langue du clip :
- tiktok : description de 1 phrase percutante (max 150 caractères, peut finir par une question ou un appel à l'action), 4 à 5 hashtags (mélange niche du sujet + découverte comme #pourtoi/#fyp).
- youtube : titre de Short accrocheur (max 70 caractères), description de 1 à 2 phrases, 3 hashtags dont #shorts.
- instagram : légende de 1 à 2 phrases avec un appel à l'action (enregistrer/partager), 6 à 8 hashtags (niche + moyens + #reels).
Pas de hashtags inventés hors sujet. Hashtags sans espaces ni accents.

Réponds UNIQUEMENT en JSON :
{{"tiktok": {{"description": "...", "hashtags": ["#..."]}},
  "youtube": {{"title": "...", "description": "...", "hashtags": ["#shorts", "#..."]}},
  "instagram": {{"description": "...", "hashtags": ["#..."]}}}}"""


def _clean_tags(tags, n: int) -> list[str]:
    out = []
    for t in tags if isinstance(tags, list) else []:
        t = _tag(str(t).lstrip("#"))
        if t and t not in out:
            out.append(t)
    return out[:n]


def generate(title: str, clip_text: str, full_text: str, lang: str | None, llm_model: str | None = None,
             seed: int = 0) -> dict:
    fix = lambda t: re.sub(r"\s+(['’])", r"\1", t or "")
    title, clip_text = fix(title), fix(clip_text)
    base = heuristic_copy(title, clip_text, full_text, lang or "fr", seed)
    if not llm_model:
        return base
    try:
        from .pipeline import _ollama
        data = _ollama(llm_model, LLM_PROMPT.format(lang=lang or "auto", title=title, text=clip_text[:3500]))
        for net in NETWORKS:
            d = data.get(net) if isinstance(data, dict) else None
            if not isinstance(d, dict):
                continue
            r = RULES[net]
            tags = _clean_tags(d.get("hashtags"), r["n_tags"] + 1)
            if net == "youtube" and "#shorts" not in tags:
                tags = ["#shorts", *tags][: r["n_tags"]]
            if d.get("description"):
                base[net]["description"] = str(d["description"]).strip()[: r["desc_max"]]
            if tags:
                base[net]["hashtags"] = tags
            if net == "youtube" and d.get("title"):
                base[net]["title"] = str(d["title"]).strip()[: r["title_max"]]
    except Exception as e:  # l'IA a échoué : on garde la version automatique
        print(f"[social] IA indisponible : {e}")
    return base


def caption(copy: dict, net: str) -> str:
    """Texte final à coller / envoyer : description + hashtags."""
    d = copy.get(net, {})
    return f"{d.get('description', '').strip()}\n\n{' '.join(d.get('hashtags', []))}".strip()
