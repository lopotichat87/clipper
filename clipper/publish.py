"""Publication automatique sur YouTube (Shorts) et TikTok.

- Les identifiants d'API (client id / secret) et les jetons de connexion restent sur ce Mac,
  dans config/ (fichiers lisibles uniquement par ton compte).
- La file d'attente est publiée par le planificateur aux heures choisies, un clip par créneau.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
import traceback
import urllib.parse
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CONF = ROOT / "config"
CONF.mkdir(exist_ok=True)
os.chmod(CONF, 0o700)

PORT = 7860
LOCK = threading.RLock()
GRAPH = "https://graph.facebook.com/v27.0"   # API Meta (Instagram via connexion Facebook)
NETWORKS = ("youtube", "tiktok", "instagram")
# Relais https (site Vercel) : TikTok et Meta exigent une adresse de retour https ;
# la page renvoie ensuite vers Clipper sur cet ordinateur (localhost:7860).
RELAY = os.environ.get("CLIPPER_RELAY", "https://clipper-autopost.vercel.app")


def _read(name: str, default):
    f = CONF / name
    try:
        return json.loads(f.read_text())
    except Exception:
        return default


def _write(name: str, data) -> None:
    f = CONF / name
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(f)


# ---------------------------------------------------------------------------
# Comptes
# ---------------------------------------------------------------------------

PROVIDERS = {
    "youtube": {
        "label": "YouTube",
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "scopes": "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly",
        "default_redirect": f"http://127.0.0.1:{PORT}/oauth/youtube/callback",
    },
    "tiktok": {
        "label": "TikTok",
        "auth_url": "https://www.tiktok.com/v2/auth/authorize/",
        "token_url": "https://open.tiktokapis.com/v2/oauth/token/",
        "scopes": "user.info.basic,video.publish",
        "default_redirect": f"{RELAY}/oauth/tiktok/callback",
    },
    "instagram": {
        "label": "Instagram",
        "auth_url": "https://www.facebook.com/v27.0/dialog/oauth",
        "token_url": f"{GRAPH}/oauth/access_token",
        "scopes": "instagram_basic,instagram_content_publish,pages_show_list,pages_read_engagement,business_management",
        "default_redirect": f"{RELAY}/oauth/instagram/callback",
    },
}

_pending_states: dict[str, str] = {}


def accounts_public() -> dict:
    creds, tokens = _read("credentials.json", {}), _read("tokens.json", {})
    out = {}
    for p, meta in PROVIDERS.items():
        c, t = creds.get(p, {}), tokens.get(p, {})
        out[p] = {"label": meta["label"], "configured": bool(c.get("client_id") and c.get("client_secret")),
                  "connected": bool(t.get("refresh_token") or t.get("access_token")),
                  "account": t.get("account_name"), "redirect_uri": c.get("redirect_uri") or meta["default_redirect"],
                  "client_id_hint": (c.get("client_id", "")[:10] + "…") if c.get("client_id") else None,
                  "privacy_options": t.get("privacy_options")}
    return out


def save_credentials(provider: str, client_id: str, client_secret: str, redirect_uri: str | None) -> None:
    with LOCK:
        creds = _read("credentials.json", {})
        prev = creds.get(provider, {})
        creds[provider] = {"client_id": client_id.strip() or prev.get("client_id", ""),
                           "client_secret": client_secret.strip() or prev.get("client_secret", ""),
                           "redirect_uri": (redirect_uri or "").strip() or PROVIDERS[provider]["default_redirect"]}
        _write("credentials.json", creds)


def disconnect(provider: str) -> None:
    with LOCK:
        tokens = _read("tokens.json", {})
        tokens.pop(provider, None)
        _write("tokens.json", tokens)


def auth_url(provider: str) -> str:
    meta = PROVIDERS[provider]
    c = _read("credentials.json", {}).get(provider) or {}
    if not c.get("client_id"):
        raise ValueError("Ajoute d'abord les identifiants de l'application.")
    state = secrets.token_urlsafe(24)
    _pending_states[state] = provider
    redirect = c.get("redirect_uri") or meta["default_redirect"]
    if provider == "youtube":
        q = {"client_id": c["client_id"], "redirect_uri": redirect, "response_type": "code",
             "scope": meta["scopes"], "access_type": "offline", "prompt": "consent", "state": state,
             "include_granted_scopes": "true"}
    elif provider == "instagram":
        q = {"client_id": c["client_id"], "redirect_uri": redirect, "response_type": "code",
             "scope": meta["scopes"], "state": state}
    else:
        q = {"client_key": c["client_id"], "redirect_uri": redirect, "response_type": "code",
             "scope": meta["scopes"], "state": state}
    return f"{meta['auth_url']}?{urllib.parse.urlencode(q)}"


def finish_oauth(provider: str, code: str, state: str) -> None:
    if _pending_states.pop(state, None) != provider:
        raise ValueError("Session de connexion expirée, recommence.")
    meta = PROVIDERS[provider]
    c = _read("credentials.json", {})[provider]
    redirect = c.get("redirect_uri") or meta["default_redirect"]
    if provider == "instagram":
        _finish_instagram(c, code, redirect)
        return
    if provider == "youtube":
        r = httpx.post(meta["token_url"], data={"code": code, "client_id": c["client_id"],
                                                "client_secret": c["client_secret"], "redirect_uri": redirect,
                                                "grant_type": "authorization_code"}, timeout=30)
    else:
        r = httpx.post(meta["token_url"], data={"client_key": c["client_id"], "client_secret": c["client_secret"],
                                                "code": code, "grant_type": "authorization_code",
                                                "redirect_uri": redirect},
                       headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=30)
    data = r.json()
    if r.status_code >= 300 or "access_token" not in data:
        raise ValueError(f"Connexion refusée : {data.get('error_description') or data.get('error') or data}")
    tok = _store_token(provider, data)
    try:
        if provider == "youtube":
            info = httpx.get("https://www.googleapis.com/youtube/v3/channels", params={"part": "snippet", "mine": "true"},
                             headers={"Authorization": f"Bearer {tok['access_token']}"}, timeout=20).json()
            items = info.get("items") or []
            tok["account_name"] = items[0]["snippet"]["title"] if items else "Chaîne YouTube"
        else:
            info = httpx.get("https://open.tiktokapis.com/v2/user/info/", params={"fields": "display_name,username"},
                             headers={"Authorization": f"Bearer {tok['access_token']}"}, timeout=20).json()
            u = (info.get("data") or {}).get("user") or {}
            tok["account_name"] = u.get("display_name") or u.get("username") or "Compte TikTok"
            tok["privacy_options"] = tiktok_creator_info(tok["access_token"]).get("privacy_level_options")
    except Exception:
        traceback.print_exc()
    with LOCK:
        tokens = _read("tokens.json", {})
        tokens[provider] = tok
        _write("tokens.json", tokens)


def _finish_instagram(c: dict, code: str, redirect: str) -> None:
    """Connexion Facebook -> jeton longue durée (60 j) -> compte Instagram professionnel relié à une Page."""
    r = httpx.get(f"{GRAPH}/oauth/access_token", params={"client_id": c["client_id"], "client_secret": c["client_secret"],
                                                          "redirect_uri": redirect, "code": code}, timeout=30).json()
    if "access_token" not in r:
        raise ValueError(f"Connexion refusée : {(r.get('error') or {}).get('message') or r}")
    long = httpx.get(f"{GRAPH}/oauth/access_token", params={"grant_type": "fb_exchange_token", "client_id": c["client_id"],
                                                             "client_secret": c["client_secret"],
                                                             "fb_exchange_token": r["access_token"]}, timeout=30).json()
    token = long.get("access_token") or r["access_token"]
    pages = httpx.get(f"{GRAPH}/me/accounts", params={"fields": "name,instagram_business_account{id,username}",
                                                      "access_token": token}, timeout=30).json()
    ig = next((pg["instagram_business_account"] for pg in pages.get("data", []) if pg.get("instagram_business_account")), None)
    if not ig:
        raise ValueError("Aucun compte Instagram professionnel relié à une de tes Pages Facebook. "
                         "Passe ton compte Instagram en Créateur ou Entreprise et relie-le à une Page.")
    tok = {"access_token": token, "expires_at": time.time() + int(long.get("expires_in", 60 * 86400)) - 3600,
           "ig_user_id": ig["id"], "account_name": f"@{ig.get('username', '')}"}
    with LOCK:
        tokens = _read("tokens.json", {})
        tokens["instagram"] = tok
        _write("tokens.json", tokens)


def _store_token(provider: str, data: dict) -> dict:
    with LOCK:
        tokens = _read("tokens.json", {})
        prev = tokens.get(provider, {})
        tok = {**prev, "access_token": data["access_token"],
               "refresh_token": data.get("refresh_token") or prev.get("refresh_token"),
               "expires_at": time.time() + int(data.get("expires_in", 3600)) - 60}
        if provider == "tiktok":
            tok["open_id"] = data.get("open_id") or prev.get("open_id")
        tokens[provider] = tok
        _write("tokens.json", tokens)
        return tok


def access_token(provider: str) -> str:
    tokens = _read("tokens.json", {})
    tok = tokens.get(provider)
    if not tok:
        raise RuntimeError(f"{PROVIDERS[provider]['label']} n'est pas connecté.")
    if provider == "instagram":
        # jeton de 60 jours : on le prolonge quand il reste moins de 10 jours
        if tok.get("expires_at", 0) - time.time() < 10 * 86400:
            c = _read("credentials.json", {})["instagram"]
            r = httpx.get(f"{GRAPH}/oauth/access_token", params={"grant_type": "fb_exchange_token", "client_id": c["client_id"],
                                                                  "client_secret": c["client_secret"],
                                                                  "fb_exchange_token": tok["access_token"]}, timeout=30).json()
            if "access_token" in r:
                tok.update(access_token=r["access_token"], expires_at=time.time() + int(r.get("expires_in", 60 * 86400)) - 3600)
                with LOCK:
                    tokens = _read("tokens.json", {})
                    tokens["instagram"] = tok
                    _write("tokens.json", tokens)
            elif tok.get("expires_at", 0) < time.time():
                raise RuntimeError("Reconnecte Instagram (jeton expiré).")
        return tok["access_token"]
    if tok.get("expires_at", 0) > time.time():
        return tok["access_token"]
    if not tok.get("refresh_token"):
        raise RuntimeError(f"Reconnecte {PROVIDERS[provider]['label']} (jeton expiré).")
    c = _read("credentials.json", {})[provider]
    if provider == "youtube":
        r = httpx.post(PROVIDERS[provider]["token_url"], data={"client_id": c["client_id"], "client_secret": c["client_secret"],
                                                               "refresh_token": tok["refresh_token"], "grant_type": "refresh_token"}, timeout=30)
    else:
        r = httpx.post(PROVIDERS[provider]["token_url"], data={"client_key": c["client_id"], "client_secret": c["client_secret"],
                                                               "refresh_token": tok["refresh_token"], "grant_type": "refresh_token"},
                       headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=30)
    data = r.json()
    if "access_token" not in data:
        raise RuntimeError(f"Reconnecte {PROVIDERS[provider]['label']} : {data.get('error_description') or data.get('error')}")
    return _store_token(provider, data)["access_token"]


# ---------------------------------------------------------------------------
# Envoi
# ---------------------------------------------------------------------------

def upload_youtube(path: str, title: str, description: str, tags: list[str], privacy: str = "public") -> dict:
    token = access_token("youtube")
    size = Path(path).stat().st_size
    meta = {"snippet": {"title": title[:100], "description": description[:4900],
                        "tags": [t.lstrip("#") for t in tags][:15], "categoryId": "22"},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}}
    r = httpx.post("https://www.googleapis.com/upload/youtube/v3/videos",
                   params={"uploadType": "resumable", "part": "snippet,status"},
                   headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8",
                            "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)},
                   json=meta, timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(f"YouTube a refusé l'envoi : {_err(r)}")
    with open(path, "rb") as f:
        up = httpx.put(r.headers["Location"], content=f.read(), headers={"Content-Type": "video/mp4"}, timeout=600)
    if up.status_code >= 300:
        raise RuntimeError(f"Envoi YouTube interrompu : {_err(up)}")
    vid = up.json().get("id")
    return {"id": vid, "url": f"https://youtube.com/shorts/{vid}"}


def tiktok_creator_info(token: str) -> dict:
    r = httpx.post("https://open.tiktokapis.com/v2/post/publish/creator_info/query/",
                   headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"},
                   timeout=30)
    return r.json().get("data") or {}


def upload_tiktok(path: str, caption: str, privacy: str = "SELF_ONLY") -> dict:
    token = access_token("tiktok")
    info = tiktok_creator_info(token)
    allowed = info.get("privacy_level_options") or ["SELF_ONLY"]
    if privacy not in allowed:
        privacy = "SELF_ONLY" if "SELF_ONLY" in allowed else allowed[0]
    size = Path(path).stat().st_size
    chunk = size if size <= 64 * 1024 * 1024 else 10 * 1024 * 1024
    n_chunks = 1 if chunk == size else size // chunk
    body = {"post_info": {"title": caption[:2200], "privacy_level": privacy,
                          "disable_duet": False, "disable_comment": False, "disable_stitch": False,
                          "video_cover_timestamp_ms": 1000},
            "source_info": {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": chunk,
                            "total_chunk_count": n_chunks}}
    r = httpx.post("https://open.tiktokapis.com/v2/post/publish/video/init/", json=body,
                   headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"},
                   timeout=60)
    data = r.json()
    if (data.get("error") or {}).get("code") not in (None, "ok"):
        raise RuntimeError(f"TikTok a refusé l'envoi : {data['error'].get('message') or data['error'].get('code')}")
    upload_url, publish_id = data["data"]["upload_url"], data["data"]["publish_id"]
    with open(path, "rb") as f:
        for i in range(n_chunks):
            start = i * chunk
            end = size - 1 if i == n_chunks - 1 else start + chunk - 1
            f.seek(start)
            part = f.read(end - start + 1)
            up = httpx.put(upload_url, content=part, timeout=600,
                           headers={"Content-Type": "video/mp4", "Content-Length": str(len(part)),
                                    "Content-Range": f"bytes {start}-{end}/{size}"})
            if up.status_code >= 300:
                raise RuntimeError(f"Envoi TikTok interrompu : {_err(up)}")
    # Attente du traitement par TikTok
    status = "PROCESSING_UPLOAD"
    for _ in range(60):
        time.sleep(5)
        s = httpx.post("https://open.tiktokapis.com/v2/post/publish/status/fetch/", json={"publish_id": publish_id},
                       headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"},
                       timeout=30).json()
        status = (s.get("data") or {}).get("status", status)
        if status == "PUBLISH_COMPLETE":
            break
        if status == "FAILED":
            raise RuntimeError(f"TikTok a rejeté la vidéo : {(s.get('data') or {}).get('fail_reason')}")
    return {"id": publish_id, "status": status, "privacy": privacy}


def upload_instagram(path: str, caption: str) -> dict:
    """Reel depuis un fichier local : conteneur « resumable », envoi des octets, attente du traitement, publication."""
    token = access_token("instagram")
    ig = _read("tokens.json", {})["instagram"]["ig_user_id"]
    r = httpx.post(f"{GRAPH}/{ig}/media", params={"access_token": token},
                   data={"media_type": "REELS", "upload_type": "resumable", "caption": caption[:2200],
                         "share_to_feed": "true"}, timeout=60).json()
    if "id" not in r:
        raise RuntimeError(f"Instagram a refusé l'envoi : {(r.get('error') or {}).get('message') or r}")
    container = r["id"]
    uri = r.get("uri") or f"https://rupload.facebook.com/ig-api-upload/v27.0/{container}"
    size = Path(path).stat().st_size
    with open(path, "rb") as f:
        up = httpx.post(uri, content=f.read(), timeout=900,
                        headers={"Authorization": f"OAuth {token}", "offset": "0", "file_size": str(size)})
    if up.status_code >= 300:
        raise RuntimeError(f"Envoi Instagram interrompu : {_err(up)}")
    status = "IN_PROGRESS"
    for _ in range(60):  # jusqu'à 10 min de traitement côté Instagram
        time.sleep(10)
        st = httpx.get(f"{GRAPH}/{container}", params={"fields": "status_code,status", "access_token": token},
                       timeout=30).json()
        status = st.get("status_code", status)
        if status == "FINISHED":
            break
        if status in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"Instagram a rejeté la vidéo : {st.get('status') or status}")
    if status != "FINISHED":
        raise RuntimeError("Instagram met trop de temps à traiter la vidéo, réessaie plus tard.")
    pub = httpx.post(f"{GRAPH}/{ig}/media_publish", params={"access_token": token},
                     data={"creation_id": container}, timeout=60).json()
    if "id" not in pub:
        raise RuntimeError(f"Publication Instagram refusée : {(pub.get('error') or {}).get('message') or pub}")
    link = httpx.get(f"{GRAPH}/{pub['id']}", params={"fields": "permalink", "access_token": token}, timeout=30).json()
    return {"id": pub["id"], "url": link.get("permalink")}


def _err(r: httpx.Response) -> str:
    try:
        j = r.json()
        e = j.get("error")
        if isinstance(e, dict):
            return e.get("message") or json.dumps(e)[:300]
        return str(e or j)[:300]
    except Exception:
        return r.text[:300]


# ---------------------------------------------------------------------------
# File d'attente et planification
# ---------------------------------------------------------------------------

DEFAULT_SCHEDULE = {
    "youtube": {"enabled": False, "times": ["18:00"], "privacy": "public"},
    "tiktok": {"enabled": False, "times": ["19:00"], "privacy": "SELF_ONLY"},
    "instagram": {"enabled": False, "times": ["12:00"], "privacy": "public"},
    "auto_queue": False,      # ajoute automatiquement les nouveaux clips à la file
    "auto_queue_min_score": 70,
    "auto_queue_networks": ["youtube", "tiktok", "instagram"],
}


def get_schedule() -> dict:
    s = _read("schedule.json", {})
    out = json.loads(json.dumps(DEFAULT_SCHEDULE))
    for k, v in s.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def save_schedule(data: dict) -> dict:
    cur = get_schedule()
    for net in NETWORKS:
        if net in data:
            d = data[net]
            times = sorted({t for t in d.get("times", cur[net]["times"]) if _valid_time(t)})
            cur[net].update({"enabled": bool(d.get("enabled", cur[net]["enabled"])), "times": times or ["18:00"],
                             "privacy": d.get("privacy", cur[net]["privacy"])})
    for k in ("auto_queue", "auto_queue_min_score", "auto_queue_networks"):
        if k in data:
            cur[k] = data[k]
    with LOCK:
        _write("schedule.json", cur)
    return cur


def _valid_time(t: str) -> bool:
    try:
        datetime.strptime(t, "%H:%M")
        return True
    except Exception:
        return False


def get_queue() -> list[dict]:
    return _read("queue.json", [])


def _save_queue(q: list[dict]) -> None:
    _write("queue.json", q)


def enqueue(items: list[dict]) -> list[dict]:
    """items : {job_id, clip_id, path, thumb, copy (textes par réseau), networks: [...]}"""
    with LOCK:
        q = get_queue()
        existing = {(i["job_id"], i["clip_id"]) for i in q}
        for it in items:
            if (it["job_id"], it["clip_id"]) in existing:
                for i in q:  # déjà dans la file : on ajoute les réseaux manquants
                    if (i["job_id"], i["clip_id"]) == (it["job_id"], it["clip_id"]):
                        for n in it["networks"]:
                            i["targets"].setdefault(n, {"status": "queued"})
                continue
            q.append({"id": uuid.uuid4().hex[:10], "job_id": it["job_id"], "clip_id": it["clip_id"],
                      "path": it["path"], "thumb": it.get("thumb"), "title": it.get("title", ""),
                      "copy": it.get("copy", {}), "added": time.time(),
                      "targets": {n: {"status": "queued"} for n in it["networks"]}})
        _save_queue(q)
        return q


def update_item(item_id: str, **kw) -> None:
    with LOCK:
        q = get_queue()
        for i in q:
            if i["id"] == item_id:
                for k, v in kw.items():
                    if k == "copy" and isinstance(v, dict):
                        i.setdefault("copy", {}).update(v)
                    else:
                        i[k] = v
        _save_queue(q)


def remove(item_id: str) -> None:
    with LOCK:
        _save_queue([i for i in get_queue() if i["id"] != item_id])


def move(item_id: str, delta: int) -> None:
    with LOCK:
        q = get_queue()
        idx = next((k for k, i in enumerate(q) if i["id"] == item_id), None)
        if idx is None:
            return
        j = max(0, min(len(q) - 1, idx + delta))
        q.insert(j, q.pop(idx))
        _save_queue(q)


def publish_item(item_id: str, network: str, notify=lambda *_: None) -> None:
    from .social import caption
    with LOCK:
        q = get_queue()
        item = next((i for i in q if i["id"] == item_id), None)
        if not item or network not in item["targets"]:
            return
        if item["targets"][network]["status"] in ("publishing", "published"):
            return
        item["targets"][network] = {"status": "publishing", "started": time.time()}
        _save_queue(q)
    sched = get_schedule()
    try:
        if not Path(item["path"]).exists():
            raise RuntimeError("Le fichier du clip n'existe plus.")
        copy = item.get("copy", {})
        if network == "youtube":
            yt = copy.get("youtube", {})
            desc = f"{yt.get('description', '')}\n\n{' '.join(yt.get('hashtags', []))}".strip()
            res = upload_youtube(item["path"], yt.get("title") or item["title"], desc, yt.get("hashtags", []),
                                 sched["youtube"]["privacy"])
        elif network == "instagram":
            res = upload_instagram(item["path"], caption(copy, "instagram"))
        else:
            res = upload_tiktok(item["path"], caption(copy, "tiktok"), sched["tiktok"]["privacy"])
        result = {"status": "published", "at": time.time(), **res}
        notify(f"Publié sur {PROVIDERS[network]['label']}", item["title"][:80])
    except Exception as e:
        traceback.print_exc()
        result = {"status": "error", "at": time.time(), "error": str(e)[:400]}
        notify(f"Échec de publication {PROVIDERS[network]['label']}", str(e)[:120])
    with LOCK:
        q = get_queue()
        for i in q:
            if i["id"] == item_id:
                i["targets"][network] = result
        _save_queue(q)


def next_slots(now: datetime | None = None) -> dict:
    """Prochaine heure de publication de chaque réseau (pour l'affichage)."""
    now = now or datetime.now()
    sched, state = get_schedule(), _read("scheduler_state.json", {})
    out = {}
    for net in NETWORKS:
        s = sched[net]
        if not s["enabled"]:
            out[net] = None
            continue
        cands = []
        for d in (0, 1):
            day = (now + timedelta(days=d)).date()
            for t in s["times"]:
                slot = datetime.combine(day, datetime.strptime(t, "%H:%M").time())
                key = slot.strftime("%Y-%m-%d %H:%M")
                if slot > now or (key not in state.get(net, []) and slot.date() == now.date()):
                    cands.append(slot)
        out[net] = min(cands).isoformat(timespec="minutes") if cands else None
    return out


class Scheduler(threading.Thread):
    """Toutes les 30 s : si un créneau du jour est passé et pas encore servi, publie le prochain clip de la file.
    Un créneau manqué (Mac éteint, en veille) est rattrapé dans la journée, jamais le lendemain."""

    def __init__(self, notify, keep_awake):
        super().__init__(daemon=True)
        self.notify, self.keep_awake = notify, keep_awake

    def run(self):
        while True:
            try:
                self.tick()
            except Exception:
                traceback.print_exc()
            time.sleep(30)

    def tick(self):
        now = datetime.now()
        sched = get_schedule()
        state = _read("scheduler_state.json", {})
        for net in NETWORKS:
            s = sched[net]
            if not s["enabled"] or not accounts_public()[net]["connected"]:
                continue
            done = state.setdefault(net, [])
            for t in s["times"]:
                slot = datetime.combine(now.date(), datetime.strptime(t, "%H:%M").time())
                key = slot.strftime("%Y-%m-%d %H:%M")
                if slot > now or key in done:
                    continue
                item = next((i for i in get_queue() if i["targets"].get(net, {}).get("status") == "queued"), None)
                done.append(key)
                state[net] = done[-60:]
                _write("scheduler_state.json", state)
                if item:
                    with self.keep_awake():
                        publish_item(item["id"], net, self.notify)
