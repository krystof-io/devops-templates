#!/usr/bin/env python3
"""Apply the local live-streaming patches to the AVTR-1 checkout at image build.

    python3 apply_patches.py /opt/avtr-1

Every replacement must match its target text exactly the stated number of times,
otherwise the build fails. That way an upstream change (a different commit, a
different aioice version) cannot leave a patch silently unapplied.

What it changes, and why:
  1. conversation_engines/builders.py   register the "local" engine
  2. local_stream_cli.py                /say and /transcript routes, /vendor route, advertise the
                                        LoadBalancer address in the SDP answer,
                                        longer idle timeout
  3. local_stream_ui.html               load Preact/htm from this server instead
                                        of the esm.sh CDN, add the Local engine
                                        tab, the text/transcript bar, and a
                                        phone layout (viewport, full-screen
                                        session view, keyboard handling)
  4. scripts/run_local_stream.py        renderer in the `renderer` pixi env,
                                        bound to 127.0.0.1 (it has no auth)
  4b. avtr1_renderer/pipeline.py and    per-avatar tuning (motion settings,
      avtr1_motion_generator.py         per-region guidance, gains) via
                                        src/kio_motion_tuning.py
  5. aioice/ice.py (streamer env)       bind WebRTC UDP sockets in a fixed port
                                        range, and wait for the browser's first
                                        packet instead of failing when none of
                                        its addresses could be resolved
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def patch(rel: str, edits: list[tuple[str, str, int]]) -> None:
    p = root / rel
    s = p.read_text(encoding="utf-8")
    for old, new, count in edits:
        found = s.count(old)
        if found != count:
            sys.exit(f"PATCH FAILED {rel}: expected {count} match(es), found {found}, for:\n{old}")
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    print(f"patched {rel} ({len(edits)} edit(s))")


# --- 1. engine registry ------------------------------------------------------
patch("src/avaturn_live_streamer/conversation_engines/builders.py", [
    (
        "from avaturn_live_streamer.event_bus import EventBus\n",
        "from avaturn_live_streamer.event_bus import EventBus\n"
        "from avaturn_live_streamer.conversation_engines.local_engine import (\n"
        "    LocalEngineOptions,\n"
        "    build_local,\n"
        ")\n",
        1,
    ),
    # Only the local engine can be selected. Upstream's two engines (OpenAI
    # Realtime, Cartesia) are cloud services; removing them from the accepted
    # options and not importing their clients means the `openai` package is never
    # loaded and a request naming either engine is rejected.
    (
        "EngineOptions = Annotated[\n"
        "    OpenAIEngineOptions | CartesiaEngineOptions,\n"
        "    Discriminator(\"type\"),\n"
        "]\n",
        "EngineOptions = LocalEngineOptions  # local change: cloud engines removed\n",
        1,
    ),
    (
        "from avaturn_live_streamer.conversation_engines.cartesia_client import CartesiaApiClient\n",
        "# local change: cartesia_client is not imported (cloud engine removed)\n",
        1,
    ),
    (
        "from avaturn_live_streamer.conversation_engines.realtime_api_client import RealtimeApiClient\n",
        "# local change: realtime_api_client is not imported (cloud engine removed; it pulls in `openai`)\n",
        1,
    ),
    (
        "        case CartesiaEngineOptions():\n"
        "            return await build_cartesia(stream_id=stream_id, options=options)\n",
        "        case CartesiaEngineOptions():\n"
        "            return await build_cartesia(stream_id=stream_id, options=options)\n"
        "        case LocalEngineOptions():\n"
        "            return await build_local(stream_id=stream_id, options=options)\n",
        1,
    ),
])

# --- 2. streamer web server ---------------------------------------------------
CLI_HELPERS = '''_UI_HTML_PATH = Path(__file__).parent / "local_stream_ui.html"

# ---- local additions (see devops-templates/Dockerfiles/avtr1/live) ----------
import os as _kio_os

import kio_required_env as _kio_required  # no defaults in the image; see that module
from fastapi.responses import Response as _KioResponse

_KIO_VENDOR_DIR = Path("/opt/avtr-1/vendor")
_KIO_VENDOR_FILES = {"preact.module.js", "hooks.module.js", "htm.module.js"}
# LAN-reachable address of the UDP LoadBalancer Service in front of this pod.
_KIO_PUBLIC_IP = _kio_required.text("KIO_WEBRTC_PUBLIC_IP").strip()


def _kio_rewrite_sdp(sdp: str) -> str:
    """Advertise the LoadBalancer address instead of the pod address.

    The pod's own address is not routable from the LAN. The Service forwards
    the same UDP port numbers, so only the address in each host candidate (and
    the c= lines) needs replacing. No-op when KIO_WEBRTC_PUBLIC_IP is unset.
    """
    if not _KIO_PUBLIC_IP:
        return sdp
    out: list[str] = []
    for line in sdp.splitlines():
        if line.startswith("a=candidate:") and " typ host" in line:
            parts = line.split(" ")
            if len(parts) > 5:
                parts[4] = _KIO_PUBLIC_IP
                line = " ".join(parts)
        elif line.startswith("c=IN IP4 ") and not line.startswith("c=IN IP4 0.0.0.0"):
            line = "c=IN IP4 " + _KIO_PUBLIC_IP
        out.append(line)
    return "\\r\\n".join(out) + "\\r\\n"


# Video codec to prefer when the browser offers several ("VP8", "H264" or ""
# for aiortc's default, which follows the browser's order). Desktop Chrome
# lists VP8 first; Safari lists H264 first. VP8 is the path proven here.
_KIO_VIDEO_CODEC = _kio_required.text("KIO_VIDEO_CODEC").strip()


def _kio_prefer_video_codec(sdp: str) -> str:
    """Move the preferred codec to the front of the offer's video line.

    aiortc answers in the browser's order and sends with the first codec, and it
    fixes that order while applying the offer, so the only place to express a
    preference is the offer text itself. Payload types are only reordered, never
    removed, so a browser that lacks the preferred codec is unaffected.
    """
    if not _KIO_VIDEO_CODEC:
        return sdp
    wanted = _KIO_VIDEO_CODEC.upper()
    lines = sdp.split("\\r\\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("m=video "):
            out.append(line)
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not lines[j].startswith("m="):
            j += 1
        section = lines[i + 1 : j]
        names: dict[str, str] = {}
        apt: dict[str, str] = {}
        for l in section:
            if l.startswith("a=rtpmap:"):
                pt, _, rest = l[len("a=rtpmap:") :].partition(" ")
                names[pt] = rest.split("/")[0].upper()
            elif l.startswith("a=fmtp:") and "apt=" in l:
                pt, _, rest = l[len("a=fmtp:") :].partition(" ")
                apt[pt] = rest.split("apt=")[1].split(";")[0].strip()
        head = line.split(" ")
        pts = head[3:]
        first = [pt for pt in pts if names.get(pt) == wanted]
        first_set = set(first)
        first += [pt for pt in pts if names.get(pt) == "RTX" and apt.get(pt) in first_set]
        if first:
            order = first + [pt for pt in pts if pt not in set(first)]
            line = " ".join(head[:3] + order)
            # aiortc takes the codec order from the a=rtpmap lines, not from the
            # m= line, so move each payload type's attribute lines as well.
            per_pt: dict[str, list[str]] = {}
            others: list[str] = []
            slot = None
            for l in section:
                key = None
                for prefix in ("a=rtpmap:", "a=fmtp:", "a=rtcp-fb:"):
                    if l.startswith(prefix):
                        key = l[len(prefix) :].split(" ")[0]
                        break
                if key is not None and key in set(pts):
                    per_pt.setdefault(key, []).append(l)
                    if slot is None:
                        slot = len(others)
                else:
                    others.append(l)
            if slot is not None:
                moved = [l for pt in order for l in per_pt.get(pt, [])]
                section = others[:slot] + moved + others[slot:]
        out.append(line)
        out.extend(section)
        i = j
    return "\\r\\n".join(out)


# Do not let the server start the WebRTC handshake towards the browser.
#
# aiortc begins sending connectivity checks to the browser's addresses as soon
# as it has the offer, before the browser has sent anything. Behind a Kubernetes
# LoadBalancer that is harmful: the node records the exchange as one the POD
# started, so every later packet to that browser - replies included - leaves
# under the node's own address instead of the LoadBalancer address. A firewall
# that (rightly) forbids the cluster from opening connections into private
# networks then drops all of it, and the session never connects. Seen
# 2026-11-22 from a phone on WireGuard: OPNsense passed phone -> 172.16.1.180
# and blocked 172.16.1.124 -> phone.
#
# With the browser's candidates removed from the offer the server has nothing to
# call, waits (aioice patch below, up to 20 s) for the browser's first packet,
# and from then on only ever replies - which leaves as the LoadBalancer address.
# This is the only mode; there is no setting for it.
def _kio_strip_candidates(sdp: str) -> str:
    sep = chr(13) + chr(10)
    return sep.join(
        line
        for line in sdp.split(sep)
        if not line.startswith("a=candidate:") and line != "a=end-of-candidates"
    )


class _KioClientLog(BaseModel):
    ua: str = ""
    events: list[dict] = []


class _KioSayBody(BaseModel):
    text: str
    interrupt: bool = False
    # "say": speak the text verbatim.  "ask": treat it as something the user said
    # and let the LLM answer.
    mode: Literal["say", "ask"] = "say"
'''

CLI_ROUTES = '''    app = FastAPI()
    slot = _SessionSlot()

    @app.get("/vendor/{name}")
    async def kio_vendor(name: str) -> _KioResponse:
        if name not in _KIO_VENDOR_FILES:
            raise HTTPException(status_code=404, detail="not found")
        return _KioResponse(
            content=(_KIO_VENDOR_DIR / name).read_bytes(),
            media_type="text/javascript",
            headers={"cache-control": "public, max-age=3600"},
        )

    @app.post("/say")
    async def kio_say(body: _KioSayBody) -> dict[str, object]:
        """Make the avatar speak the given text (local engine only)."""
        from avaturn_live_streamer.conversation_engines.local_engine import active_engine

        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is empty")
        if len(text) > 2000:
            raise HTTPException(status_code=400, detail="text too long (max 2000 characters)")
        engine = active_engine()
        if engine is None:
            raise HTTPException(status_code=409, detail="no active local-engine session")
        try:
            engine.enqueue(text, interrupt=body.interrupt, mode=body.mode)
        except asyncio.QueueFull:
            raise HTTPException(status_code=429, detail="speech queue is full") from None
        return {"queued": True, "chars": len(text), "mode": body.mode}

    @app.post("/clientlog")
    async def kio_clientlog(body: _KioClientLog) -> dict[str, bool]:
        """Browser-side diagnostics (what the page actually received and played).

        A phone has no console to read, so the page reports here and it lands in
        the pod log: `kubectl logs -n avtr1 deploy/avtr1 | grep "client report"`.
        """
        for ev in body.events[:20]:
            _LOGGER.info("client report", ua=body.ua[:160], **{"c_" + str(k)[:40]: v for k, v in list(ev.items())[:24]})
        return {"ok": True}

    @app.get("/avatar-settings")
    async def kio_settings_all() -> dict[str, object]:
        """Every avatar's saved tuning, with the schema that describes each setting.

        This is the export for anything else that wants the tuned values (for
        example the Hermes HUD): `avatars` holds only what differs from
        `defaults`; `schema` gives label, range and meaning of each key.
        """
        import kio_avatar_settings as ks

        return {"file": ks.settings_path(), "defaults": ks.DEFAULTS, "avatars": ks.load_all(max_age=0.0), "schema": ks.SCHEMA}

    @app.get("/avatar-settings/{avatar_id}")
    async def kio_settings_get(avatar_id: str) -> dict[str, object]:
        import kio_avatar_settings as ks

        ks.load_all(max_age=0.0)
        return {"avatar": avatar_id, "overrides": ks.overrides(avatar_id), "values": ks.effective(avatar_id)}

    @app.put("/avatar-settings/{avatar_id}")
    async def kio_settings_put(avatar_id: str, body: dict[str, object]) -> dict[str, object]:
        """Replace one avatar's saved tuning. Takes effect within a moment, mid-session."""
        import kio_avatar_settings as ks

        if not avatar_id or len(avatar_id) > 80 or not all(c.isalnum() or c in "-_" for c in avatar_id):
            raise HTTPException(status_code=400, detail="bad avatar id")
        values = body.get("values")
        if not isinstance(values, dict):
            raise HTTPException(status_code=400, detail="body must be an object with a 'values' object")
        try:
            saved = ks.save(avatar_id, values)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"could not write settings: {exc}") from exc
        _LOGGER.info("avatar settings saved", avatar=avatar_id, overrides=saved)
        return {"avatar": avatar_id, "overrides": saved, "values": ks.effective(avatar_id)}

    @app.get("/voices")
    async def kio_voices() -> dict[str, object]:
        """Voices the TTS server offers (asked live, cached briefly), for the tuning panel."""
        from avaturn_live_streamer.conversation_engines.local_engine import list_voices

        return await list_voices()

    @app.get("/transcript")
    async def kio_transcript() -> dict[str, object]:
        """What was heard and said recently, with per-turn timings (local engine)."""
        from avaturn_live_streamer.conversation_engines.local_engine import (
            active_engine,
            transcript_entries,
        )

        return {"active": active_engine() is not None, "entries": transcript_entries()}
'''

patch("src/avaturn_live_streamer/local_stream_cli.py", [
    ('_UI_HTML_PATH = Path(__file__).parent / "local_stream_ui.html"\n', CLI_HELPERS, 1),
    ("    app = FastAPI()\n    slot = _SessionSlot()\n", CLI_ROUTES, 1),
    # One session at a time, but the newest request wins: starting a session on
    # another device ends the one still running elsewhere instead of being
    # refused with "Another session is active". KIO_SESSION_TAKEOVER=0 restores
    # upstream's refusal.
    (
        "            if self._task is not None and not self._task.done():\n"
        "                raise HTTPException(status_code=409, detail=\"Another session is active\")\n",
        "            if self._task is not None and not self._task.done():\n"
        "                if not _kio_required.flag(\"KIO_SESSION_TAKEOVER\"):\n"
        "                    raise HTTPException(status_code=409, detail=\"Another session is active\")\n"
        "                _LOGGER.info(\"new session requested; ending the one in progress\")\n"
        "                previous = self._task\n"
        "                previous.cancel()\n"
        "                await asyncio.wait({previous}, timeout=10.0)\n"
        "                if not previous.done():\n"
        "                    raise HTTPException(status_code=409, detail=\"Previous session did not stop in time\")\n",
        1,
    ),
    (
        "        try:\n"
        "            built_engine = await build_engine(body.engine, stream_id=\"local\")\n",
        "        try:\n"
        "            from avaturn_live_streamer.conversation_engines.local_engine import set_session_avatar\n"
        "\n"
        "            set_session_avatar(body.avatar_id.strip())\n"
        "            built_engine = await build_engine(body.engine, stream_id=\"local\")\n",
        1,
    ),
    (
        '        return {"sdp": sdp_out, "type": local.type}\n',
        '        return {"sdp": _kio_rewrite_sdp(sdp_out), "type": local.type}\n',
        2,
    ),
    (
        "        await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))\n"
        "        answer = await pc.createAnswer()\n",
        "        await pc.setRemoteDescription(\n"
        "            RTCSessionDescription(\n"
        "                sdp=_kio_strip_candidates(_kio_prefer_video_codec(body.sdp)), type=body.type\n"
        "            )\n"
        "        )\n"
        "        answer = await pc.createAnswer()\n",
        1,
    ),
    (
        "        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))\n",
        "        await pc.setRemoteDescription(\n"
        "            RTCSessionDescription(sdp=_kio_strip_candidates(sdp), type=type_)\n"
        "        )\n",
        1,
    ),
    (
        "    cfg = get_config()\n    cfg.logging.level = log_level.value\n",
        "    _kio_required.check(\"streamer\")  # every setting must come from Kubernetes\n"
        "    cfg = get_config()\n    cfg.logging.level = log_level.value\n",
        1,
    ),
    (
        '    idle_timeout: Annotated[int, typer.Option(help="Idle timeout (s)")] = 30,\n',
        '    idle_timeout: Annotated[int, typer.Option(help="Idle timeout (s)")] = _kio_required.number(\n'
        '        "KIO_IDLE_TIMEOUT", int\n'
        "    ),\n",
        1,
    ),
])

# --- 2a. no STUN, no TURN ------------------------------------------------------
# Everything is on local networks and the browser is told exactly where to send
# media, so there is nothing for STUN or TURN to do. Upstream reads STUN_URL
# (defaulting to Google's public server), TURN_URL and Cloudflare TURN keys from
# the environment; this makes the function return no servers regardless, so none
# of those variables has any effect and nothing can re-enable them by accident.
patch("src/avaturn_live_streamer/localrtc/ice.py", [
    (
        "async def resolve_ice_servers() -> list[RTCIceServer]:\n"
        "    servers: list[RTCIceServer] = []\n",
        "async def resolve_ice_servers() -> list[RTCIceServer]:\n"
        "    return []  # local change: no STUN/TURN servers, ever (see apply_patches.py)\n"
        "    servers: list[RTCIceServer] = []\n",
        1,
    ),
])

# --- 2b. session start-up allowance ---------------------------------------------
# The session's workers must all report ready within this time or the session is
# torn down. Upstream allows 5 s, which a slow first handshake (mobile network,
# VPN) can exceed now that the server waits for the browser to make contact.
patch("src/avaturn_live_streamer/runner.py", [
    (
        "    bus = EventBus()\n",
        "    import kio_required_env as _kio_required\n"
        "    bus = EventBus(ready_timeout=_kio_required.number(\"KIO_READY_TIMEOUT\"))\n",
        1,
    ),
])

# --- 3. browser page ----------------------------------------------------------
MOBILE_CSS = '''
    /* ---- local additions: phone layout + the text/transcript bar ---- */
    :root { --kio-bar-h: 0px; }
    .app { padding-bottom: calc(var(--kio-bar-h) + 24px); }
    .modal-backdrop { padding-bottom: calc(var(--kio-bar-h) + 12px); }
    .modal { max-height: calc(100vh - 36px - var(--kio-bar-h)); max-height: calc(100dvh - 36px - var(--kio-bar-h)); }
    /* upstream sizes the video without knowing about the bar; keep the mic controls on screen */
    .video-wrap { max-height: calc(100vh - 230px - var(--kio-bar-h)); max-height: calc(100dvh - 230px - var(--kio-bar-h)); }

    .tabs { display: none; }  /* one engine only: nothing to choose between */

    #kio-say {
      position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000;
      padding: 8px 14px calc(10px + env(safe-area-inset-bottom, 0px));
      background: rgba(20, 20, 24, 0.96); border-top: 1px solid var(--border);
      font: 14px/1.35 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color: #ddd;
    }
    #kio-transcript { max-height: 8.5em; overflow-y: auto; margin-bottom: 6px; font-size: 13px; overscroll-behavior: contain; }
    #kio-transcript:empty { display: none; }
    .kio-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    #kio-say-text { flex: 1 1 220px; width: auto; min-width: 0; padding: 9px 10px; border-radius: 6px; border: 1px solid #555; background: #111; color: #eee; }
    .kio-opts { display: flex; gap: 12px; align-items: center; flex: 0 0 auto; }
    .kio-check { display: flex; align-items: center; gap: 5px; margin: 0; color: #bbb; font-size: 13px; font-weight: 400; text-transform: none; letter-spacing: 0; cursor: pointer; }
    .kio-check input { width: 18px; height: 18px; padding: 0; margin: 0; flex: 0 0 auto; }
    #kio-say-send { padding: 9px 16px; border-radius: 6px; border: 1px solid #666; background: #2a5; color: #fff; cursor: pointer; }
    #kio-say-status { color: #999; font-size: 12px; min-width: 4em; }

    /* tuning panel */
    #kio-tune-toggle { padding: 9px 12px; border-radius: 6px; border: 1px solid #666; background: #333; color: #eee; cursor: pointer; }
    #kio-tune-toggle.on { background: #46c; border-color: #68e; }
    #kio-tune { display: none; max-height: 46vh; overflow-y: auto; margin-bottom: 8px; padding: 2px 4px 8px 0; border-bottom: 1px solid var(--border); overscroll-behavior: contain; }
    #kio-tune.open { display: block; }
    .kio-tune-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 6px; }
    .kio-tune-head b { font-size: 14px; }
    .kio-tune-head .kio-grow { flex: 1 1 auto; color: #9a9; font-size: 12px; }
    .kio-btn { padding: 5px 10px; border-radius: 6px; border: 1px solid #555; background: #2a2d35; color: #ddd; cursor: pointer; font-size: 12px; }
    #kio-tune details { border: 1px solid var(--border); border-radius: 6px; margin-bottom: 6px; background: rgba(255,255,255,0.02); }
    #kio-tune summary { cursor: pointer; padding: 6px 10px; font-weight: 600; font-size: 13px; }
    #kio-tune summary .kio-count { color: #fc9; font-weight: 400; margin-left: 6px; }
    .kio-fields { padding: 2px 10px 8px; }
    .kio-field { display: grid; grid-template-columns: minmax(150px, 220px) 1fr 62px 28px; gap: 8px; align-items: center; padding: 3px 0; }
    .kio-field > span.kio-label { font-size: 13px; color: #ccc; }
    .kio-field.changed > span.kio-label { color: #fc9; }
    .kio-field input[type=range] { width: 100%; padding: 0; margin: 0; border: 0; background: transparent; accent-color: #6a9fff; }
    .kio-field input[type=number] { width: 62px; padding: 3px 4px; font-size: 13px; text-align: right; }
    .kio-field input[type=checkbox] { width: 18px; height: 18px; padding: 0; justify-self: start; }
    .kio-field input[type=text], .kio-field textarea, .kio-field select { grid-column: 2 / 4; width: 100%; padding: 5px 8px; font-size: 13px; }
    .kio-field textarea { min-height: 64px; }
    .kio-reset { width: 28px; height: 26px; padding: 0; border-radius: 5px; border: 1px solid #555; background: transparent; color: #aaa; cursor: pointer; visibility: hidden; }
    .kio-field.changed .kio-reset { visibility: visible; }
    #kio-tune-json { display: none; margin: 4px 0 6px; }
    #kio-tune-json.open { display: block; }
    #kio-tune-json pre { margin: 0 0 6px; padding: 8px; max-height: 14em; overflow: auto; background: #0c0d10; border: 1px solid var(--border); border-radius: 6px; font-size: 12px; white-space: pre-wrap; word-break: break-word; user-select: all; }

    /* Wide screens: tuning is a column on the right, the page and the session
       video keep the space on the left, so both are visible at once. */
    @media (min-width: 1000px) {
      :root { --kio-tune-w: 440px; }
      #kio-tune.open {
        position: fixed; top: 0; right: 0; bottom: var(--kio-bar-h); width: var(--kio-tune-w);
        max-height: none; margin: 0; padding: 12px 12px 12px 14px; z-index: 1001;
        background: rgba(20, 20, 24, 0.98); border-left: 1px solid var(--border); border-bottom: 0;
      }
      body.kio-tuning .app { margin-right: calc(var(--kio-tune-w) + 24px); }
      body.kio-tuning .modal-backdrop { right: var(--kio-tune-w); }
      #kio-tune.open .kio-field { grid-template-columns: 1fr 62px 28px; }
      #kio-tune.open .kio-field > span.kio-label { grid-column: 1 / 4; }
      #kio-tune.open .kio-field input[type=text], #kio-tune.open .kio-field textarea, #kio-tune.open .kio-field select { grid-column: 1 / 3; }
    }
    @media (min-width: 1500px) { :root { --kio-tune-w: 520px; } }

    @media (max-width: 640px) {
      #kio-tune { max-height: 52vh; max-height: 52dvh; }
      .kio-field { grid-template-columns: 1fr 62px 28px; }
      .kio-field > span.kio-label { grid-column: 1 / 4; }
      .kio-field input[type=text], .kio-field textarea, .kio-field select { grid-column: 1 / 3; }
      /* 16px stops iOS Safari zooming the page when a field is focused */
      select, input, textarea, #kio-say-text { font-size: 16px; }
      .app { padding: 16px 12px calc(var(--kio-bar-h) + 16px); }
      h1 { font-size: 19px; margin-bottom: 14px; }
      .card { padding: 14px; }
      .btn { width: 100%; padding: 12px 16px; font-size: 16px; }
      .actions { justify-content: stretch; }
      .tabs { flex-wrap: wrap; }

      /* session view: full screen above the bar, video cropped to portrait */
      .modal-backdrop { padding: 0 0 var(--kio-bar-h); align-items: stretch; }
      .modal {
        width: 100% !important; max-width: 100%; border: 0; border-radius: 0;
        max-height: none; height: 100%;
      }
      .modal-header { padding: 8px 12px; padding-top: calc(8px + env(safe-area-inset-top, 0px)); }
      .modal-body { padding: 0; display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0; overflow: hidden; }
      .video-wrap { flex: 1 1 auto; min-height: 0; max-height: none; border-radius: 0; aspect-ratio: auto !important; }
      video { object-fit: cover; object-position: 50% 30%; }
      .modal-controls { margin: 0; padding: 6px 12px; flex: 0 0 auto; }
      .mic-row { min-height: 0; gap: 10px; }
      .mic-toggle-wrap { width: 64px; height: 64px; }
      .mic-select { max-width: none; }

      #kio-say { padding: 6px 10px calc(8px + env(safe-area-inset-bottom, 0px)); }
      #kio-transcript { max-height: 5.4em; }
      #kio-say-text { flex-basis: 100%; }
      .kio-opts { flex: 1 1 auto; }
      #kio-say-status { margin-left: auto; text-align: right; }
    }
    /* phone held sideways: the picture is already wide, show all of it */
    @media (max-height: 500px) and (orientation: landscape) {
      .modal-backdrop { padding: 0 0 var(--kio-bar-h); align-items: stretch; }
      .modal { width: 100% !important; border: 0; border-radius: 0; max-height: none; height: 100%; }
      .modal-header { display: none; }
      .modal-body { padding: 0; display: flex; flex-direction: row; flex: 1 1 auto; min-height: 0; overflow: hidden; }
      .video-wrap { flex: 1 1 auto; min-height: 0; max-height: none; border-radius: 0; aspect-ratio: auto !important; }
      video { object-fit: contain; }
      .modal-controls { margin: 0; padding: 6px; flex: 0 0 auto; display: flex; align-items: center; }
      .mic-row { flex-direction: column; min-height: 0; }
      .mic-select { display: none; }
      #kio-transcript { max-height: 2.8em; }
    }
  </style>'''

SAY_BAR = '''  <div id="kio-say">
    <div id="kio-tune">
      <div class="kio-tune-head">
        <b>Tuning: <span id="kio-tune-avatar">…</span></b>
        <span class="kio-grow" id="kio-tune-status"></span>
        <button class="kio-btn" id="kio-tune-show" type="button">Saved values</button>
        <button class="kio-btn" id="kio-tune-resetall" type="button">Reset all</button>
      </div>
      <div id="kio-tune-json">
        <pre id="kio-tune-pre"></pre>
        <button class="kio-btn" id="kio-tune-copy" type="button">Copy</button>
        <a class="kio-btn" href="/avatar-settings" target="_blank" rel="noopener" style="text-decoration:none">All avatars + schema (JSON)</a>
      </div>
      <div id="kio-tune-groups"></div>
    </div>
    <div id="kio-transcript" aria-live="polite"></div>
    <div class="kio-row">
      <input id="kio-say-text" type="text" enterkeyhint="send" autocomplete="off" autocapitalize="sentences"
             placeholder="Type for her to say (or tick ask)">
      <div class="kio-opts">
        <label class="kio-check" title="Send the text to the LLM as a user turn instead of speaking it verbatim"><input id="kio-say-ask" type="checkbox"> ask</label>
        <label class="kio-check" title="Cut off whatever she is saying first"><input id="kio-say-interrupt" type="checkbox"> interrupt</label>
        <span id="kio-say-status"></span>
        <button id="kio-tune-toggle" type="button" title="Per-avatar motion and conversation settings; changes apply live and are saved">Tune</button>
        <button id="kio-say-send" type="button">Send</button>
      </div>
    </div>
  </div>
  <script>
    (function () {
      var bar = document.getElementById("kio-say");
      var input = document.getElementById("kio-say-text");
      var status = document.getElementById("kio-say-status");
      var box = document.getElementById("kio-transcript");

      // Tell the layout how tall the bar is, so the page and the session view
      // leave room for it instead of being covered.
      function measure() {
        document.documentElement.style.setProperty("--kio-bar-h", bar.offsetHeight + "px");
      }
      if (window.ResizeObserver) new ResizeObserver(measure).observe(bar);
      window.addEventListener("resize", measure);
      measure();

      // Phones: keep the bar above the on-screen keyboard. A fixed element is
      // positioned against the layout viewport, which the keyboard does not shrink.
      var vv = window.visualViewport;
      function followKeyboard() {
        if (!vv) return;
        var hidden = Math.max(0, window.innerHeight - vv.height - vv.offsetTop);
        bar.style.bottom = hidden + "px";
      }
      if (vv) {
        vv.addEventListener("resize", followKeyboard);
        vv.addEventListener("scroll", followKeyboard);
      }

      // Phones: some browsers leave the remote audio paused until a tap. Any tap
      // on the page resumes whatever stream is attached but not playing.
      document.addEventListener("click", function () {
        document.querySelectorAll("video, audio").forEach(function (el) {
          if (el.srcObject && el.paused) el.play().catch(function () {});
        });
      }, true);

      // Diagnostics. Remember the page's RTCPeerConnection so its statistics can
      // be read, and report what was received/decoded/played to the server log.
      var kioPc = null;
      var OrigPC = window.RTCPeerConnection;
      if (OrigPC) {
        window.RTCPeerConnection = function (cfg) {
          var pc = new OrigPC(cfg);
          if (!(cfg && cfg.kioProbe)) kioPc = pc;
          pc.addEventListener("track", function (ev) {
            // The avatar's sound comes from the separate <audio> element. Muting
            // the <video> lets every phone browser start it without a tap.
            setTimeout(function () {
              document.querySelectorAll("video").forEach(function (v) {
                v.muted = true;
                v.setAttribute("muted", "");
                if (v.srcObject && v.paused) v.play().catch(function () {});
              });
              document.querySelectorAll("audio").forEach(function (a) {
                if (a.srcObject && a.paused) a.play().catch(function () {});
              });
            }, 50);
          });
          return pc;
        };
        window.RTCPeerConnection.prototype = OrigPC.prototype;
        Object.keys(OrigPC).forEach(function (k) { try { window.RTCPeerConnection[k] = OrigPC[k]; } catch (e) {} });
        if (OrigPC.generateCertificate) window.RTCPeerConnection.generateCertificate = OrigPC.generateCertificate.bind(OrigPC);
      }
      var reports = 0, lastPc = null;
      async function report() {
        var pc = kioPc;
        if (!pc || pc.connectionState === "closed") return;
        if (pc !== lastPc) { lastPc = pc; reports = 0; }
        if (reports >= 8) return;  // about the first 40 s of each session
        reports++;
        var ev = { n: reports, pc: pc.connectionState, ice: pc.iceConnectionState, w: window.innerWidth, h: window.innerHeight };
        var v = document.querySelector("video"), a = document.querySelector("audio");
        if (v) { ev.v_src = !!v.srcObject; ev.v_paused = v.paused; ev.v_ready = v.readyState; ev.v_w = v.videoWidth; ev.v_h = v.videoHeight; ev.v_muted = v.muted; ev.v_err = v.error ? v.error.message || v.error.code : null; ev.v_box = v.clientWidth + "x" + v.clientHeight; }
        if (a) { ev.a_src = !!a.srcObject; ev.a_paused = a.paused; ev.a_ready = a.readyState; }
        try {
          var stats = await pc.getStats(), codecs = {};
          stats.forEach(function (s) { if (s.type === "codec") codecs[s.id] = s.mimeType; });
          stats.forEach(function (s) {
            if (s.type !== "inbound-rtp") return;
            var k = s.kind === "video" ? "rv_" : "ra_";
            ev[k + "codec"] = codecs[s.codecId] || null;
            ev[k + "packets"] = s.packetsReceived; ev[k + "lost"] = s.packetsLost;
            if (s.kind === "video") { ev.rv_frames_received = s.framesReceived; ev.rv_frames_decoded = s.framesDecoded; ev.rv_frames_dropped = s.framesDropped; ev.rv_size = (s.frameWidth || 0) + "x" + (s.frameHeight || 0); ev.rv_decoder = s.decoderImplementation || null; ev.rv_keyframes = s.keyFramesDecoded; ev.rv_pli = s.pliCount; ev.rv_nack = s.nackCount; }
          });
        } catch (e) { ev.stats_error = String(e.message || e); }
        try {
          fetch("/clientlog", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ ua: navigator.userAgent, events: [ev] }) });
        } catch (e) {}
      }
      setInterval(report, 5000);

      // Phones: do not let the screen sleep while she is on it.
      var wakeLock = null;
      async function keepAwake() {
        try {
          if (navigator.wakeLock && !wakeLock && document.visibilityState === "visible") {
            wakeLock = await navigator.wakeLock.request("screen");
            wakeLock.addEventListener("release", function () { wakeLock = null; });
          }
        } catch (e) { /* not supported or denied: nothing to do */ }
      }
      document.addEventListener("playing", function (e) { if (e.target.tagName === "VIDEO") keepAwake(); }, true);
      document.addEventListener("visibilitychange", function () {
        var v = document.querySelector("video");
        if (document.visibilityState === "visible" && v && v.srcObject && !v.paused) keepAwake();
      });

      // ---- Tuning panel: per-avatar settings, applied live and saved on every change.
      (function () {
        var panel = document.getElementById("kio-tune"), groupsEl = document.getElementById("kio-tune-groups");
        var toggle = document.getElementById("kio-tune-toggle"), statusEl = document.getElementById("kio-tune-status");
        var avatarEl = document.getElementById("kio-tune-avatar"), pre = document.getElementById("kio-tune-pre");
        var jsonBox = document.getElementById("kio-tune-json");
        var schema = null, defaults = {}, values = {}, avatar = "", controls = {}, saveTimer = null, saving = 0, voices = null;

        function currentAvatar() { var sel = document.getElementById("avatar"); return sel && sel.value ? sel.value : ""; }
        function differs(key) { return values[key] !== defaults[key]; }
        function overrides() { var o = {}; Object.keys(values).forEach(function (k) { if (differs(k)) o[k] = values[k]; }); return o; }
        function showJson() { pre.textContent = JSON.stringify({ avatar: avatar, overrides: overrides(), values: values }, null, 2); }

        function refreshMarks() {
          var counts = {};
          schema.forEach(function (f) {
            var c = controls[f.key]; if (!c) return;
            var ch = differs(f.key); c.row.classList.toggle("changed", ch);
            if (ch) counts[f.group] = (counts[f.group] || 0) + 1;
          });
          groupsEl.querySelectorAll("details").forEach(function (d) {
            var n = counts[d.dataset.group] || 0;
            d.querySelector(".kio-count").textContent = n ? n + " changed" : "";
          });
          showJson();
        }

        function setControl(f) {
          var c = controls[f.key], v = values[f.key];
          if (f.kind === "number") { c.range.value = v; c.num.value = v; }
          else if (f.kind === "bool") c.box.checked = !!v;
          else {
            if (c.text.tagName === "SELECT" && v && !Array.prototype.some.call(c.text.options, function (o) { return o.value === v; })) {
              var o = document.createElement("option"); o.value = v; o.textContent = v + " (not offered by the server)"; c.text.appendChild(o);
            }
            c.text.value = v || "";
          }
        }

        function queueSave() {
          statusEl.textContent = "saving…";
          clearTimeout(saveTimer);
          saveTimer = setTimeout(save, 200);
        }
        async function save() {
          if (!avatar) return;
          var mine = ++saving;
          try {
            var r = await fetch("/avatar-settings/" + encodeURIComponent(avatar), {
              method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ values: overrides() }),
            });
            if (mine !== saving) return;  // a newer save is on its way
            if (!r.ok) { var b = await r.json().catch(function () { return {}; }); statusEl.textContent = "not saved: " + (b.detail || r.status); return; }
            statusEl.textContent = "saved " + new Date().toLocaleTimeString() + " · applies live";
          } catch (e) { statusEl.textContent = "not saved: " + (e.message || e); }
        }
        function change(f, v) {
          if (f.kind === "number") {
            v = Number(v); if (!isFinite(v)) return;
            v = Math.min(f.max, Math.max(f.min, v));
          }
          values[f.key] = v; setControl(f); refreshMarks(); queueSave();
        }

        function build() {
          groupsEl.textContent = ""; controls = {};
          var groups = {}, order = [];
          schema.forEach(function (f) { if (!groups[f.group]) { groups[f.group] = []; order.push(f.group); } groups[f.group].push(f); });
          order.forEach(function (g, gi) {
            var d = document.createElement("details"); d.dataset.group = g; if (gi === 0) d.open = true;
            var s = document.createElement("summary"); s.textContent = g;
            var cnt = document.createElement("span"); cnt.className = "kio-count"; s.appendChild(cnt); d.appendChild(s);
            var wrap = document.createElement("div"); wrap.className = "kio-fields";
            groups[g].forEach(function (f) {
              var row = document.createElement("div"); row.className = "kio-field"; row.title = f.help || "";
              var lab = document.createElement("span"); lab.className = "kio-label"; lab.textContent = f.label; row.appendChild(lab);
              var c = { row: row };
              if (f.kind === "number") {
                c.range = document.createElement("input"); c.range.type = "range"; c.range.min = f.min; c.range.max = f.max; c.range.step = f.step;
                c.num = document.createElement("input"); c.num.type = "number"; c.num.min = f.min; c.num.max = f.max; c.num.step = f.step;
                c.range.addEventListener("input", function () { change(f, c.range.value); });
                c.num.addEventListener("change", function () { change(f, c.num.value); });
                row.appendChild(c.range); row.appendChild(c.num);
              } else if (f.kind === "bool") {
                c.box = document.createElement("input"); c.box.type = "checkbox";
                c.box.addEventListener("change", function () { change(f, c.box.checked); });
                row.appendChild(c.box); row.appendChild(document.createElement("span"));
              } else if (f.key === "tts_voice" && voices && voices.voices && voices.voices.length) {
                // The TTS server told us which voices exist: offer them instead of free text.
                c.text = document.createElement("select");
                var first = document.createElement("option"); first.value = "";
                first.textContent = "(pod default: " + (voices["default"] || "?") + ")"; c.text.appendChild(first);
                voices.voices.forEach(function (v) { var o = document.createElement("option"); o.value = v; o.textContent = v; c.text.appendChild(o); });
                c.text.addEventListener("change", function () { change(f, c.text.value); });
                row.title = (f.help || "") + " — " + voices.voices.length + " voices from " + (voices.model || "the TTS server");
                row.appendChild(c.text);
              } else {
                c.text = document.createElement(f.kind === "longtext" ? "textarea" : "input");
                if (f.kind !== "longtext") c.text.type = "text";
                c.text.placeholder = "(pod default)";
                c.text.addEventListener("change", function () { change(f, c.text.value.trim()); });
                row.appendChild(c.text);
              }
              var rst = document.createElement("button"); rst.type = "button"; rst.className = "kio-reset"; rst.textContent = "↺";
              rst.title = "Back to default (" + f["default"] + ")";
              rst.addEventListener("click", function () { change(f, defaults[f.key]); });
              row.appendChild(rst);
              controls[f.key] = c; wrap.appendChild(row);
            });
            d.appendChild(wrap); groupsEl.appendChild(d);
          });
        }

        async function load() {
          avatar = currentAvatar(); avatarEl.textContent = avatar || "(pick an avatar)";
          if (!avatar) return;
          try {
            if (!schema) {
              var all = await (await fetch("/avatar-settings")).json();
              schema = all.schema; defaults = all.defaults;
              try { voices = await (await fetch("/voices")).json(); } catch (e) { voices = null; }
              build();
            }
            var one = await (await fetch("/avatar-settings/" + encodeURIComponent(avatar))).json();
            values = one.values; schema.forEach(setControl); refreshMarks();
            statusEl.textContent = Object.keys(one.overrides).length ? "loaded saved values" : "all defaults";
          } catch (e) { statusEl.textContent = "could not load settings: " + (e.message || e); }
        }

        function setOpen(open) {
          panel.classList.toggle("open", open); toggle.classList.toggle("on", open);
          document.body.classList.toggle("kio-tuning", open);
          try { localStorage.setItem("kio_tune_open", open ? "1" : "0"); } catch (e) {}
          window.dispatchEvent(new Event("resize"));  // let the session view re-fit beside the panel
          if (open) load();
        }
        toggle.addEventListener("click", function () { setOpen(!panel.classList.contains("open")); });
        // Wide screens: open beside the video unless it was closed last time. Phones: closed unless opened.
        var remembered = null;
        try { remembered = localStorage.getItem("kio_tune_open"); } catch (e) {}
        if (remembered === "1" || (remembered === null && window.matchMedia("(min-width: 1000px)").matches)) setOpen(true);
        document.getElementById("kio-tune-show").addEventListener("click", function () { jsonBox.classList.toggle("open"); showJson(); });
        document.getElementById("kio-tune-copy").addEventListener("click", function () {
          var t = pre.textContent;
          if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(function () { statusEl.textContent = "copied"; }, function () { statusEl.textContent = "select the text and copy it"; });
          else statusEl.textContent = "select the text and copy it";
        });
        document.getElementById("kio-tune-resetall").addEventListener("click", function () {
          if (!schema || !avatar) return;
          if (!window.confirm("Reset every setting for " + avatar + " to its default?")) return;
          schema.forEach(function (f) { values[f.key] = defaults[f.key]; setControl(f); });
          refreshMarks(); queueSave();
        });
        // The avatar picker belongs to the page's own code; just notice when its value changes.
        setInterval(function () { if (panel.classList.contains("open") && currentAvatar() !== avatar) load(); }, 700);
      })();

      async function send() {
        var text = input.value.trim();
        if (!text) return;
        status.textContent = "sending…";
        try {
          var r = await fetch("/say", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              text: text,
              interrupt: document.getElementById("kio-say-interrupt").checked,
              mode: document.getElementById("kio-say-ask").checked ? "ask" : "say",
            }),
          });
          if (r.ok) { status.textContent = "queued"; input.value = ""; }
          else { var b = await r.json().catch(function () { return {}; }); status.textContent = r.status + ": " + (b.detail || "error"); }
        } catch (e) { status.textContent = "failed: " + (e.message || e); }
      }
      document.getElementById("kio-say-send").addEventListener("click", send);
      input.addEventListener("keydown", function (e) { if (e.key === "Enter") send(); });

      var colors = { you: "#8cf", avatar: "#fc9", system: "#888" };
      var lastKey = "";
      async function poll() {
        if (document.visibilityState !== "visible") return;
        try {
          var r = await fetch("/transcript");
          if (!r.ok) return;
          var data = await r.json();
          var entries = data.entries.slice(-12);
          var key = JSON.stringify(entries);
          if (key === lastKey) return;
          lastKey = key;
          box.textContent = "";
          entries.forEach(function (e) {
            var line = document.createElement("div");
            var who = document.createElement("b");
            who.textContent = (e.role === "avatar" && e.name ? e.name : e.role) + ": ";
            who.style.color = colors[e.role] || "#ccc";
            line.appendChild(who);
            line.appendChild(document.createTextNode(e.text));
            var bits = [];
            if (e.stt_s != null) bits.push("stt " + e.stt_s + "s");
            if (e.llm_first_sentence_s != null) bits.push("llm " + e.llm_first_sentence_s + "s");
            if (e.heard_to_first_audio_s != null) bits.push("to first audio " + e.heard_to_first_audio_s + "s");
            if (bits.length) {
              var t = document.createElement("span");
              t.textContent = "  (" + bits.join(", ") + ")";
              t.style.color = "#777";
              line.appendChild(t);
            }
            box.appendChild(line);
          });
          box.scrollTop = box.scrollHeight;
        } catch (e) { /* page still works without the transcript */ }
      }
      setInterval(poll, 1000);
      poll();
    })();
  </script>

  <script type="importmap">{"imports": {"preact": "/vendor/preact.module.js"}}</script>
  <script type="module">
    import { h, render } from "preact";
    import { useState, useEffect, useRef, useCallback } from "/vendor/hooks.module.js";
    import htm from "/vendor/htm.module.js";
'''

patch("src/avaturn_live_streamer/local_stream_ui.html", [
    (
        '  <meta charset="utf-8" />\n',
        '  <meta charset="utf-8" />\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />\n'
        '  <meta name="theme-color" content="#0f1115" />\n',
        1,
    ),
    ("  </style>", MOBILE_CSS, 1),
    (
        '  <script type="module">\n'
        '    import { h, render } from "https://esm.sh/preact@10.22.0";\n'
        '    import { useState, useEffect, useRef, useCallback } from "https://esm.sh/preact@10.22.0/hooks";\n'
        '    import htm from "https://esm.sh/htm@3.1.1";\n',
        SAY_BAR,
        1,
    ),
    (
        '    const ENGINE_TABS = [\n      { id: "openai", label: "OpenAI Realtime" },\n'
        '      { id: "cartesia", label: "Cartesia" },\n    ];\n',
        '    const ENGINE_TABS = [\n      { id: "local", label: "Local" },\n    ];\n',
        1,
    ),
    (
        '      const buildEngineConfig = useCallback(() => {\n        if (engine === "openai") {\n',
        '      const buildEngineConfig = useCallback(() => {\n'
        '        if (engine === "local") {\n          return { type: "local" };\n        }\n'
        '        if (engine === "openai") {\n',
        1,
    ),
    # The engine is fixed; ignore whatever an older version of the page remembered.
    (
        '      const [engine, setEngine] = useLocalStorage("engine", "openai");\n',
        '      const engine = "local", setEngine = () => {};\n',
        1,
    ),
    # The "Connectivity" card ran a test handshake on every page load and reported
    # on STUN/TURN reachability. None of that applies on local networks, and each
    # run took one of the media ports for 15 s.
    ("      const probe = useConnectivityProbe();\n", "", 1),
    ("          <${ConnectivityCard} probe=${probe} />\n", "", 1),
    ('<div class="title">Session timed out</div>', '<div class="title">Session ended</div>', 1),
    (
        "The server ended this session (timeout or shutdown). Close this window and start a new session.",
        "This session was ended by the server: it was started on another device, timed out, or the "
        "server restarted. Close this window to start a new one here.",
        1,
    ),
])
if "esm.sh" in (root / "src/avaturn_live_streamer/local_stream_ui.html").read_text():
    sys.exit("PATCH FAILED: the page still references esm.sh")

# --- 4. launcher ----------------------------------------------------------------
patch("scripts/run_local_stream.py", [
    (
        "    host = os.environ.get(\"STREAMER_HOST\", \"127.0.0.1\")\n",
        "    import kio_required_env\n\n"
        "    kio_required_env.check(\"launcher\")  # every setting must come from Kubernetes\n"
        "    host = os.environ.get(\"STREAMER_HOST\", \"127.0.0.1\")\n",
        1,
    ),
    ('        "pixi", "run", "-e", "default",\n', '        "pixi", "run", "-e", "renderer",\n', 1),
    (
        '        "avtr1_renderer.api.app:app",\n        "--host", "0.0.0.0",\n',
        '        "avtr1_renderer.api.app:app",\n        "--host", "127.0.0.1",\n',
        1,
    ),
])

# --- 4b. renderer: per-avatar tuning ---------------------------------------------
# The logic lives in src/kio_motion_tuning.py (copied in by the Dockerfile); the
# patches below only call it. Settings come from
# $AVTR1_LOCAL_STORAGE/avatar_settings.json (see src/kio_avatar_settings.py) and
# are re-read when the file changes, so no restart is needed.
#   - override_options: upstream's motion settings per avatar (the streamer
#     always sends the defaults)
#   - guidance_vector:  per-region guidance (head / brow / eyes / mouth)
#   - apply_gains:      head, brow, eyes, eyelids and mouth gains on the result
patch("src/avtr1_renderer/pipeline.py", [
    (
        'TRANSPARENT_BG_ID = "transparent"\n',
        'TRANSPARENT_BG_ID = "transparent"\n\nimport kio_motion_tuning as _kio_tuning  # local addition\n',
        1,
    ),
    (
        "        motions, next_state = self._motion_generator.generate_chunk(\n"
        "            chunk, avatar, state, options\n"
        "        )\n",
        "        options = _kio_tuning.override_options(options, avatar)\n"
        "        motions, next_state = self._motion_generator.generate_chunk(\n"
        "            chunk, avatar, state, options\n"
        "        )\n"
        "        _kio_tuning.apply_gains(motions, avatar)\n",
        1,
    ),
])
_GEN_EDITS = [
    (
        "import roma\n",
        "import roma\n\nimport kio_motion_tuning as _kio_tuning  # local addition\n",
        1,
    ),
]
for _name, _attr in (("w_self", "cfg_self_audio"), ("w_other", "cfg_other_audio"), ("w_kp", "cfg_kp")):
    _GEN_EDITS.append((
        f"        {_name} = torch.full(\n"
        f"            (self.latent_dim,), options.{_attr}, device=device, dtype=torch.float32\n"
        "        )\n",
        f"        {_name} = _kio_tuning.guidance_vector(\n"
        f'            "{_attr}", options.{_attr}, avatar, self.latent_dim, device\n'
        "        )\n",
        1,
    ))
patch("src/avtr1_renderer/avtr1_motion_generator.py", _GEN_EDITS)

# --- 5. aioice (streamer env) -----------------------------------------------------
AIOICE_HELPER = '''

# ---- local additions (see devops-templates/Dockerfiles/avtr1/live) ----------
async def _kio_bind(loop, factory, address):
    """Bind in the fixed range KIO_WEBRTC_UDP_PORTS ("40000-40009") if set.

    A Kubernetes Service can only forward ports it lists, so the random
    ephemeral port aioice normally uses cannot be reached from outside.
    """
    import os

    # The live pod's start-up check guarantees this is set. An unset value keeps
    # aioice's own behaviour, which other users of the library (tests) rely on.
    spec = os.environ.get("KIO_WEBRTC_UDP_PORTS", "").strip()
    if not spec:
        return await loop.create_datagram_endpoint(factory, local_addr=(address, 0))
    lo, _, hi = spec.partition("-")
    last_exc = None
    for port in range(int(lo), int(hi or lo) + 1):
        try:
            return await loop.create_datagram_endpoint(factory, local_addr=(address, port))
        except OSError as exc:
            last_exc = exc
    raise OSError(f"no free UDP port in {spec}: {last_exc}")


def _kio_first_packet_wait() -> float:
    import os

    try:
        return max(0.0, float(os.environ.get("KIO_READY_TIMEOUT", "")))
    except ValueError:
        return 0.0
'''

patch(".pixi/envs/streamer/lib/python3.12/site-packages/aioice/ice.py", [
    (
        "                transport, protocol = await loop.create_datagram_endpoint(\n"
        "                    lambda: StunProtocol(self), local_addr=(address, 0)\n"
        "                )\n",
        "                transport, protocol = await _kio_bind(\n"
        "                    loop, lambda: StunProtocol(self), address\n"
        "                )\n",
        1,
    ),
    (
        "        if self._check_list:\n"
        "            res = await self._check_list_state.get()\n"
        "        else:\n"
        "            res = ICE_FAILED\n",
        "        if self._check_list:\n"
        "            res = await self._check_list_state.get()\n"
        "        else:\n"
        "            # local change: a browser without microphone permission hides its\n"
        "            # address behind an mDNS name this pod cannot resolve, leaving no\n"
        "            # pairs to check. Wait for the browser to reach us first; its\n"
        "            # packet creates the pair (peer-reflexive candidate).\n"
        "            # How long: KIO_READY_TIMEOUT, the same allowance the session start\n"
        "            # uses. Unset (any other user of this library) = upstream behaviour.\n"
        "            try:\n"
        "                res = await asyncio.wait_for(self._check_list_state.get(), _kio_first_packet_wait())\n"
        "            except asyncio.TimeoutError:\n"
        "                res = ICE_FAILED\n",
        1,
    ),
])
p = root / ".pixi/envs/streamer/lib/python3.12/site-packages/aioice/ice.py"
p.write_text(p.read_text(encoding="utf-8") + AIOICE_HELPER, encoding="utf-8")
print("all patches applied")
