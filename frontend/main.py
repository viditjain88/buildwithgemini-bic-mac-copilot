"""Minimal FastAPI proxy for a deployed A2A agent (Agent Runtime, agents-cli 1.1.0+).

The browser talks ONLY to this proxy (same origin, no CORS, no GCP creds in the
browser). The proxy authenticates with Application Default Credentials and
forwards chat to the deployed agent over the A2A protocol, returning replies as
structured parts the chat UI knows how to show:

  * {"kind": "text", "text": ...}  -> a normal chat bubble
  * {"kind": "a2ui", "data": ...}  -> one A2UI message (beginRendering /
    surfaceUpdate); static/index.html renders these as a card.
"""

import base64
import json
import os
import uuid

import google.auth
import google.auth.transport.requests
import httpx
from google.protobuf.json_format import MessageToDict
from a2a.client import ClientConfig, create_client
from a2a.types import (
    Message,
    Part,
    Role,
    SendMessageRequest,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

RESOURCE = os.environ["AGENT_ENGINE_RESOURCE_NAME"]
AGENT_DIRECTORY = os.environ.get("AGENT_DIRECTORY", "app")
LOCATION = RESOURCE.split("/locations/")[1].split("/")[0]

A2A_BASE = (
    f"https://{LOCATION}-aiplatform.googleapis.com/reasoningEngines/v1/"
    f"{RESOURCE}/api/a2a/{AGENT_DIRECTORY}"
)
_A2UI_MIME = "application/json+a2ui"

_creds, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)


def _auth_headers() -> dict[str, str]:
    _creds.refresh(google.auth.transport.requests.Request())
    return {
        "Authorization": f"Bearer {_creds.token}",
        "Content-Type": "application/json",
    }


app = FastAPI()


@app.exception_handler(Exception)
async def _json_errors(request: Request, exc: Exception):
    return JSONResponse(
        status_code=200,
        content={
            "parts": [{"kind": "text", "text": f"Error: {type(exc).__name__}: {exc}"}]
        },
    )


_contexts: dict[str, str] = {}


def _extract_parts_from_artifact(artifact) -> list[dict]:
    out: list[dict] = []
    if not artifact:
        return out
    art_dict = MessageToDict(artifact)
    for p in art_dict.get("parts", []):
        text = p.get("text")
        if text:
            out.append({"kind": "text", "text": text})

        data_field = p.get("data")
        if isinstance(data_field, dict):
            meta = data_field.get("metadata", {})
            mime = meta.get("mimeType") if isinstance(meta, dict) else None
            a2ui_data = data_field.get("data")
            if (mime == _A2UI_MIME or "a2ui" in str(mime)) and a2ui_data:
                out.append({"kind": "a2ui", "data": a2ui_data})
            elif "text" in data_field:
                out.append({"kind": "text", "text": str(data_field["text"])})
        elif isinstance(data_field, str):
            try:
                out.append({"kind": "a2ui", "data": json.loads(data_field)})
            except Exception:
                out.append({"kind": "text", "text": data_field})
    return out


def _extract_parts_from_message(msg) -> list[dict]:
    out: list[dict] = []
    if not msg:
        return out
    msg_dict = MessageToDict(msg)
    for p in msg_dict.get("parts", []):
        text = p.get("text")
        if text:
            out.append({"kind": "text", "text": text})
    return out


@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    message = body.get("message", "")
    image_data = body.get("image_data") or body.get("image")
    user_id = body.get("user_id") or "web-user"
    parts: list[dict] = []

    msg_parts: list[Part] = []
    if message:
        msg_parts.append(Part(text=message))

    if image_data:
        if "," in image_data:
            header, b64_str = image_data.split(",", 1)
            mime_type = header.split(";")[0].replace("data:", "") if "data:" in header else "image/png"
        else:
            b64_str = image_data
            mime_type = "image/png"
        try:
            img_bytes = base64.b64decode(b64_str)
            msg_parts.append(Part(raw=img_bytes, media_type=mime_type))
        except Exception as img_err:
            print(f"Error decoding image data: {img_err}", flush=True)

    if not msg_parts:
        msg_parts.append(Part(text="Analysis requested for uploaded CT scan."))

    async with httpx.AsyncClient(headers=_auth_headers(), timeout=120) as client:
        a2a_client = await create_client(
            A2A_BASE,
            client_config=ClientConfig(httpx_client=client),
        )

        ctx_id = _contexts.get(user_id)
        msg_kwargs = {
            "message_id": str(uuid.uuid4()),
            "role": Role.ROLE_USER,
            "parts": msg_parts,
        }
        if ctx_id:
            msg_kwargs["context_id"] = ctx_id

        msg = Message(**msg_kwargs)

        last_task = None
        async for event in a2a_client.send_message(SendMessageRequest(message=msg)):
            print(f"DEBUG A2A Event: {event}", flush=True)

            if hasattr(event, "HasField"):
                if event.HasField("message"):
                    parts.extend(_extract_parts_from_message(event.message))
                if event.HasField("artifact_update"):
                    parts.extend(_extract_parts_from_artifact(event.artifact_update.artifact))
                if event.HasField("status_update") and event.status_update.status.HasField("message"):
                    parts.extend(_extract_parts_from_message(event.status_update.status.message))
                if event.HasField("task"):
                    last_task = event.task
                    if event.task.context_id:
                        _contexts[user_id] = event.task.context_id

        if not parts and last_task is not None:
            for msg_item in last_task.history:
                if msg_item.role == Role.ROLE_AGENT:
                    parts.extend(_extract_parts_from_message(msg_item))
            if not parts:
                for artifact in last_task.artifacts:
                    parts.extend(_extract_parts_from_artifact(artifact))

    if not parts:
        parts = [{"kind": "text", "text": "(The agent didn't return a reply.)"}]
    return JSONResponse({"parts": parts})


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
