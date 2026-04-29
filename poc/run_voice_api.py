from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("MOJO_VOICE_API_HOST", "0.0.0.0")
    port = int(os.getenv("MOJO_VOICE_API_PORT", "9089"))
    uvicorn.run("mojo_voice.voice_api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
