from mojo_voice.config import load_config
from mojo_voice.pipeline import VoiceChatPipeline
from mojo_voice.runtime_mock import MockSpeechRuntime
from mojo_voice.runtime_executorch import ExecuTorchSpeechRuntime


def main() -> None:
    cfg = load_config()
    print(f"runtime={cfg.voice_runtime} mode={cfg.mcp_mode} mcp={cfg.mcp_url}")
    runtime = MockSpeechRuntime() if cfg.voice_runtime == "mock" else ExecuTorchSpeechRuntime()
    pipeline = VoiceChatPipeline(cfg, runtime)
    try:
        pipeline.run_sync()
    except Exception as exc:
        print(f"startup error: {exc}")
        print("hint: set MOJO_MCP_URL and MOJO_MCP_API_KEY (or MCP_API_KEY).")


if __name__ == "__main__":
    main()
