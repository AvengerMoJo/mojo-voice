from __future__ import annotations


class ExecuTorchSpeechRuntime:
    """
    ExecuTorch-ready runtime seam.

    Replace these placeholder methods with real on-device model inference:
    - `transcribe_from_mic`: mic audio -> STT model -> text
    - `speak`: text -> TTS model/vocoder -> speaker
    """

    async def transcribe_from_mic(self) -> str:
        # Placeholder for quick desktop demo fallback.
        return input("you(exec)> ").strip()

    async def speak(self, text: str) -> None:
        # Placeholder for quick desktop demo fallback.
        print(f"assistant(exec)> {text}")
