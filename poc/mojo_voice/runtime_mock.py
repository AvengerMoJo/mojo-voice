from __future__ import annotations


class MockSpeechRuntime:
    async def transcribe_from_mic(self) -> str:
        return input("you> ").strip()

    async def speak(self, text: str) -> None:
        print(f"assistant> {text}")
