#!/usr/bin/env python3
"""
MoJo-Voice baseline latency benchmark.

Measures each pipeline stage:
1. STT (FunASR): WAV → text
2. MCP search_memory: text → memory results  
3. MCP llm_direct_chat (audio brain): prompt → reply text
4. TTS (CosyVoice3): text → WAV
5. Total end-to-end: audio in → audio out

Usage:
    python benchmark_baseline.py --wav test.wav --mcp-url http://localhost:8000
"""

import argparse
import json
import os
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

# Add mojo-voice to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "submodules" / "mojo-voice" / "poc"))


@dataclass
class StageResult:
    name: str
    duration_ms: float
    output_size: int = 0  # chars for text, bytes for audio
    error: Optional[str] = None


@dataclass  
class PipelineResult:
    input_audio: str
    input_duration_s: float
    stages: list[StageResult] = field(default_factory=list)
    total_ms: float = 0
    total_audio_out_bytes: int = 0
    error: Optional[str] = None


def timed_stage(name: str, fn, *args, **kwargs):
    """Run a function and time it."""
    t0 = time.perf_counter()
    result = None
    error = None
    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        error = str(e)
    elapsed = (time.perf_counter() - t0) * 1000
    size = 0
    if isinstance(result, str):
        size = len(result)
    elif isinstance(result, bytes):
        size = len(result)
    return StageResult(name=name, duration_ms=elapsed, output_size=size, error=error), result


def benchmark_mojo_voice(
    wav_path: str,
    mcp_url: str = "http://localhost:8000",
    mcp_api_key: Optional[str] = None,
) -> PipelineResult:
    """Run full mojo-voice pipeline benchmark."""
    
    # Read WAV
    with wave.open(wav_path, "rb") as wf:
        audio_duration = wf.getnframes() / wf.getframerate()
    
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()
    
    result = PipelineResult(
        input_audio=os.path.basename(wav_path),
        input_duration_s=audio_duration,
    )
    
    t_total_start = time.perf_counter()
    
    # Stage 1: STT
    try:
        from mojo_voice.audio_utils import to_wav16k_mono_bytes, write_temp_wav, safe_unlink, decode_audio_base64
        from mojo_voice.stt_funasr import FunASRSTT
        
        stt = FunASRSTT(device="cpu")
        stage_result, transcript = timed_stage("STT (FunASR)", lambda: (
            stt.load(),
            wav16k := to_wav16k_mono_bytes(audio_bytes),
            wav_tmp := write_temp_wav(wav16k),
            result := stt.transcribe_wav_path(wav_tmp),
            safe_unlink(wav_tmp),
            result.text
        )[-1])
        result.stages.append(stage_result)
    except ImportError as e:
        result.stages.append(StageResult("STT (FunASR)", 0, error=f"Import error: {e}"))
        result.error = "STT unavailable"
        return result
    except Exception as e:
        result.stages.append(StageResult("STT (FunASR)", 0, error=str(e)))
        result.error = f"STT failed: {e}"
        return result
    
    # Stage 2: MCP search_memory
    stage_result, memory_text = timed_stage("MCP search_memory", lambda: _mcp_call(
        mcp_url, mcp_api_key, "search_memory", {
            "query": transcript,
            "types": ["conversations", "documents"],
            "limit_per_type": 3,
            "max_content_chars": 220,
        }
    ))
    result.stages.append(stage_result)
    
    # Stage 3: MCP llm_direct_chat (audio brain)
    audio_brain_prompt = f"You are a helpful voice assistant. Reply concisely in 1-2 spoken sentences.\nMemory context:\n{memory_text}\n\nUser said: {transcript}"
    stage_result, reply_text = timed_stage("MCP llm_direct_chat", lambda: _mcp_call(
        mcp_url, mcp_api_key, "llm_direct_chat", {
            "system_prompt": "You are a helpful voice assistant. Reply concisely.",
            "message": audio_brain_prompt,
            "resource_id": "lmstudio",
            "max_tokens": 256,
        }
    ))
    result.stages.append(stage_result)
    
    # Stage 4: TTS
    try:
        cosy_model_path = os.getenv("COSYVOICE_MODEL_PATH", "pretrained_models/Fun-CosyVoice3-0.5B")
        cosy_speaker = os.getenv("COSYVOICE_SPEAKER", "")
        from mojo_voice.tts_cosyvoice2 import CosyVoiceTTS
        
        tts = CosyVoiceTTS(model_path=cosy_model_path, speaker=cosy_speaker)
        stage_result, audio_out = timed_stage("TTS (CosyVoice3)", lambda: (
            tts.load(),
            tts.synthesize_wav(reply_text)
        )[-1])
        result.stages.append(stage_result)
        if audio_out:
            result.total_audio_out_bytes = len(audio_out)
    except Exception as e:
        result.stages.append(StageResult("TTS (CosyVoice3)", 0, error=str(e)))
    
    result.total_ms = (time.perf_counter() - t_total_start) * 1000
    return result


def _mcp_call(base_url: str, api_key: Optional[str], tool_name: str, arguments: dict) -> str:
    """Make a single MCP tools/call and return text content."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["MCP-API-Key"] = api_key
    
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}}
    
    with httpx.Client(timeout=30) as client:
        resp = client.post(base_url, json=body, headers=headers)
        resp.raise_for_status()
        payload = resp.json()
    
    if "error" in payload:
        raise RuntimeError(f"MCP error: {payload['error']}")
    
    result = payload.get("result", {})
    content = result.get("content", [])
    if content:
        try:
            data = json.loads(content[0].get("text", "{}"))
            return data.get("reply", "") or data.get("message", "") or json.dumps(data)
        except json.JSONDecodeError:
            return content[0].get("text", "")
    return ""


def print_pipeline_result(r: PipelineResult):
    """Pretty print a pipeline benchmark result."""
    print()
    print(f"Input: {r.input_audio} ({r.input_duration_s:.1f}s)")
    print("-" * 60)
    total = 0
    for stage in r.stages:
        pct = (stage.duration_ms / r.total_ms * 100) if r.total_ms > 0 else 0
        status = f"❌ {stage.error[:40]}" if stage.error else f"{stage.duration_ms:>8.0f}ms ({pct:>5.1f}%)"
        size_info = f" [{stage.output_size} chars]" if stage.output_size and stage.name.startswith(("STT", "MCP")) else ""
        size_info = f" [{stage.output_size} bytes]" if stage.output_size and stage.name.startswith("TTS") else size_info
        print(f"  {stage.name:<25} {status}{size_info}")
    print("-" * 60)
    print(f"  {'TOTAL':<25} {r.total_ms:>8.0f}ms")
    if r.error:
        print(f"  ERROR: {r.error}")


def main():
    parser = argparse.ArgumentParser(description="mojo-voice baseline benchmark")
    parser.add_argument("--wav", required=True, nargs="+", help="WAV files to benchmark")
    parser.add_argument("--mcp-url", default="http://localhost:8000", help="MCP server URL")
    parser.add_argument("--mcp-key", default=None, help="MCP API key")
    parser.add_argument("--skip-tts", action="store_true", help="Skip TTS stage")
    args = parser.parse_args()
    
    for wav in args.wav:
        if not os.path.exists(wav):
            print(f"Skipping: {wav} (not found)", file=sys.stderr)
            continue
        print(f"\nBenchmarking: {wav}")
        r = benchmark_mojo_voice(wav, args.mcp_url, args.mcp_key)
        print_pipeline_result(r)


if __name__ == "__main__":
    main()
