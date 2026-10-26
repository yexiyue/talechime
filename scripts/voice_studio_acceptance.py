#!/usr/bin/env python3
"""Voice studio CLI acceptance checklist.

Runs the model-independent parts against a temporary home (no downloads,
no user data touched) and prints the real-model steps for manual execution.
Record real-model results under docs/records per repository convention.

Usage: python3 scripts/voice_studio_acceptance.py [path-to-talechime-binary]
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import wave


def run(binary, home, *args, check=True):
    command = [binary, "--config", os.path.join(home, "config.json")]
    command += list(args)
    result = subprocess.run(command, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(command)}\n{result.stderr}")
    return result


def write_reference(path):
    with wave.open(path, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        frames = b"".join(
            struct.pack("<h", int(8000 * __import__("math").sin(i * 0.1))) for i in range(24000)
        )
        writer.writeframes(frames)


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else "target/release/talechime"
    with tempfile.TemporaryDirectory() as home:
        print(f"== model-independent checks (home: {home}) ==")

        listing = run(binary, home, "voices", "list").stdout
        assert "voice library is empty" in listing, listing
        print("[ok] voices list works without --backend on an empty library")

        engines = run(binary, home, "voices", "engines").stdout
        print(f"[ok] voices engines reported:\n{engines}")

        reference = os.path.join(home, "reference.wav")
        write_reference(reference)
        imported = run(binary, home, "voices", "import", reference,
                       "--name", "验收音色", "--text", "试音转写").stdout
        assert "Imported voice" in imported, imported
        print("[ok] voices import without --backend")

        listing = run(binary, home, "voices", "list").stdout
        assert "voice\t验收音色\timported" in listing, listing
        print("[ok] imported voice listed with source")

        bundle = os.path.join(home, "check.voice")
        run(binary, home, "voices", "export", "voice", "--output", bundle)
        run(binary, home, "voices", "remove", "voice")
        restored = run(binary, home, "voices", "import-bundle", bundle, "--id", "voice").stdout
        assert "Imported voice" in restored, restored
        print("[ok] export -> remove -> import-bundle roundtrip")

        run(binary, home, "voices", "rename", "voice", "--name", "改名")
        listing = run(binary, home, "voices", "list").stdout
        assert "改名" in listing, listing
        print("[ok] rename")

        # The model resources directory is never created by voice commands.
        assert not os.path.exists(os.path.join(home, "models")), "models dir created!"
        assert not os.path.exists(os.path.join(home, "resources")), "resources dir created!"
        print("[ok] voice commands download nothing")

    print(
        """
== manual real-model steps (record results in docs/records) ==
1. voices design --name X --description "低沉温和的中年男声" [--engine voxcpm]
   - engine preparation progress streams; result saved into the library
   - automatic policy prefers an already-prepared narration model (zero download)
2. talechime --voice <library-id> chapter.txt
   - first use materializes the voice into the selected model implicitly
   - switching --backend/--model reuses the same library voice
3. JSON Lines: DesignVoice then reference the returned voice id in a plan
   - ModelProgress during preparation, VoiceReady on completion
"""
    )


if __name__ == "__main__":
    main()
