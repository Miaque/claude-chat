from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_response_processor():
    # python 读取 debug_streams 目录下的所有文件
    debug_files = Path("debug_streams/stream_20251203_101141.jsonl")
    with open(debug_files, encoding="utf-8") as f:
        for line in f:
            print(line.strip())
