from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import aiohttp

from .timebase import now_pair


@dataclass(slots=True)
class TimedResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    request_started_wall_ns: int
    request_started_mono_ns: int
    response_received_wall_ns: int
    response_received_mono_ns: int
    url: str

    def json(self) -> Any:
        import orjson

        return orjson.loads(self.body)


class TimedHttpClient:
    def __init__(self, timeout_seconds: float = 20, user_agent: str | None = None) -> None:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        agent = user_agent or os.getenv(
            "HTTP_USER_AGENT", "polymarket-weather-alpha-tester/0.1.8 contact=unset"
        )
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": agent, "Accept-Encoding": "gzip, deflate"},
            auto_decompress=True,
        )

    async def close(self) -> None:
        await self._session.close()

    async def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> TimedResponse:
        start = now_pair()
        request_options: dict[str, Any] = {"params": params, "headers": headers}
        if timeout_seconds is not None:
            request_options["timeout"] = aiohttp.ClientTimeout(total=timeout_seconds)
        async with self._session.get(url, **request_options) as response:
            body = await response.read()
            end = now_pair()
            return TimedResponse(
                status=response.status,
                headers={k: v for k, v in response.headers.items()},
                body=body,
                request_started_wall_ns=start.wall_ns,
                request_started_mono_ns=start.mono_ns,
                response_received_wall_ns=end.wall_ns,
                response_received_mono_ns=end.mono_ns,
                url=str(response.url),
            )
