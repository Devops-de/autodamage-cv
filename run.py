"""
Entry point to launch the ClaimScope application.
Runs Uvicorn web server hosting FastAPI backend and static frontend.
"""

import ipaddress
import os

import uvicorn


def _resolve_display_host(host: str) -> str:
    if host == "localhost":
        return "localhost"
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_loopback or addr.is_unspecified:
            return "localhost"
    except ValueError:
        pass
    return host


if __name__ == "__main__":
    host = os.getenv("CLAIMSCOPE_HOST", "127.0.0.1")
    port = int(os.getenv("CLAIMSCOPE_PORT", "8000"))
    reload_enabled = os.getenv("CLAIMSCOPE_RELOAD", "false").lower() == "true"
    display_host = _resolve_display_host(host)

    print("=" * 70)
    print("  ClaimScope - Vehicle Damage & Collision Appraisal Engine")
    print("  Backend: OpenCV Signal Processing Pipeline")
    print("  Actuarial Standards: Mitchell / CCC ONE US Collision Guidelines")
    print("=" * 70)
    print(f"  Server launching at: http://{display_host}:{port}")
    print(f"  API Docs available at: http://{display_host}:{port}/docs")
    print("=" * 70)

    uvicorn.run("app.main:app", host=host, port=port, reload=reload_enabled)
