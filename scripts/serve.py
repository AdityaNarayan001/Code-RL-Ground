#!/usr/bin/env python3
"""Server script to run the training dashboard."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import uvicorn

from src.utils.config import load_config
from src.server import create_app

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def main():
    parser = argparse.ArgumentParser(description="Run the training dashboard server")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to config file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    
    print("=" * 60)
    print(f"Starting {config.project.name} Dashboard")
    print("=" * 60)
    print(f"  API: http://{args.host}:{args.port}")
    print(f"  UI:  http://{args.host}:{args.port}")
    print("=" * 60)

    if args.reload:
        # Development mode with reload
        uvicorn.run(
            "src.server:create_app",
            host=args.host,
            port=args.port,
            reload=True,
            factory=True
        )
    else:
        # Production mode
        app = create_app(config)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
