"""
WSGI entry-point for production deployment.

Usage:
  gunicorn wsgi:server -b 0.0.0.0:8050 --workers 2 --timeout 120
"""

from app import server  # noqa: F401 – re-export

if __name__ == "__main__":
    server.run()
