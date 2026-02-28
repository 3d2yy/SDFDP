#!/usr/bin/env python3
"""
Script de inicio para el Sistema de Detección de Descargas Parciales UHF

Uso:
    python start_gui.py              # Modo normal
    python start_gui.py --debug      # Modo debug
    python start_gui.py --port 8080  # Puerto personalizado
"""

import sys
import argparse
from app import app


def main():
    """Iniciar la aplicación."""
    parser = argparse.ArgumentParser(
        description='Sistema de Detección de Descargas Parciales UHF'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Ejecutar en modo debug'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Puerto para la aplicación web (default: 8050)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host para la aplicación web (default: 0.0.0.0)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SISTEMA DE DETECCIÓN DE DESCARGAS PARCIALES UHF")
    print("=" * 70)
    print()
    print(f"🚀 Iniciando aplicación web...")
    print(f"📊 Interfaz disponible en: http://localhost:{args.port}")
    print()
    
    if args.debug:
        print("⚠️  Modo DEBUG activado")
    
    print("✓ Sistema listo para operar")
    print("=" * 70)
    print()
    
    try:
        app.run(
            debug=args.debug,
            host=args.host,
            port=args.port
        )
    except KeyboardInterrupt:
        print("\n")
        print("=" * 70)
        print("🛑 Aplicación detenida por el usuario")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        print("\n")
        print("=" * 70)
        print(f"❌ Error al iniciar la aplicación: {e}")
        print("=" * 70)
        sys.exit(1)


if __name__ == '__main__':
    main()
