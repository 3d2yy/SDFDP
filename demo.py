#!/usr/bin/env python3
"""
Demo del Sistema de Detección de Descargas Parciales

Este script muestra las capacidades del sistema en la terminal.
"""

import numpy as np
import time
from main import generate_synthetic_signal, process_and_analyze_signal

def print_banner():
    """Mostrar banner del sistema."""
    print("\n" + "=" * 70)
    print("   🔌 SISTEMA DE DETECCIÓN DE DESCARGAS PARCIALES UHF - DEMO")
    print("=" * 70)


def print_signal_info(state, result):
    """Mostrar información de la señal analizada."""
    colors = {
        'verde': '\033[92m',
        'amarillo': '\033[93m',
        'naranja': '\033[93m',
        'rojo': '\033[91m'
    }
    symbols = {
        'verde': '🟢',
        'amarillo': '🟡',
        'naranja': '🟠',
        'rojo': '🔴'
    }
    reset = '\033[0m'
    
    detected_state = result['traffic_light_state']
    color = colors.get(detected_state, '')
    symbol = symbols.get(detected_state, '⚪')
    
    print(f"\n{'-' * 70}")
    print(f"Estado Generado:  {state.upper()}")
    print(f"Estado Detectado: {color}{symbol} {detected_state.upper()}{reset}")
    print(f"Índice Severidad: {result['severity_index']:.4f}")
    print(f"{'-' * 70}")
    
    desc = result['descriptors']
    print(f"\nDescriptores Principales:")
    print(f"  • Energía Total:        {desc['energy_total']:.6f}")
    print(f"  • RMS:                  {desc['rms']:.6f}")
    print(f"  • Conteo de Picos:      {desc['peak_count']}")
    print(f"  • Factor de Cresta:     {desc['crest_factor']:.4f}")
    print(f"  • Entropía Espectral:   {desc['spectral_entropy']:.4f}")


def demo_classification():
    """Demostración de clasificación de diferentes estados."""
    print_banner()
    print("\n📊 Demostración de Clasificación Automática")
    print("\nGenerando y analizando señales de diferentes estados...\n")
    
    states = ['verde', 'amarillo', 'naranja', 'rojo']
    fs = 10000
    
    for state in states:
        print(f"\n⏳ Procesando estado: {state.upper()}...", end=' ')
        
        # Generar señal
        signal = generate_synthetic_signal(state, duration=1000, fs=fs)
        
        # Analizar
        result = process_and_analyze_signal(signal, fs)
        
        print("✓")
        
        # Mostrar resultados
        print_signal_info(state, result)
        
        # Pequeña pausa para efecto
        time.sleep(1)
    
    print("\n" + "=" * 70)
    print("✅ Demostración completada")
    print("=" * 70)


def demo_progression():
    """Demostración de progresión de deterioro."""
    print_banner()
    print("\n📈 Demostración de Progresión de Deterioro")
    print("\nSimulando degradación progresiva del equipo...\n")
    
    fs = 10000
    
    # Configuraciones progresivas
    configs = [
        ('Normal', 'verde', 0.05),
        ('Leve deterioro', 'verde', 0.10),
        ('Deterioro moderado', 'amarillo', 0.15),
        ('Deterioro avanzado', 'naranja', 0.20),
        ('Falla inminente', 'rojo', 0.25),
    ]
    
    print("Tiempo | Estado    | Severidad | Picos | Energía")
    print("-" * 70)
    
    for i, (label, state, noise) in enumerate(configs):
        # Generar señal con ruido creciente
        signal = generate_synthetic_signal(state, duration=1000, fs=fs, noise_level=noise)
        result = process_and_analyze_signal(signal, fs)
        
        symbols = {'verde': '🟢', 'amarillo': '🟡', 'naranja': '🟠', 'rojo': '🔴'}
        symbol = symbols.get(result['traffic_light_state'], '⚪')
        
        print(f"T+{i:02d}h  | {symbol} {result['traffic_light_state']:<8s} | "
              f"{result['severity_index']:9.4f} | "
              f"{result['descriptors']['peak_count']:5d} | "
              f"{result['descriptors']['energy_total']:8.2f}")
        
        time.sleep(0.5)
    
    print("-" * 70)
    print("\n⚠️  Observe cómo aumenta la severidad con el tiempo")
    print("=" * 70)


def demo_comparison():
    """Demostración de comparación de algoritmos."""
    print_banner()
    print("\n🔬 Demostración de Algoritmos de Procesamiento")
    print("\nComparando diferentes técnicas de filtrado...\n")
    
    fs = 10000
    signal = generate_synthetic_signal('naranja', duration=1000, fs=fs)
    result = process_and_analyze_signal(signal, fs)
    
    print("Algoritmos Ciegos:")
    print("-" * 70)
    
    blind = result['blind_algorithms']
    for alg_name, alg_result in sorted(blind.items()):
        print(f"  {alg_name:<25s} Score: {alg_result['score']:8.4f}")
    
    print("-" * 70)
    print("\n💡 Score más bajo = Mejor filtrado de ruido")
    print("=" * 70)


def main_menu():
    """Menú principal de demos."""
    while True:
        print_banner()
        print("\nSeleccione una demostración:\n")
        print("  1. Clasificación Automática")
        print("  2. Progresión de Deterioro")
        print("  3. Comparación de Algoritmos")
        print("  4. Todas las Demos")
        print("  5. Iniciar Interfaz Gráfica")
        print("  0. Salir\n")
        
        try:
            choice = input("Opción: ").strip()
            
            if choice == '1':
                demo_classification()
            elif choice == '2':
                demo_progression()
            elif choice == '3':
                demo_comparison()
            elif choice == '4':
                demo_classification()
                input("\nPresione Enter para continuar...")
                demo_progression()
                input("\nPresione Enter para continuar...")
                demo_comparison()
            elif choice == '5':
                print("\n🚀 Iniciando interfaz gráfica...")
                print("📊 Interfaz disponible en: http://localhost:8050\n")
                import subprocess
                subprocess.run(["python", "app.py"])
                break
            elif choice == '0':
                print("\n👋 ¡Hasta luego!\n")
                break
            else:
                print("\n❌ Opción inválida")
            
            if choice in ['1', '2', '3', '4']:
                input("\n\nPresione Enter para volver al menú...")
                
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break


if __name__ == '__main__':
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego!\n")
