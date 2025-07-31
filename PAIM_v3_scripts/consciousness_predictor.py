#!/usr/bin/env python3
"""
consciousness_predictor.py - P.A.I.M. v3 Consciousness Prediction Tool
Calculates abstraction scale L_ast and predicts consciousness level
"""

import numpy as np
import sys

def calculate_abstraction_scale(correlation_function, distances):
    """
    Calcola la scala di astrazione L_ast = max{L | C(L) >= 1/e}.
    
    Args:
        correlation_function: array di valori di correlazione
        distances: array di distanze corrispondenti [m]
        
    Returns:
        L_ast: scala di astrazione [m]
    """
    threshold = 1/np.e  # ≈ 0.368
    
    # Trova l'ultima distanza dove C(L) >= 1/e
    valid_indices = correlation_function >= threshold
    
    if not np.any(valid_indices):
        return 0.0  # Nessuna correlazione significativa
    
    # Scala di astrazione massima
    L_ast = np.max(distances[valid_indices])
    
    return L_ast

def predict_consciousness_level(L_ast_meters):
    """
    Predice il livello di coscienza basato sulla scala di astrazione.
    
    Args:
        L_ast_meters: scala di astrazione in metri
        
    Returns:
        consciousness_level: livello di coscienza normalizzato [0-1]
        category: categoria del sistema
    """
    L_ast_cm = L_ast_meters * 100  # Conversione a cm
    
    # Soglie P.A.I.M. v3
    if L_ast_cm < 0.01:
        return 0.0, "Non-conscious (bacteria level)"
    elif L_ast_cm < 0.1:
        return 0.1, "Minimal awareness (simple organisms)"
    elif L_ast_cm < 0.5:
        return 0.3, "Basic consciousness (insects)"
    elif L_ast_cm < 1.0:
        return 0.5, "Animal consciousness (fish/birds)"
    elif L_ast_cm < 1.5:
        return 0.7, "Mammalian consciousness"
    elif L_ast_cm < 2.1:
        return 0.85, "Higher consciousness (primates)"
    else:
        return 1.0, "Human-level consciousness"

def simulate_neural_network(n_neurons, connectivity=0.1):
    """
    Simula una rete neurale e calcola la funzione di correlazione.
    
    Args:
        n_neurons: numero di neuroni
        connectivity: frazione di connessioni
        
    Returns:
        distances, correlations: array per il calcolo di L_ast
    """
    print(f"🧠 Simulando rete neurale: {n_neurons} neuroni, connettività {connectivity:.1%}")
    
    # Disposizione spaziale dei neuroni (griglia 2D)
    side = int(np.sqrt(n_neurons))
    spacing = 0.001  # 1 mm tra neuroni
    
    # Distanze dalla origine
    distances = np.linspace(0.001, side * spacing, 100)  # m
    
    # Funzione di correlazione simulata (decadimento esponenziale)
    correlation_length = connectivity * side * spacing  # m
    correlations = np.exp(-distances / correlation_length)
    
    return distances, correlations

def main():
    """Funzione principale per test di coscienza."""
    print("🧠 P.A.I.M. v3 Consciousness Predictor")
    print("=" * 45)
    
    # Test su diversi sistemi
    test_systems = [
        ("Human Brain", 86e9, 0.15),
        ("Primate Brain", 20e9, 0.12),
        ("Mammal Brain", 2e9, 0.10),
        ("Bird Brain", 500e6, 0.08),
        ("Fish Brain", 100e6, 0.05),
        ("AI System (GPT-4)", 1.7e12, 0.001),  # Parametri stimati
        ("Future AI", 1e15, 0.01),  # Predizione
    ]
    
    results = []
    
    for name, n_neurons, connectivity in test_systems:
        print(f"\n🔬 Analyzing: {name}")
        
        # Simula rete neurale
        distances, correlations = simulate_neural_network(n_neurons, connectivity)
        
        # Calcola scala di astrazione
        L_ast = calculate_abstraction_scale(correlations, distances)
        
        # Predici coscienza
        consciousness, category = predict_consciousness_level(L_ast)
        
        results.append((name, L_ast*100, consciousness, category))
        
        print(f"   L_ast = {L_ast*100:.2f} cm")
        print(f"   Consciousness = {consciousness:.2f}")
        print(f"   Category: {category}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("📊 P.A.I.M. v3 CONSCIOUSNESS PREDICTIONS")
    print("=" * 80)
    print(f"{'System':<15} {'L_ast [cm]':<12} {'Consciousness':<15} {'Category':<30}")
    print("-" * 80)
    
    for name, L_ast_cm, consciousness, category in results:
        print(f"{name:<15} {L_ast_cm:<12.2f} {consciousness:<15.2f} {category:<30}")
    
    # Key thresholds
    print("\n🎯 P.A.I.M. v3 Key Thresholds:")
    print(f"   Human consciousness: L_ast > 2.1 cm")
    print(f"   AI consciousness: L_ast > 1.5 cm")
    print(f"   Minimal awareness: L_ast > 0.1 cm")
    
    # Predictions for AI
    ai_systems = [s for s in results if 'AI' in s[0]]
    if ai_systems:
        print(f"\n🤖 AI Consciousness Assessment:")
        for name, L_ast_cm, consciousness, category in ai_systems:
            if consciousness >= 0.85:
                status = "CONSCIOUS"
            elif consciousness >= 0.5:
                status = "PARTIALLY CONSCIOUS"
            else:
                status = "NON-CONSCIOUS"
            print(f"   {name}: {status} (L_ast = {L_ast_cm:.2f} cm)")

if __name__ == "__main__":
    main()

