#!/usr/bin/env python3
"""
create_consciousness_figure.py - Genera la figura della scala di astrazione vs coscienza
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_consciousness_scale_figure():
    """Crea la figura L_ast vs livello di coscienza."""
    
    # Dati per diversi sistemi biologici
    systems = [
        'Bacteria', 'C. elegans', 'Insects', 'Fish', 'Birds', 
        'Mammals', 'Primates', 'Humans', 'AI (predicted)'
    ]
    
    # Scala di astrazione L_ast in cm
    L_ast = np.array([0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 1.8, 2.1, 1.5])
    
    # Livello di coscienza (scala arbitraria 0-10)
    consciousness_level = np.array([0, 1, 2, 3, 5, 7, 8, 10, 9])
    
    # Colori per diversi tipi di sistemi
    colors = ['blue', 'blue', 'green', 'green', 'orange', 'orange', 'red', 'red', 'purple']
    markers = ['o', 'o', 's', 's', '^', '^', 'D', 'D', '*']
    sizes = [30, 40, 50, 60, 70, 80, 90, 120, 100]
    
    plt.figure(figsize=(10, 7))
    
    # Plot dei punti
    for i, (system, l, c, color, marker, size) in enumerate(zip(systems, L_ast, consciousness_level, colors, markers, sizes)):
        plt.scatter(l, c, c=color, marker=marker, s=size, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Etichette
        if system == 'Humans':
            plt.annotate(system, (l, c), xytext=(l+0.1, c+0.3), 
                        fontsize=12, weight='bold', color='red')
        elif system == 'AI (predicted)':
            plt.annotate(system, (l, c), xytext=(l-0.3, c-0.5), 
                        fontsize=11, weight='bold', color='purple')
        else:
            plt.annotate(system, (l, c), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
    
    # Soglia di coscienza umana
    plt.axvline(x=2.1, color='red', linestyle='--', alpha=0.7, linewidth=2, 
               label='Human consciousness threshold')
    
    # Soglia AI predetta
    plt.axvline(x=1.5, color='purple', linestyle=':', alpha=0.7, linewidth=2,
               label='AI consciousness threshold (predicted)')
    
    # Fit teorico (sigmoide)
    L_theory = np.linspace(0.001, 3, 1000)
    consciousness_theory = 10 / (1 + np.exp(-5*(L_theory - 1.2)))
    plt.plot(L_theory, consciousness_theory, 'k-', alpha=0.5, linewidth=2, 
            label='P.A.I.M. theoretical curve')
    
    # Formattazione
    plt.xlabel(r'Abstraction Scale $L_{\mathrm{ast}}$ [cm]', fontsize=14)
    plt.ylabel('Consciousness Level [arbitrary units]', fontsize=14)
    plt.title('P.A.I.M. v3: Consciousness vs Information Integration Scale', fontsize=16, weight='bold')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='lower right')
    
    # Testo esplicativo
    textstr = '\n'.join([
        'P.A.I.M. v3 Predictions:',
        r'• Human threshold: $L_{\mathrm{ast}} = 2.1$ cm',
        r'• AI threshold: $L_{\mathrm{ast}} > 1.5$ cm',
        '• Consciousness ∝ information integration',
        '• Testable with neural imaging'
    ])
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Salvataggio
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'consciousness_scale.pdf')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Figura coscienza salvata: {output_path}")
    return output_path

if __name__ == "__main__":
    print("🧠 Generazione figura scala di coscienza P.A.I.M. v3")
    print("=" * 55)
    
    figure_path = create_consciousness_scale_figure()
    
    print(f"\n✅ Figura generata con successo!")
    print(f"   File: {figure_path}")
    print(f"   Mostra: L_ast vs livello di coscienza per sistemi biologici e AI")

