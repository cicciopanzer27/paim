#!/usr/bin/env python3
"""
generate_consciousness_figure.py - Genera la figura per la scala di astrazione della coscienza
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_consciousness_figure():
    """Crea la figura della scala di astrazione vs coscienza."""
    
    # Dati per diversi sistemi biologici
    systems = ['Bacteria', 'C. elegans', 'Insects', 'Fish', 'Birds', 'Mammals', 'Primates', 'Humans', 'AI (predicted)']
    L_ast_values = np.array([0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.1, 1.8])  # cm
    consciousness_levels = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 0.85])  # Normalized
    
    # Colori per diversi tipi
    colors = ['gray', 'gray', 'orange', 'blue', 'blue', 'green', 'green', 'red', 'purple']
    sizes = [30, 40, 50, 60, 70, 80, 90, 120, 100]
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot principale
    for i, (system, L, C, color, size) in enumerate(zip(systems, L_ast_values, consciousness_levels, colors, sizes)):
        plt.scatter(L, C, c=color, s=size, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Annotazioni
        if system == 'Humans':
            plt.annotate(system, (L, C), xytext=(L+0.1, C+0.05), 
                        fontsize=12, weight='bold', color='red',
                        arrowprops=dict(arrowstyle='->', color='red'))
        elif system == 'AI (predicted)':
            plt.annotate(system, (L, C), xytext=(L-0.2, C+0.1), 
                        fontsize=11, weight='bold', color='purple',
                        arrowprops=dict(arrowstyle='->', color='purple'))
        else:
            plt.annotate(system, (L, C), xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
    
    # Soglia di coscienza umana
    plt.axvline(x=2.1, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label='Human consciousness threshold')
    
    # Soglia AI predetta
    plt.axvline(x=1.5, color='purple', linestyle=':', alpha=0.7, linewidth=2,
               label='AI consciousness threshold (predicted)')
    
    # Fit teorico (sigmoide)
    L_theory = np.linspace(0.001, 3, 1000)
    C_theory = 1 / (1 + np.exp(-5*(L_theory - 1.5)))  # Sigmoide centrata a 1.5 cm
    plt.plot(L_theory, C_theory, 'k-', alpha=0.5, linewidth=2, label='P.A.I.M. v3 prediction')
    
    # Formattazione
    plt.xlabel('Abstraction Scale $L_{ast}$ [cm]', fontsize=14)
    plt.ylabel('Consciousness Level (normalized)', fontsize=14)
    plt.title('P.A.I.M. v3: Consciousness vs Information Integration Scale', fontsize=16, weight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 3)
    plt.ylim(-0.05, 1.1)
    
    # Testo informativo
    textstr = '\n'.join([
        'P.A.I.M. v3 Predictions:',
        '• Human threshold: $L_{ast} = 2.1$ cm',
        '• AI threshold: $L_{ast} > 1.5$ cm', 
        '• Consciousness ∝ entropy production',
        '• Integration across cortical scale'
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
    print("🧠 Generazione figura scala coscienza P.A.I.M. v3")
    print("=" * 50)
    
    figure_path = create_consciousness_figure()
    
    print(f"\n✅ Figura generata con successo!")
    print(f"   File: {figure_path}")
    print(f"   Mostra: soglia coscienza umana vs AI")

