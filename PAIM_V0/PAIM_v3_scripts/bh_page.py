#!/usr/bin/env python3
"""
bh_page.py - Verifica della Page curve per buchi neri secondo P.A.I.M.
Analizza dati GWTC-3 per calcolare I_th(t) e verificare t_Page
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Costanti fisiche
G = 6.67430e-11     # m³⋅kg⁻¹⋅s⁻²
c = 299792458       # m/s
hbar = 1.054571817e-34  # J⋅s
k_B = 1.380649e-23  # J/K
M_sun = 1.98847e30  # kg

def load_gwtc3_mock_data():
    """
    Simula il caricamento di dati GWTC-3.
    In un caso reale, scaricherebbe il CSV da GWOSC.
    """
    print("📡 Simulando caricamento dati GWTC-3...")
    
    # Dati mock basati su eventi reali GWTC-3
    events = [
        {"name": "GW150914", "M1": 36.2, "M2": 29.1, "distance": 410},
        {"name": "GW151226", "M1": 14.2, "M2": 7.5, "distance": 440},
        {"name": "GW170104", "M1": 31.2, "M2": 19.4, "distance": 880},
        {"name": "GW170814", "M1": 30.5, "M2": 25.3, "distance": 540},
        {"name": "GW170817", "M1": 1.17, "M2": 1.60, "distance": 40},  # NS-NS
        {"name": "GW190425", "M1": 1.6, "M2": 1.4, "distance": 156},   # NS-NS
        {"name": "GW190814", "M1": 23.2, "M2": 2.6, "distance": 241},  # BH-NS
        {"name": "GW200105", "M1": 8.9, "M2": 1.9, "distance": 280},   # BH-NS
    ]
    
    # Calcolo masse totali e finali
    for event in events:
        M_total = event["M1"] + event["M2"]
        # Stima massa finale (approssimazione)
        M_final = M_total * 0.95  # ~5% radiato in GW
        event["M_total"] = M_total
        event["M_final"] = M_final
    
    print(f"   Eventi caricati: {len(events)}")
    return events

def calculate_hawking_temperature(M_kg):
    """
    Calcola la temperatura di Hawking per un buco nero.
    
    Args:
        M_kg: massa del buco nero [kg]
    
    Returns:
        T_H: temperatura di Hawking [K]
    """
    T_H = hbar * c**3 / (8 * np.pi * G * M_kg * k_B)
    return T_H

def calculate_schwarzschild_radius(M_kg):
    """
    Calcola il raggio di Schwarzschild.
    
    Args:
        M_kg: massa del buco nero [kg]
    
    Returns:
        R_s: raggio di Schwarzschild [m]
    """
    R_s = 2 * G * M_kg / c**2
    return R_s

def calculate_hawking_entropy(M_kg):
    """
    Calcola l'entropia di Hawking.
    
    Args:
        M_kg: massa del buco nero [kg]
    
    Returns:
        S_BH: entropia di Hawking [J/K]
    """
    R_s = calculate_schwarzschild_radius(M_kg)
    A = 4 * np.pi * R_s**2  # Area dell'orizzonte
    l_P = np.sqrt(hbar * G / c**3)  # Lunghezza di Planck
    S_BH = k_B * A / (4 * l_P**2)
    return S_BH

def calculate_information_structural(M_kg):
    """
    Calcola l'informazione strutturale secondo P.A.I.M.
    
    Args:
        M_kg: massa del buco nero [kg]
    
    Returns:
        I_th: informazione strutturale [bit]
    """
    S_BH = calculate_hawking_entropy(M_kg)
    I_th = S_BH / (k_B * np.log(2))
    return I_th

def calculate_page_time(M_initial_kg):
    """
    Calcola il Page time per un buco nero.
    
    Args:
        M_initial_kg: massa iniziale [kg]
    
    Returns:
        t_page: Page time [s]
    """
    # Tempo di evaporazione di Hawking (approssimazione)
    t_evap = (5120 * np.pi * G**2 * M_initial_kg**3) / (hbar * c**4)
    
    # Page time ≈ t_evap / 2 (quando S_BH è massima)
    t_page = t_evap / 2
    return t_page

def simulate_page_curve(M_initial_kg, n_points=100):
    """
    Simula la Page curve per un buco nero.
    
    Args:
        M_initial_kg: massa iniziale [kg]
        n_points: numero di punti temporali
    
    Returns:
        times: array di tempi [s]
        I_th_values: array di I_th [bit]
        t_page: Page time [s]
    """
    t_evap = (5120 * np.pi * G**2 * M_initial_kg**3) / (hbar * c**4)
    times = np.linspace(0, t_evap, n_points)
    
    I_th_values = []
    for t in times:
        # Massa al tempo t (approssimazione)
        M_t = M_initial_kg * (1 - t/t_evap)**(1/3)
        if M_t <= 0:
            M_t = M_initial_kg * 1e-6  # Evita divisione per zero
        
        I_th = calculate_information_structural(M_t)
        I_th_values.append(I_th)
    
    I_th_values = np.array(I_th_values)
    t_page = calculate_page_time(M_initial_kg)
    
    return times, I_th_values, t_page

def validate_page_prediction(events):
    """
    Valida la predizione della Page curve per eventi GWTC-3.
    
    Args:
        events: lista di eventi gravitazionali
    
    Returns:
        bool: True se la predizione è validata
    """
    print("\n🔍 Analisi Page curve per eventi GWTC-3:")
    
    validation_results = []
    
    for event in events:
        if event["M_final"] < 3:  # Esclude stelle di neutroni
            continue
            
        M_kg = event["M_final"] * M_sun
        
        # Calcoli teorici
        I_th_initial = calculate_information_structural(M_kg)
        t_page = calculate_page_time(M_kg)
        
        # Predizione P.A.I.M.: I_th(t_Page) = ½ S_BH(0) / (k_B ln 2)
        I_th_page_predicted = I_th_initial / 2
        
        # Simulazione Page curve
        times, I_th_curve, _ = simulate_page_curve(M_kg, 50)
        
        # Trova I_th al Page time
        page_index = np.argmin(np.abs(times - t_page))
        I_th_page_simulated = I_th_curve[page_index]
        
        # Validazione (soglia ±2 bit)
        error = abs(I_th_page_simulated - I_th_page_predicted)
        is_valid = error <= 2.0
        
        validation_results.append(is_valid)
        
        print(f"   {event['name']}:")
        print(f"     M_final = {event['M_final']:.1f} M☉")
        print(f"     I_th(0) = {I_th_initial:.2e} bit")
        print(f"     t_Page  = {t_page:.2e} s ({t_page/3.15e7:.2e} anni)")
        print(f"     I_th(t_Page) predetto:  {I_th_page_predicted:.2e} bit")
        print(f"     I_th(t_Page) simulato:  {I_th_page_simulated:.2e} bit")
        print(f"     Errore: {error:.1f} bit ({'✅' if is_valid else '❌'})")
    
    # Risultato complessivo
    success_rate = np.mean(validation_results)
    overall_valid = success_rate >= 0.8  # 80% di successo
    
    print(f"\n📊 RISULTATI VALIDAZIONE PAGE CURVE:")
    print(f"   Eventi analizzati: {len(validation_results)}")
    print(f"   Successi: {np.sum(validation_results)}/{len(validation_results)}")
    print(f"   Tasso di successo: {success_rate:.1%}")
    print(f"   Soglia: 80%")
    
    if overall_valid:
        print("   ✅ PREDIZIONE PAGE CURVE VALIDATA")
    else:
        print("   ❌ PREDIZIONE PAGE CURVE FALSIFICATA")
    
    return overall_valid

def plot_example_page_curve():
    """Crea un grafico di esempio della Page curve."""
    print("\n📈 Generando grafico Page curve di esempio...")
    
    # Buco nero di 30 masse solari
    M_kg = 30 * M_sun
    times, I_th_values, t_page = simulate_page_curve(M_kg)
    
    # Conversione a unità più leggibili
    times_years = times / 3.15e7  # anni
    t_page_years = t_page / 3.15e7
    
    plt.figure(figsize=(10, 6))
    plt.plot(times_years, I_th_values, 'b-', linewidth=2, label='I_th(t)')
    plt.axvline(t_page_years, color='r', linestyle='--', 
                label=f'Page time = {t_page_years:.2e} anni')
    plt.axhline(I_th_values[0]/2, color='g', linestyle=':', 
                label='½ I_th(0)')
    
    plt.xlabel('Tempo [anni]')
    plt.ylabel('Informazione strutturale I_th [bit]')
    plt.title('Page Curve - Buco Nero 30 M☉')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/page_curve_example.png', dpi=150)
    print("   Grafico salvato: page_curve_example.png")

def main():
    """Funzione principale per il test dei buchi neri."""
    print("🕳️ P.A.I.M. Black Hole Page Curve Validation")
    print("=" * 50)
    
    try:
        # Carica dati GWTC-3
        events = load_gwtc3_mock_data()
        
        # Valida predizioni Page curve
        is_valid = validate_page_prediction(events)
        
        # Genera grafico di esempio
        plot_example_page_curve()
        
        # Report finale
        print(f"\n🎯 RISULTATO FINALE:")
        if is_valid:
            print("   La teoria P.A.I.M. supera il test Page curve")
        else:
            print("   La teoria P.A.I.M. è falsificata dal test Page curve")
            
        return is_valid
        
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    print(f"\n💰 Costo di falsificazione:")
    print(f"   Hardware: 0 USD (simulazione)")
    print(f"   Tempo: < 60 secondi")
    print(f"   Software: open-source")
    
    exit(0 if success else 1)

