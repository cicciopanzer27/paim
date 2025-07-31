#!/usr/bin/env python3
"""
cosmo_check.py - Verifica della predizione cosmologica P.A.I.M.
Calcola I_th(z=0) da dati SPHEREx IR e confronta con predizione teorica
Utilizza protocollo di validazione numerica standard
"""

import numpy as np
import sys
import os

# Aggiungi il path per il modulo di validazione
sys.path.append('/home/ubuntu/validation')
from model_reliability import ModelReliabilityValidator

# Costanti fisiche
k_B = 1.380649e-23  # J/K
c = 299792458       # m/s
h = 6.62607015e-34  # J⋅s

def download_spherex_mock_data():
    """
    Simula il download di dati SPHEREx IR.
    In un caso reale, questo scaricherebbe i dati FITS da IRSA.
    """
    print("📡 Simulando download dati SPHEREx IR...")
    
    # Dati mock basati su osservazioni IR tipiche
    # Densità di energia IR di fondo in MJy/sr
    wavelengths = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0])  # μm
    flux_density = np.array([0.5, 1.2, 2.1, 1.8, 1.3, 0.9])  # MJy/sr
    
    # Conversione a densità di energia
    # E = h*c/λ per fotone, moltiplicato per densità di flusso
    energy_density = []
    for i, (wl, flux) in enumerate(zip(wavelengths, flux_density)):
        # Conversione da MJy/sr a J/m³
        # 1 Jy = 10^-26 W⋅m⁻²⋅Hz⁻¹
        # 1 MJy = 10^-20 W⋅m⁻²⋅Hz⁻¹
        freq = c / (wl * 1e-6)  # Hz
        energy_dens = flux * 1e-20 * freq / c  # J/m³
        energy_density.append(energy_dens)
    
    return np.array(wavelengths), np.array(energy_density)

def calculate_information_density(wavelengths, energy_density, T_cmb=2.725):
    """
    Calcola la densità di informazione strutturale I_th da dati IR.
    
    Args:
        wavelengths: array di lunghezze d'onda [μm]
        energy_density: array di densità di energia [J/m³]
        T_cmb: temperatura CMB [K]
    
    Returns:
        I_th: densità di informazione [bit/m³]
    """
    print(f"🧮 Calcolando I_th con T_CMB = {T_cmb} K...")
    
    # Calcolo dell'entropia di eccesso rispetto al CMB
    # ΔS_exch ≈ ρ_IR / T_CMB (approssimazione)
    total_energy_density = np.sum(energy_density)  # J/m³
    
    # Entropia di scambio per unità di volume
    delta_S_exch = total_energy_density / T_cmb  # J/(K⋅m³)
    
    # Entropia di von Neumann (assumiamo stato termico)
    # Per radiazione termica: S = (4/3) * σ * T³ / c
    sigma_sb = 5.670374419e-8  # W⋅m⁻²⋅K⁻⁴
    delta_S_vn = (4/3) * sigma_sb * T_cmb**3 / c  # J/(K⋅m³)
    
    # Informazione strutturale secondo P2
    delta_S = delta_S_exch - delta_S_vn  # J/(K⋅m³)
    I_th = delta_S / (k_B * np.log(2))  # bit/m³
    
    print(f"   ΔS_exch = {delta_S_exch:.2e} J/(K⋅m³)")
    print(f"   ΔS_vN   = {delta_S_vn:.2e} J/(K⋅m³)")
    print(f"   ΔS      = {delta_S:.2e} J/(K⋅m³)")
    
    return I_th

def main():
    """Funzione principale per il test cosmologico."""
    print("🌌 P.A.I.M. Cosmological Validation Test")
    print("=" * 50)
    
    try:
        # Download dati (simulati)
        wavelengths, energy_density = download_spherex_mock_data()
        
        print(f"📈 Dati scaricati:")
        print(f"   Bande spettrali: {len(wavelengths)}")
        print(f"   Range λ: {wavelengths.min():.1f} - {wavelengths.max():.1f} μm")
        print(f"   Energia totale: {np.sum(energy_density):.2e} J/m³")
        
        # Calcolo I_th
        I_th_measured = calculate_information_density(wavelengths, energy_density)
        
        # Validazione con protocollo standard
        validator = ModelReliabilityValidator(n_bootstrap=1000)
        
        # Parametri di validazione secondo todo.md sezione [5]
        I_th_predicted = 6.2e10  # bit/m³
        threshold = 0.1 * I_th_predicted  # 10% tolerance
        uncertainty = 0.05 * I_th_measured  # 5% incertezza sperimentale stimata
        
        result = validator.validate_prediction(
            prediction=I_th_predicted,
            measurement=I_th_measured,
            threshold=threshold,
            test_name="Cosmologia",
            measurement_uncertainty=uncertainty
        )
        
        # Report risultati
        print(f"\n📊 RISULTATI VALIDAZIONE PROTOCOLLO STANDARD:")
        print(f"   I_th misurato:  {result.measurement:.2e} bit/m³")
        print(f"   I_th predetto:  {result.prediction:.2e} bit/m³")
        print(f"   Errore:         {result.error:.2e} bit/m³")
        print(f"   Soglia:         {result.threshold:.2e} bit/m³")
        print(f"   P-value:        {result.p_value:.3f}")
        print(f"   CI 95%:         [{result.confidence_interval[0]:.2e}, {result.confidence_interval[1]:.2e}]")
        print(f"   Criterio:       P(|ε| < ε_max) ≥ 0.95")
        
        if result.is_valid:
            print("   ✅ PREDIZIONE VALIDATA (protocollo standard)")
        else:
            print("   ❌ PREDIZIONE FALSIFICATA (protocollo standard)")
            
        return result.is_valid
        
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")
        return False

def test_sensitivity():
    """Test di sensibilità per diversi parametri."""
    print("\n🔬 Test di sensibilità parametrica:")
    
    # Test con diverse temperature CMB
    temperatures = [2.5, 2.725, 3.0]
    wavelengths, energy_density = download_spherex_mock_data()
    
    validator = ModelReliabilityValidator(n_bootstrap=500)
    results = []
    
    for T in temperatures:
        I_th = calculate_information_density(wavelengths, energy_density, T)
        
        result = validator.validate_prediction(
            prediction=6.2e10,
            measurement=I_th,
            threshold=6.2e9,
            test_name=f"T_CMB={T}K",
            measurement_uncertainty=0.05 * I_th
        )
        
        results.append(result)
        print(f"   T = {T} K → I_th = {I_th:.2e} bit/m³, P-value = {result.p_value:.3f}")
    
    return results

if __name__ == "__main__":
    # Test principale
    success = main()
    
    # Test di sensibilità
    sensitivity_results = test_sensitivity()
    
    # Costo computazionale
    print(f"\n💰 Costo di falsificazione:")
    print(f"   Hardware: 0 USD (simulazione)")
    print(f"   Tempo: < 30 secondi")
    print(f"   Software: open-source")
    print(f"   Bootstrap: 1000 campioni")
    
    exit(0 if success else 1)

