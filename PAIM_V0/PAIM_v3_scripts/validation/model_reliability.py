#!/usr/bin/env python3
"""
model_reliability.py - Protocollo standard di validazione numerica P.A.I.M.
Implementa model reliability metric con bootstrap per validazione statistica
"""

import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Risultato di una validazione numerica."""
    prediction: float
    measurement: float
    error: float
    threshold: float
    p_value: float
    is_valid: bool
    confidence_interval: Tuple[float, float]
    test_name: str

class ModelReliabilityValidator:
    """Validatore per il protocollo standard P.A.I.M."""
    
    def __init__(self, n_bootstrap=10000, confidence_level=0.95):
        """
        Inizializza il validatore.
        
        Args:
            n_bootstrap: numero di campioni bootstrap
            confidence_level: livello di confidenza (default 95%)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def calculate_error(self, prediction: float, measurement: float) -> float:
        """
        Calcola la deviazione ε = |predizione - misurazione|.
        
        Args:
            prediction: valore predetto dalla teoria
            measurement: valore misurato sperimentalmente
            
        Returns:
            error: deviazione assoluta
        """
        return abs(prediction - measurement)
    
    def bootstrap_error_distribution(self, 
                                   predictions: np.ndarray, 
                                   measurements: np.ndarray) -> np.ndarray:
        """
        Genera distribuzione bootstrap degli errori.
        
        Args:
            predictions: array di predizioni
            measurements: array di misurazioni
            
        Returns:
            bootstrap_errors: distribuzione bootstrap degli errori
        """
        n_samples = len(predictions)
        bootstrap_errors = []
        
        for _ in range(self.n_bootstrap):
            # Campionamento bootstrap
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            pred_boot = predictions[indices]
            meas_boot = measurements[indices]
            
            # Calcolo errore medio per questo campione
            errors = np.abs(pred_boot - meas_boot)
            mean_error = np.mean(errors)
            bootstrap_errors.append(mean_error)
        
        return np.array(bootstrap_errors)
    
    def validate_prediction(self, 
                          prediction: float, 
                          measurement: float, 
                          threshold: float,
                          test_name: str = "Unknown",
                          measurement_uncertainty: float = 0.0) -> ValidationResult:
        """
        Valida una singola predizione secondo il protocollo standard.
        
        Args:
            prediction: valore predetto
            measurement: valore misurato
            threshold: soglia di accettazione
            test_name: nome del test
            measurement_uncertainty: incertezza sperimentale
            
        Returns:
            ValidationResult: risultato della validazione
        """
        # Calcolo errore base
        error = self.calculate_error(prediction, measurement)
        
        # Se c'è incertezza sperimentale, genera distribuzione
        if measurement_uncertainty > 0:
            # Genera campioni dalla distribuzione dell'incertezza
            measurements_dist = np.random.normal(
                measurement, measurement_uncertainty, self.n_bootstrap
            )
            predictions_dist = np.full(self.n_bootstrap, prediction)
            
            # Bootstrap sulla distribuzione
            bootstrap_errors = self.bootstrap_error_distribution(
                predictions_dist, measurements_dist
            )
        else:
            # Bootstrap semplice (replica il singolo valore)
            bootstrap_errors = np.full(self.n_bootstrap, error)
        
        # Calcolo probabilità P(|ε| < ε_max)
        p_value = np.mean(bootstrap_errors < threshold)
        
        # Intervallo di confidenza per l'errore
        ci_lower = np.percentile(bootstrap_errors, 100 * self.alpha / 2)
        ci_upper = np.percentile(bootstrap_errors, 100 * (1 - self.alpha / 2))
        
        # Validazione: P(|ε| < ε_max) ≥ 0.95
        is_valid = p_value >= 0.95
        
        return ValidationResult(
            prediction=prediction,
            measurement=measurement,
            error=error,
            threshold=threshold,
            p_value=p_value,
            is_valid=is_valid,
            confidence_interval=(ci_lower, ci_upper),
            test_name=test_name
        )
    
    def validate_multiple_predictions(self, 
                                    predictions: List[float],
                                    measurements: List[float],
                                    thresholds: List[float],
                                    test_names: List[str],
                                    uncertainties: List[float] = None) -> List[ValidationResult]:
        """
        Valida multiple predizioni.
        
        Args:
            predictions: lista di predizioni
            measurements: lista di misurazioni
            thresholds: lista di soglie
            test_names: lista di nomi dei test
            uncertainties: lista di incertezze (opzionale)
            
        Returns:
            List[ValidationResult]: risultati delle validazioni
        """
        if uncertainties is None:
            uncertainties = [0.0] * len(predictions)
        
        results = []
        for pred, meas, thresh, name, unc in zip(
            predictions, measurements, thresholds, test_names, uncertainties
        ):
            result = self.validate_prediction(pred, meas, thresh, name, unc)
            results.append(result)
        
        return results
    
    def generate_validation_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Genera un report di validazione completo.
        
        Args:
            results: lista di risultati di validazione
            
        Returns:
            Dict: report completo
        """
        n_total = len(results)
        n_valid = sum(1 for r in results if r.is_valid)
        success_rate = n_valid / n_total if n_total > 0 else 0
        
        # Statistiche degli errori
        errors = [r.error for r in results]
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # P-values
        p_values = [r.p_value for r in results]
        mean_p_value = np.mean(p_values)
        
        report = {
            "summary": {
                "total_tests": n_total,
                "valid_tests": n_valid,
                "success_rate": success_rate,
                "overall_valid": success_rate >= 0.8  # 80% soglia
            },
            "error_statistics": {
                "mean_error": mean_error,
                "std_error": std_error,
                "max_error": max(errors) if errors else 0,
                "min_error": min(errors) if errors else 0
            },
            "p_value_statistics": {
                "mean_p_value": mean_p_value,
                "min_p_value": min(p_values) if p_values else 0
            },
            "individual_results": [
                {
                    "test_name": r.test_name,
                    "prediction": r.prediction,
                    "measurement": r.measurement,
                    "error": r.error,
                    "threshold": r.threshold,
                    "p_value": r.p_value,
                    "is_valid": r.is_valid,
                    "confidence_interval": r.confidence_interval
                }
                for r in results
            ]
        }
        
        return report
    
    def plot_validation_results(self, results: List[ValidationResult], 
                              save_path: str = None):
        """
        Crea grafici dei risultati di validazione.
        
        Args:
            results: risultati di validazione
            save_path: percorso per salvare il grafico
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Errori vs soglie
        errors = [r.error for r in results]
        thresholds = [r.threshold for r in results]
        colors = ['green' if r.is_valid else 'red' for r in results]
        
        ax1.scatter(range(len(results)), errors, c=colors, alpha=0.7)
        ax1.plot(range(len(results)), thresholds, 'k--', label='Soglie')
        ax1.set_xlabel('Test #')
        ax1.set_ylabel('Errore')
        ax1.set_title('Errori vs Soglie')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribuzione P-values
        p_values = [r.p_value for r in results]
        ax2.hist(p_values, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(0.95, color='red', linestyle='--', label='Soglia P=0.95')
        ax2.set_xlabel('P-value')
        ax2.set_ylabel('Frequenza')
        ax2.set_title('Distribuzione P-values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Predizioni vs Misurazioni
        predictions = [r.prediction for r in results]
        measurements = [r.measurement for r in results]
        
        ax3.scatter(predictions, measurements, c=colors, alpha=0.7)
        min_val = min(min(predictions), min(measurements))
        max_val = max(max(predictions), max(measurements))
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')
        ax3.set_xlabel('Predizioni')
        ax3.set_ylabel('Misurazioni')
        ax3.set_title('Predizioni vs Misurazioni')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Tasso di successo
        test_names = [r.test_name for r in results]
        success = [1 if r.is_valid else 0 for r in results]
        
        ax4.bar(range(len(results)), success, color=colors, alpha=0.7)
        ax4.set_xlabel('Test #')
        ax4.set_ylabel('Successo (1=Sì, 0=No)')
        ax4.set_title('Risultati per Test')
        ax4.set_xticks(range(len(results)))
        ax4.set_xticklabels([name[:10] for name in test_names], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Grafico salvato: {save_path}")
        
        return fig

def demo_validation():
    """Dimostra l'uso del validatore con dati di esempio."""
    print("🧪 Demo Protocollo Validazione P.A.I.M.")
    print("=" * 50)
    
    # Crea validatore
    validator = ModelReliabilityValidator(n_bootstrap=1000)
    
    # Dati di esempio (simulano i test P.A.I.M.)
    test_data = [
        {
            "name": "Cosmologia",
            "prediction": 6.2e10,
            "measurement": 5.8e10,
            "threshold": 6.2e9,  # 10% di 6.2e10
            "uncertainty": 0.5e10
        },
        {
            "name": "Buchi Neri",
            "prediction": 1.0e50,
            "measurement": 1.1e50,
            "threshold": 2.0,  # ±2 bit (in scala log)
            "uncertainty": 0.1e50
        },
        {
            "name": "Neutrini",
            "prediction": 2.4e-3,
            "measurement": 2.1e-3,
            "threshold": 0.6e-3,
            "uncertainty": 0.2e-3
        }
    ]
    
    # Esegui validazioni
    results = []
    for data in test_data:
        result = validator.validate_prediction(
            prediction=data["prediction"],
            measurement=data["measurement"],
            threshold=data["threshold"],
            test_name=data["name"],
            measurement_uncertainty=data["uncertainty"]
        )
        results.append(result)
    
    # Genera report
    report = validator.generate_validation_report(results)
    
    # Stampa risultati
    print("\n📊 RISULTATI VALIDAZIONE:")
    print(f"   Test totali: {report['summary']['total_tests']}")
    print(f"   Test validi: {report['summary']['valid_tests']}")
    print(f"   Tasso successo: {report['summary']['success_rate']:.1%}")
    print(f"   Validazione generale: {'✅' if report['summary']['overall_valid'] else '❌'}")
    
    print(f"\n📈 STATISTICHE ERRORI:")
    print(f"   Errore medio: {report['error_statistics']['mean_error']:.2e}")
    print(f"   Deviazione std: {report['error_statistics']['std_error']:.2e}")
    
    print(f"\n🎯 RISULTATI INDIVIDUALI:")
    for result in results:
        print(f"   {result.test_name}:")
        print(f"     Errore: {result.error:.2e}")
        print(f"     P-value: {result.p_value:.3f}")
        print(f"     Valido: {'✅' if result.is_valid else '❌'}")
    
    # Crea grafici
    validator.plot_validation_results(results, '/home/ubuntu/validation_results.png')
    
    return report['summary']['overall_valid']

if __name__ == "__main__":
    success = demo_validation()
    exit(0 if success else 1)

