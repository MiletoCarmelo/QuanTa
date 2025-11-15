import polars as pl
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats


class RiskClient:
    """Client for calculating risk metrics: VaR (Value at Risk) and ES (Expected Shortfall)."""
    
    def __init__(self):
        pass
    
    def calculate_var(self, 
                     pnl: pl.Series, 
                     confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).
        
        VaR represents the maximum potential loss at a given confidence level.
        For example, VaR(95%) = -0.05 means we expect losses to exceed 5% 
        in only 5% of cases.
        
        Args:
            pnl: Series of PnL values (profits and losses)
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: Method to use ('historical', 'parametric', 'monte_carlo')
        
        Returns:
            VaR value (negative number representing potential loss)
        """
        if len(pnl) == 0:
            return 0.0
        
        pnl_values = pnl.to_numpy()
        
        if method == 'historical':
            # Historical simulation: use actual percentile
            var = np.percentile(pnl_values, (1 - confidence_level) * 100)
            return float(var)
        
        elif method == 'parametric':
            # Parametric (Gaussian) method: assume normal distribution
            mean = np.mean(pnl_values)
            std = np.std(pnl_values)
            # Use inverse CDF of normal distribution
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean + z_score * std
            return float(var)
        
        elif method == 'monte_carlo':
            # Monte Carlo simulation (simplified)
            mean = np.mean(pnl_values)
            std = np.std(pnl_values)
            # Generate random samples from fitted distribution
            samples = np.random.normal(mean, std, size=10000)
            var = np.percentile(samples, (1 - confidence_level) * 100)
            return float(var)
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'historical', 'parametric', or 'monte_carlo'")
    
    def calculate_es(self, 
                    pnl: pl.Series, 
                    confidence_level: float = 0.95,
                    method: str = 'historical') -> float:
        """
        Calculate Expected Shortfall (ES) / Conditional VaR (CVaR).
        
        ES is the average loss given that the loss exceeds VaR.
        It's a more conservative risk measure than VaR.
        
        Args:
            pnl: Series of PnL values (profits and losses)
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: Method to use ('historical', 'parametric')
        
        Returns:
            ES value (negative number representing average loss beyond VaR)
        """
        if len(pnl) == 0:
            return 0.0
        
        pnl_values = pnl.to_numpy()
        
        if method == 'historical':
            # Historical simulation: average of losses beyond VaR
            var = self.calculate_var(pnl, confidence_level, method='historical')
            # Filter losses that exceed VaR
            tail_losses = pnl_values[pnl_values <= var]
            if len(tail_losses) == 0:
                return var  # If no tail losses, return VaR
            es = np.mean(tail_losses)
            return float(es)
        
        elif method == 'parametric':
            # Parametric method: use analytical formula for normal distribution
            mean = np.mean(pnl_values)
            std = np.std(pnl_values)
            # ES for normal distribution
            z_score = stats.norm.ppf(1 - confidence_level)
            # Expected value of truncated normal distribution
            es = mean - std * (stats.norm.pdf(z_score) / (1 - confidence_level))
            return float(es)
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'historical' or 'parametric'")
    
    def calculate_risk_metrics(self, 
                              pnl: pl.Series,
                              confidence_levels: Optional[list] = None) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            pnl: Series of PnL values
            confidence_levels: List of confidence levels (default: [0.90, 0.95, 0.99])
        
        Returns:
            Dictionary with risk metrics
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        metrics = {}
        
        # Basic statistics
        pnl_values = pnl.to_numpy()
        metrics['mean'] = float(np.mean(pnl_values))
        metrics['std'] = float(np.std(pnl_values))
        metrics['min'] = float(np.min(pnl_values))
        metrics['max'] = float(np.max(pnl_values))
        metrics['skewness'] = float(stats.skew(pnl_values))
        metrics['kurtosis'] = float(stats.kurtosis(pnl_values))
        
        # VaR and ES for each confidence level
        for cl in confidence_levels:
            var_hist = self.calculate_var(pnl, cl, method='historical')
            var_param = self.calculate_var(pnl, cl, method='parametric')
            es_hist = self.calculate_es(pnl, cl, method='historical')
            es_param = self.calculate_es(pnl, cl, method='parametric')
            
            metrics[f'VaR_{int(cl*100)}_historical'] = var_hist
            metrics[f'VaR_{int(cl*100)}_parametric'] = var_param
            metrics[f'ES_{int(cl*100)}_historical'] = es_hist
            metrics[f'ES_{int(cl*100)}_parametric'] = es_param
        
        return metrics
    
    def plot_pnl_distribution(self, pnl: pl.Series) -> Dict[str, np.ndarray]:
        """
        Prepare data for PnL distribution histogram.
        
        Args:
            pnl: Series of PnL values
        
        Returns:
            Dictionary with histogram data (bins, counts, bin_centers)
        """
        if len(pnl) == 0:
            return {'bins': np.array([]), 'counts': np.array([]), 'bin_centers': np.array([])}
        
        pnl_values = pnl.to_numpy()
        
        # Use automatic binning (Freedman-Diaconis rule or similar)
        counts, bins = np.histogram(pnl_values, bins='auto')
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        return {
            'bins': bins,
            'counts': counts,
            'bin_centers': bin_centers
        }

