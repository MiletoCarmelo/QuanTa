import optuna
from optuna.samplers import TPESampler
from quanta.utils.ta import TAClient, INDICATOR_CLASSES
from quanta.ta_clients.optimization_strategy import STRATEGIES_IMPLEMENTED
import polars as pl
from typing import Dict, Any, List, Callable, Tuple, Union
from pathlib import Path
from optuna.importance import get_param_importances
import re

class OptimizationClient:
    """Client pour optimiser des stratégies de trading avec Optuna."""

    def __init__(self, optimization_config: Dict[str, Any]):
        """
        Initialise le client d'optimisation.
        
        Args:
            optimization_config: Configuration d'optimisation (dict ou chemin vers JSON)
        """
        
        # Extraire les classes d'indicateurs et la configuration d'optimisation
        self.optimization_config = optimization_config
        self.study = None
        
        # Valider que la stratégie existe
        strategy_name = optimization_config.get("strategy_name")
        if not strategy_name:
            raise ValueError("'strategy_name' manquant dans la config")
        
        if strategy_name not in STRATEGIES_IMPLEMENTED:
            raise ValueError(f"Stratégie '{strategy_name}' inconnue")
        
        # Valider que les indicateurs requis sont bien configurés
        required = STRATEGIES_IMPLEMENTED[strategy_name]['required_indicators']
        configured = set(optimization_config['indicators'].keys())
        
        # Filtrer automatiquement la config pour ne garder que les indicateurs requis
        filtered_indicators = {
            ind_name: optimization_config['indicators'][ind_name] 
            for ind_name in required 
            if ind_name in optimization_config['indicators']
        }
        
        # Vérifier qu'on a bien tous les indicateurs requis
        missing = set(required) - set(filtered_indicators.keys())
        if missing:
            raise ValueError(f"Indicateurs manquants pour {strategy_name}: {missing}")
        
        # Afficher les indicateurs ignorés
        ignored = configured - set(required)
        if ignored:
            print(f"ℹ️  Indicateurs ignorés (non utilisés par '{strategy_name}'): {ignored}")
        
        self.optimization_config['indicators'] = filtered_indicators

        # NOUVEAU : Filtrer aussi les contraintes pour ne garder que celles valides
        filtered_constraints = []
        for constraint in self.optimization_config.get('constraints', []):
            condition = constraint['condition']
            # Vérifier si tous les paramètres de la contrainte sont dans les indicateurs requis
            is_valid = True
            for ind_name in configured:  # Tous les indicateurs de la config originale
                if ind_name not in required and ind_name in condition:
                    is_valid = False
                    break
            
            if is_valid:
                filtered_constraints.append(constraint)
            else:
                print(f"ℹ️  Contrainte ignorée (indicateur non utilisé): {condition}")

        self.optimization_config['constraints'] = filtered_constraints

        print(f"✓ Stratégie '{strategy_name}' initialisée avec les indicateurs: {list(filtered_indicators.keys())}")
        

    def _parse_constraints(self) -> Dict[str, List[Tuple[str, Union[str, float]]]]:
        """
        Parse les contraintes pour identifier les dépendances entre paramètres.
        Supporte:
        - param1 < param2 (comparaison entre paramètres)
        - param1 < 50 (comparaison avec constante)
        - 30 < param1 < 70 (à implémenter si nécessaire)
        
        Returns:
            Dict mapping chaque paramètre à ses contraintes de type (operator, other_param_or_value)
            Exemple: {
                'SMA_long_period': [('<', 'SMA_short_period')],
                'RSI_period': [('>', 5), ('<', 25)]
            }
        """
        constraints_map = {}
        
        for constraint in self.optimization_config.get('constraints', []):
            condition = constraint['condition']
            
            # Parser la contrainte avec regex
            # Supporte: A < B, A > 50, 30 < A, etc.
            pattern = r'([\w.]+)\s*([<>=!]+)\s*([\w.]+)'
            match = re.match(pattern, condition)
            
            if match:
                left = match.group(1)
                operator = match.group(2)
                right = match.group(3)
                
                # Déterminer si left et right sont des paramètres ou des constantes
                def is_number(s):
                    try:
                        float(s)
                        return True
                    except ValueError:
                        return False
                
                left_is_param = not is_number(left)
                right_is_param = not is_number(right)
                
                # Inverser l'opérateur
                inverse_ops = {
                    '<': '>',
                    '>': '<',
                    '<=': '>=',
                    '>=': '<=',
                    '==': '=='
                }
                
                # Cas 1: param < param (ou param > param, etc.)
                if left_is_param and right_is_param:
                    if left not in constraints_map:
                        constraints_map[left] = []
                    constraints_map[left].append((operator, right))
                    
                    if right not in constraints_map:
                        constraints_map[right] = []
                    constraints_map[right].append((inverse_ops.get(operator, operator), left))
                
                # Cas 2: param < 50 (paramètre comparé à constante)
                elif left_is_param and not right_is_param:
                    if left not in constraints_map:
                        constraints_map[left] = []
                    constraints_map[left].append((operator, float(right)))
                
                # Cas 3: 30 < param (constante comparée à paramètre)
                elif not left_is_param and right_is_param:
                    if right not in constraints_map:
                        constraints_map[right] = []
                    # Inverser: 30 < param → param > 30
                    constraints_map[right].append((inverse_ops.get(operator, operator), float(left)))
        
        return constraints_map

    def _apply_constraint_to_range(
        self,
        param_name: str,
        low: Union[int, float],
        high: Union[int, float],
        suggested_values: Dict[str, Union[int, float]]
    ) -> Tuple[Union[int, float], Union[int, float]]:
        """
        Ajuste les bornes d'un paramètre en fonction des contraintes et des valeurs déjà suggérées.
        
        Args:
            param_name: Nom du paramètre
            low: Borne inférieure originale
            high: Borne supérieure originale
            suggested_values: Dict des valeurs déjà suggérées
        
        Returns:
            Nouvelles bornes (low, high) respectant les contraintes
        """
        if not hasattr(self, '_constraints_map'):
            self._constraints_map = self._parse_constraints()
        
        constraints = self._constraints_map.get(param_name, [])
        
        for operator, other in constraints:
            # Déterminer si 'other' est un paramètre ou une valeur
            if isinstance(other, (int, float)):
                # C'est une constante
                other_value = other
            elif other in suggested_values:
                # C'est un paramètre déjà suggéré
                other_value = suggested_values[other]
            else:
                # Paramètre pas encore suggéré, on ne peut pas appliquer cette contrainte
                continue
            
            # Appliquer la contrainte
            if operator == '<':
                # param < other → param doit être < other
                if isinstance(low, int) and isinstance(other_value, (int, float)):
                    high = min(high, int(other_value) - 1)
                else:
                    high = min(high, other_value - 0.001)  # Pour les floats
            elif operator == '>':
                # param > other → param doit être > other
                if isinstance(low, int) and isinstance(other_value, (int, float)):
                    low = max(low, int(other_value) + 1)
                else:
                    low = max(low, other_value + 0.001)
            elif operator == '<=':
                if isinstance(low, int):
                    high = min(high, int(other_value))
                else:
                    high = min(high, other_value)
            elif operator == '>=':
                if isinstance(low, int):
                    low = max(low, int(other_value))
                else:
                    low = max(low, other_value)
            elif operator == '==':
                low = high = other_value
        
        # S'assurer que low <= high
        if low > high:
            # Contrainte impossible à satisfaire
            return (low, low - 1)
        
        return (low, high)

    def suggest_params_from_config(self, trial: optuna.Trial, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère les suggestions de paramètres en respectant dynamiquement les contraintes.
        Version CORRIGÉE : Suggestion PUIS contraintes.
        """
        params = {
            'indicators': {},
            'strategy': {}
        }
        
        suggested_values = {}
        
        # 1. Suggérer TOUS les paramètres d'indicateurs
        for indicator_name, indicator_config in config.get("indicators", {}).items():
            params['indicators'][indicator_name] = {}
            
            for param_name, param_config in indicator_config.get("params", {}).items():
                param_type = param_config["type"]
                full_param_name = f"{indicator_name}_{param_name}"
                
                if param_type == "int":
                    low = param_config["low"]
                    high = param_config["high"]
                    
                    # Ajuster selon contraintes SEULEMENT entre indicateurs
                    low, high = self._apply_constraint_to_range(
                        full_param_name, low, high, suggested_values
                    )
                    
                    if low > high:
                        value = param_config["low"]
                    else:
                        value = trial.suggest_int(
                            full_param_name,
                            int(low),
                            int(high),
                            step=param_config.get("step", 1)
                        )
                    
                    suggested_values[full_param_name] = value
                    
                elif param_type == "float":
                    low = param_config["low"]
                    high = param_config["high"]
                    
                    low, high = self._apply_constraint_to_range(
                        full_param_name, low, high, suggested_values
                    )
                    
                    if low > high:
                        value = param_config["low"]
                    else:
                        value = trial.suggest_float(
                            full_param_name,
                            low,
                            high,
                            log=param_config.get("log", False),
                            step=param_config.get("step", None)
                        )
                    
                    suggested_values[full_param_name] = value
                    
                elif param_type == "categorical":
                    value = trial.suggest_categorical(
                        full_param_name,
                        param_config["choices"]
                    )
                    suggested_values[full_param_name] = value
                
                params['indicators'][indicator_name][param_name] = value
        
        # 2. Suggérer TOUS les paramètres de stratégie (SANS contraintes pour l'instant)
        for param_name, param_config in config.get("strategy", {}).items():
            param_type = param_config["type"]
            
            if param_type == "int":
                value = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1)
                )
                
            elif param_type == "float":
                value = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                    step=param_config.get("step", None)
                )
                
            elif param_type == "categorical":
                value = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
            
            params['strategy'][param_name] = value
            suggested_values[param_name] = value
        
        return params
    
    def create_indicators_from_params(self, config: Dict[str, Any], params: Dict[str, Any]) -> List:
        """
        Crée automatiquement les indicateurs depuis les paramètres.
        
        Args:
            config: Configuration JSON complète
            params: Paramètres suggérés (avec structure 'indicators' et 'strategy')
        
        Returns:
            Liste des objets indicateurs instanciés
        """
        indicators = []
        
        for indicator_name, indicator_params in params['indicators'].items():
            indicator_config = config['indicators'][indicator_name]
            indicator_class_name = indicator_config['class']

            if indicator_class_name not in INDICATOR_CLASSES:
                raise ValueError(f"Indicateur '{indicator_class_name}' non trouvé dans INDICATOR_CLASSES")

            indicator_class = INDICATOR_CLASSES[indicator_class_name]

            # Instancier l'indicateur avec les paramètres
            try:
                # Essayer d'abord avec kwargs
                indicator = indicator_class(**indicator_params)
            except TypeError:
                # Si ça échoue, essayer avec args positionnels
                indicator = indicator_class(*indicator_params.values())
            
            indicators.append(indicator)
        
        return indicators
    
    def check_constraints(self, params: Dict[str, Any], constraints: list) -> bool:
        """
        Vérifie si les paramètres respectent les contraintes.
        
        Args:
            params: Paramètres suggérés
            constraints: Liste des contraintes
        
        Returns:
            True si toutes les contraintes sont respectées
        """
        # Aplatir la structure des params pour l'évaluation
        flat_params = {}
        
        # Ajouter les paramètres d'indicateurs
        for indicator_name, indicator_params in params['indicators'].items():
            for param_name, param_value in indicator_params.items():
                flat_params[f"{indicator_name}_{param_name}"] = param_value
        
        # Ajouter les paramètres de stratégie
        flat_params.update(params['strategy'])
        
        # Vérifier les contraintes
        for constraint in constraints:
            condition = constraint["condition"]
            try:
                if not eval(condition, {}, flat_params):
                    return False
            except Exception as e:
                print(f"Erreur lors de l'évaluation de la contrainte '{condition}': {e}")
                return False
        
        return True

    def _map_params_for_strategy(self, params: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
        """
        Mappe les paramètres d'optimisation vers les paramètres attendus par la fonction de stratégie.
        
        Args:
            params: Paramètres suggérés par Optuna (structure avec 'indicators' et 'strategy')
            strategy_name: Nom de la stratégie
        
        Returns:
            Dictionnaire avec les paramètres mappés pour la fonction de backtest
        """
        from financeta.ta_clients.optimization_strategy import STRATEGIES_IMPLEMENTED
        
        if strategy_name not in STRATEGIES_IMPLEMENTED:
            raise ValueError(f"Stratégie '{strategy_name}' non trouvée")
        
        strategy_info = STRATEGIES_IMPLEMENTED[strategy_name]
        expected_params = strategy_info['parameters']
        required_indicators = strategy_info['required_indicators']
        
        backtest_params = {}
        
        # Mapper les paramètres d'indicateurs
        for expected_param in expected_params:
            # Vérifier si c'est un paramètre de stratégie (stop_loss, take_profit, etc.)
            if expected_param in params['strategy']:
                backtest_params[expected_param] = params['strategy'][expected_param]
                continue
            
            # Sinon, chercher dans les indicateurs requis
            param_found = False
            for indicator_name in required_indicators:
                if indicator_name in params['indicators']:
                    indicator_params = params['indicators'][indicator_name]
                    
                    # Mapper selon des conventions de nommage
                    # Par exemple: "short_window" → chercher "period" dans "SMA_short"
                    if 'short' in expected_param.lower() and 'short' in indicator_name.lower():
                        if 'period' in indicator_params:
                            backtest_params[expected_param] = indicator_params['period']
                            param_found = True
                            break
                    elif 'long' in expected_param.lower() and 'long' in indicator_name.lower():
                        if 'period' in indicator_params:
                            backtest_params[expected_param] = indicator_params['period']
                            param_found = True
                            break
                    elif 'rsi' in expected_param.lower() and 'RSI' in indicator_name:
                        if 'period' in indicator_params:
                            backtest_params[expected_param] = indicator_params['period']
                            param_found = True
                            break
                    elif 'atr' in expected_param.lower() and 'ATR' in indicator_name:
                        if 'period' in indicator_params:
                            backtest_params[expected_param] = indicator_params['period']
                            param_found = True
                            break
            
            if not param_found:
                raise ValueError(f"Impossible de mapper le paramètre '{expected_param}' pour la stratégie '{strategy_name}'")
        
        return backtest_params

    def objective_function_create(self, 
                  trial: optuna.Trial,
                  backtest_func: Callable,
                  df_init: pl.DataFrame,
                  strategy_name: str) -> float:
        """
        Fonction objectif pour l'optimisation.
        Args:
            trial: Objet trial d'Optuna
            config: Configuration JSON complète
            backtest_func: Fonction de backtest
            df_init: DataFrame initial
        Returns:
            Sharpe Ratio (float)
        """
        
        # Suggérer les paramètres
        params = self.suggest_params_from_config(trial, self.optimization_config)
        
        # Vérifier les contraintes
        constraints = self.optimization_config.get("constraints", [])
        if not self.check_constraints(params, constraints):
            return float('-inf')  # Penaliser les trials qui ne respectent pas les contraintes
        # Créer les indicateurs
        indicators = self.create_indicators_from_params(self.optimization_config, params)
        
        # Appliquer les indicateurs
        df = df_init.clone()
        ta_client = TAClient()
        df = ta_client.calculate_indicators(df, indicators)

        # Récupérer le nom de la stratégie depuis la config
        if strategy_name is None or not isinstance(strategy_name, str):
            raise ValueError("Le champ 'strategy_name' doit être défini dans OPTIMIZATION_CONFIG et être une chaîne de caractères")
        
        # Mapper dynamiquement les paramètres
        backtest_params = self._map_params_for_strategy(params, strategy_name)
        
        # Exécuter le backtest
        try:
            # Utiliser contextlib.redirect_stdout pour supprimer les prints si verbose=False
            from contextlib import redirect_stdout
            from io import StringIO
            
            if not getattr(self, '_verbose', True):
                # Rediriger stdout vers un StringIO pour supprimer les prints
                with redirect_stdout(StringIO()):
                    metrics = backtest_func(df, **backtest_params)
            else:
                metrics = backtest_func(df, **backtest_params)
            
            sharpe_ratio = metrics.get("sharpe_ratio", float('-inf'))
            # Enregistrer les métriques dans les user_attrs du trial
            trial.set_user_attr("total_return", metrics.get("total_return", 0.0))
            trial.set_user_attr("cagr", metrics.get("cagr", 0.0))
            trial.set_user_attr("max_drawdown", metrics.get("max_drawdown", 0.0))
            
            if "trades_df" in metrics:
                trades_json = metrics["trades_df"].write_json()
                trial.set_user_attr("trades_json", trades_json)
                trial.set_user_attr("num_trades", len(metrics["trades_df"]))
                
            return sharpe_ratio if sharpe_ratio is not None else float('-inf')
        
        except Exception as e:
            if getattr(self, '_verbose', True):
                print(f"Erreur lors du backtest: {e}")
            return float('-inf')
    
    def optimize(self, 
                 backtest_func: Callable,
                 df: pl.DataFrame,
                 verbose: bool = True) -> optuna.Study:
        """
        Lance l'optimisation avec la configuration fournie.
        
        Args:
            config: Configuration de l'optimisation (dict ou chemin vers JSON)
            backtest_func: Fonction de backtest
            df: DataFrame initial
            verbose: Afficher les informations détaillées (config, résultats, barre de progression ET prints de stratégie)
        
        Returns:
            Study Optuna avec les résultats
        """
        opt_config = self.optimization_config["optimization"]
        indic_config = self.optimization_config["indicators"]
        strat_config = self.optimization_config["strategy"]
        
        # Afficher la configuration
        if verbose:
            print("="*70)
            print("CONFIGURATION DE L'OPTIMISATION")
            print("="*70)
            print(f"\nIndicateurs à optimiser:")
            for ind_name, ind_config in indic_config.items():
                print(f"  - {ind_name} ({ind_config['class']}): {list(ind_config['params'].keys())}")
            
            print(f"\nParamètres de stratégie:")
            for param_name in strat_config.keys():
                print(f"  - {param_name}")
            
            print(f"\nNombre de trials: {opt_config['n_trials']}")
            print(f"Direction: {opt_config['direction']}")
            print("="*70 + "\n")
        
        # Stocker verbose pour utilisation dans objective_function_create
        self._verbose = verbose
        
        # Créer la fonction objectif avec la config
        def objective(trial):
            strategy_name_val = self.optimization_config.get("strategy_name")
            if strategy_name_val is None or not isinstance(strategy_name_val, str):
                raise ValueError("Le champ 'strategy_name' doit être défini dans OPTIMIZATION_CONFIG et être une chaîne de caractères")
            return self.objective_function_create(
                trial=trial,
                backtest_func=backtest_func,
                df_init=df,
                strategy_name=strategy_name_val
            )
        
        # Contrôler les logs d'Optuna
        import logging
        optuna_logger = logging.getLogger("optuna")
        
        # Créer l'étude
        self.study = optuna.create_study(
            direction=opt_config["direction"],
            study_name=opt_config.get("study_name", "trading_strategy_optimization"),
            sampler=TPESampler(seed=opt_config.get("seed", None)),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )
        
        # Désactiver les logs d'Optuna si verbose=False
        if not verbose:
            optuna_logger.setLevel(logging.WARNING)  # Seulement les warnings et erreurs
        
        # Optimiser
        # Toujours afficher la barre de progression (tqdm), mais les prints de stratégie et logs sont contrôlés par verbose
        self.study.optimize(
            objective, 
            n_trials=opt_config["n_trials"],
            n_jobs=opt_config.get("n_jobs", 1),
            show_progress_bar=True,  # Toujours afficher le tqdm
            timeout=opt_config.get("timeout", None)
        )
        
        # Restaurer le niveau de log après optimisation
        if not verbose:
            optuna_logger.setLevel(logging.INFO)
        
        # Afficher les résultats
        if verbose:
            self._print_results()
        
        return self.study
    
    def _print_results(self):
        """Affiche les résultats de l'optimisation."""
        print("\n" + "="*70)
        print("RÉSULTATS DE L'OPTIMISATION")
        print("="*70)
        print(f"\nMeilleur Sharpe Ratio: {self.study.best_value:.4f}")
        
        print(f"\nMeilleurs paramètres:")
        print("\n  Indicateurs:")
        for key, value in self.study.best_params.items():
            if any(ind_name in key for ind_name in self.optimization_config['indicators'].keys()):
                print(f"    {key}: {value}")
        
        print("\n  Stratégie:")
        for key, value in self.study.best_params.items():
            if key in self.optimization_config['strategy'].keys():
                print(f"    {key}: {value}")
                
        print(f"\nMétriques de performance:")
        for key, value in self.study.best_trial.user_attrs.items():
            print(f"    {key}: {value:.4f}")
            
    def save_results(self, output_dir: str = "."):
        """
        Sauvegarde les résultats de l'optimisation.
        
        Args:
            output_dir: Répertoire de sortie
        """
        if self.study is None:
            print("Aucune optimisation n'a été effectuée")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Sauvegarder tous les trials
        trials_df = self.study.trials_dataframe()
        trials_df.to_csv(output_path / 'optimization_results.csv', index=False)
        
        # Sauvegarder la meilleure config
        best_config = self.get_best_config()
        self.save_config(best_config, output_path / 'best_config.json')
        
        print(f"\nRésultats sauvegardés dans '{output_dir}/'")
        print(f"  - optimization_results.csv")
        print(f"  - best_config.json")
    
    def get_best_config(self) -> Dict[str, Any]:
        """
        Récupère la meilleure configuration trouvée.
        
        Returns:
            Dictionnaire avec les meilleurs paramètres
        """
        if self.study is None:
            return {}
        
        best_config = {
            'indicators': {},
            'strategy': {}
        }
        
        # Obtenir uniquement les indicateurs requis par la stratégie (pas tous ceux configurés)
        strategy_name = self.optimization_config.get("strategy_name")
        from financeta.ta_clients.optimization_strategy import STRATEGIES_IMPLEMENTED
        if strategy_name and strategy_name in STRATEGIES_IMPLEMENTED:
            required_indicators = STRATEGIES_IMPLEMENTED[strategy_name]['required_indicators']
        else:
            required_indicators = list(self.optimization_config['indicators'].keys())
        
        # Parcourir tous les paramètres du meilleur trial
        for key, value in self.study.best_params.items():
            # Vérifier si c'est un paramètre d'indicateur
            matched = False
            for ind_name in required_indicators:
                if key.startswith(ind_name + "_") or key == ind_name:
                    if ind_name not in best_config['indicators']:
                        best_config['indicators'][ind_name] = {}
                    param_name = key.replace(f"{ind_name}_", "")
                    best_config['indicators'][ind_name][param_name] = value
                    matched = True
                    break
            
            if not matched:
                # C'est un paramètre de stratégie
                best_config['strategy'][key] = value
        
        return best_config
    
    def get_best_indicators(self) -> List:
        """
        Récupère les indicateurs avec les meilleurs paramètres.
        
        Returns:
            Liste des objets indicateurs instanciés
        """
        if self.study is None:
            return []
        
        best_config = self.get_best_config()
        params = {
            'indicators': best_config['indicators'],
            'strategy': best_config['strategy']
        }

        return self.create_indicators_from_params(self.optimization_config, params)

    def analyze_parameter_importance(self):
        """
        Analyse l'importance de chaque paramètre dans l'optimisation.
        
        Returns:
            Dict avec l'importance de chaque paramètre
        """
        if self.study is None:
            print("Aucune optimisation n'a été effectuée")
            return {}
        
        from optuna.importance import get_param_importances
        import warnings
        
        # Vérifier qu'il y a assez de trials valides
        valid_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(valid_trials) < 2:
            print("⚠️  Pas assez de trials valides pour calculer l'importance des paramètres")
            return {}
        
        # Compter les trials avec inf ou nan qui seront omitted par Optuna
        import math
        omitted_count = 0
        for t in self.study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                try:
                    value = t.value
                    if value is None or (isinstance(value, (int, float)) and 
                                        (math.isinf(value) or math.isnan(value))):
                        omitted_count += 1
                except (AttributeError, TypeError):
                    omitted_count += 1
        
        importances = {}
        
        # Supprimer TOUS les warnings et logs d'Optuna sur les trials omitted (approche agressive)
        import logging
        
        # Sauvegarder le niveau du logger d'Optuna et le mettre à ERROR pour supprimer les warnings
        optuna_logger = logging.getLogger("optuna")
        old_level = optuna_logger.level
        optuna_logger.setLevel(logging.ERROR)  # Seulement les erreurs
        
        # Capturer et supprimer TOUS les warnings Python
        with warnings.catch_warnings():
            # Filtrer tous les warnings concernant les trials omitted
            warnings.filterwarnings("ignore", message=".*omitted.*")
            warnings.filterwarnings("ignore", message=".*omitted in visualization.*")
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.simplefilter("ignore")
            
            try:
                # Essayer d'abord FANOVA (meilleure méthode)
                try:
                    importances = get_param_importances(
                        self.study, 
                        evaluator=optuna.importance.FanovaImportanceEvaluator()
                    )
                    print("\n" + "="*70)
                    print("IMPORTANCE DES PARAMÈTRES (FANOVA)")
                    print("="*70)
                    
                    # Afficher le compte des trials omitted une seule fois
                    if omitted_count > 0:
                        print(f"ℹ️  Nombre de trials omitted: {omitted_count}")
                    
                    for param, importance in sorted(importances.items(), 
                                                key=lambda x: x[1], reverse=True):
                        status = "⚠️  INUTILE" if importance < 0.01 else "✓ Utile"
                        print(f"{param:30s}: {importance:.4f} {status}")
                    return importances
                except Exception as e:
                    print(f"ℹ️  FANOVA non disponible ({type(e).__name__}), utilisation de la méthode par défaut")
                
                # Essayer la méthode basée sur les arbres (plus robuste)
                try:
                    importances = get_param_importances(
                        self.study,
                        evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
                    )
                    print("\n" + "="*70)
                    print("IMPORTANCE DES PARAMÈTRES (Mean Decrease Impurity)")
                    print("="*70)
                    
                    # Afficher le compte des trials omitted une seule fois
                    if omitted_count > 0:
                        print(f"ℹ️  Nombre de trials omitted: {omitted_count}")
                    
                    for param, importance in sorted(importances.items(), 
                                                key=lambda x: x[1], reverse=True):
                        status = "⚠️  INUTILE" if importance < 0.01 else "✓ Utile"
                        print(f"{param:30s}: {importance:.4f} {status}")
                    return importances
                except Exception as e:
                    print(f"ℹ️  Mean Decrease Impurity non disponible ({type(e).__name__})")
                
                # Fallback : méthode par défaut (toujours disponible)
                try:
                    importances = get_param_importances(self.study)
                    print("\n" + "="*70)
                    print("IMPORTANCE DES PARAMÈTRES")
                    print("="*70)
                    
                    # Afficher le compte des trials omitted une seule fois
                    if omitted_count > 0:
                        print(f"ℹ️  Nombre de trials omitted: {omitted_count}")
                    
                    for param, importance in sorted(importances.items(), 
                                                key=lambda x: x[1], reverse=True):
                        status = "⚠️  INUTILE" if importance < 0.01 else "✓ Utile"
                        print(f"{param:30s}: {importance:.4f} {status}")
                    return importances
                except Exception as e:
                    print(f"❌ Impossible de calculer l'importance des paramètres: {e}")
                    return {}
            finally:
                # Restaurer le niveau du logger d'Optuna
                optuna_logger.setLevel(old_level)
    
    def get_useless_parameters(self, threshold: float = 0.01):
        """
        Identifie les paramètres qui ont peu d'impact.
        
        Args:
            threshold: Seuil en dessous duquel un paramètre est considéré inutile
        
        Returns:
            Liste des paramètres inutiles
        """
        from optuna.importance import get_param_importances
        
        if self.study is None:
            return []
        
        try:
            importances = get_param_importances(
                self.study,
                evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
            )
        except Exception as e:
            print(f"❌ Impossible de calculer l'importance: {e}")
            return []
        
        # Séparer les paramètres par type
        indicator_params = {}
        strategy_params = {}
        
        for param, imp in importances.items():
            # Vérifier si c'est un paramètre de stratégie
            if param in self.optimization_config.get('strategy', {}).keys():
                strategy_params[param] = imp
            else:
                # Sinon c'est un paramètre d'indicateur
                indicator_params[param] = imp
        
        # Identifier les inutiles
        useless_indicators = {p: imp for p, imp in indicator_params.items() if imp < threshold}
        useless_strategy = {p: imp for p, imp in strategy_params.items() if imp < threshold}
        
        print("\n" + "="*70)
        print("ANALYSE DES PARAMÈTRES À FAIBLE IMPACT")
        print("="*70)
        
        if useless_indicators:
            print(f"\n⚠️  Indicateurs techniques avec faible impact (< {threshold}):")
            for param, imp in sorted(useless_indicators.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {param:30s} (importance: {imp:.4f})")
        else:
            print(f"\n✓ Tous les indicateurs techniques ont un impact significatif")
        
        if useless_strategy:
            print(f"\n⚠️  Paramètres de stratégie avec faible impact (< {threshold}):")
            for param, imp in sorted(useless_strategy.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {param:30s} (importance: {imp:.4f})")
                # Ajouter une explication
                if param in ['stop_loss', 'take_profit']:
                    print(f"      → Cela peut indiquer que votre stratégie ne les utilise pas efficacement")
                elif param == 'position_size':
                    print(f"      → Le Sharpe Ratio est indépendant de la taille de position")
        else:
            print(f"\n✓ Tous les paramètres de stratégie ont un impact significatif")
        
        return list(useless_indicators.keys()) + list(useless_strategy.keys())
    
    def plot_importance(self):
        """Visualise l'importance des paramètres."""
        if self.study is None:
            print("Aucune optimisation n'a été effectuée")
            return
        
        import optuna.visualization as vis
        import warnings
        import math
        import logging
        
        # Compter les trials omitted
        omitted_count = sum(1 for t in self.study.trials 
                          if t.state == optuna.trial.TrialState.COMPLETE
                          and (not hasattr(t, 'value') or t.value is None 
                               or (isinstance(t.value, (int, float)) and 
                                  (math.isinf(t.value) or math.isnan(t.value)))))
        
        # Sauvegarder et modifier le niveau du logger d'Optuna pour supprimer les warnings
        optuna_logger = logging.getLogger("optuna")
        old_level = optuna_logger.level
        optuna_logger.setLevel(logging.ERROR)  # Seulement les erreurs
        
        try:
            # Supprimer les warnings d'Optuna sur les trials omitted
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*omitted.*")
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.simplefilter("ignore")
                
                # Créer le graphique d'importance
                fig = vis.plot_param_importances(self.study)
                fig.show()
        finally:
            # Restaurer le niveau du logger d'Optuna
            optuna_logger.setLevel(old_level)
        
        # Afficher le compte une seule fois si nécessaire
        if omitted_count > 0:
            print(f"\nℹ️  Nombre de trials omitted: {omitted_count}")

    def get_trades_history(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Récupère l'historique détaillé des trades avec les meilleurs paramètres.
        
        Args:
            df: DataFrame avec les données de prix
        
        Returns:
            DataFrame avec l'historique des trades
        """
        if self.study is None:
            print("⚠️ Aucune optimisation n'a été effectuée")
            return pl.DataFrame()
        
        from financeta.ta_clients.optimization_strategy import OptimizationStrategyClient
        
        # Récupérer les meilleurs paramètres
        best_config = self.get_best_config()
        
        # Vérifier que la config n'est pas vide
        if not best_config.get('indicators') and not best_config.get('strategy'):
            print("⚠️ Erreur: Impossible de récupérer la meilleure configuration")
            return pl.DataFrame()
        
        params = {
            'indicators': best_config['indicators'],
            'strategy': best_config['strategy']
        }
        
        try:
            # Créer les indicateurs
            indicators = self.create_indicators_from_params(self.optimization_config, params)
            
            if not indicators:
                print("⚠️ Erreur: Aucun indicateur créé")
                return pl.DataFrame()
            
            # Appliquer les indicateurs
            df_with_indicators = df.clone()
            from financeta.utils.ta import TAClient
            ta_client = TAClient()
            df_with_indicators = ta_client.calculate_indicators(df_with_indicators, indicators)
            
            # Vérifier que le DataFrame n'est pas vide après calcul des indicateurs
            if len(df_with_indicators) == 0:
                print("⚠️ Erreur: DataFrame vide après calcul des indicateurs")
                return pl.DataFrame()
            
            # Récupérer la fonction de stratégie
            strategy_name = self.optimization_config.get("strategy_name")
            opt_strategy = OptimizationStrategyClient()
            backtest_func = opt_strategy.get_strategy_fct(strategy_name)
            
            # Mapper les paramètres
            backtest_params = self._map_params_for_strategy(params, strategy_name)
            
            # Exécuter avec return_trades=True
            results = backtest_func(df_with_indicators, **backtest_params, return_trades=True)
            
            # Vérifier que results contient trades_df
            if not isinstance(results, dict):
                print(f"⚠️ Erreur: La fonction de stratégie n'a pas retourné un dictionnaire, type: {type(results)}")
                return pl.DataFrame()
            
            trades_df = results.get("trades_df")
            if trades_df is None:
                print("⚠️ Erreur: La clé 'trades_df' est absente du résultat")
                print(f"   Clés disponibles: {list(results.keys())}")
                return pl.DataFrame()
            
            if isinstance(trades_df, pl.DataFrame) and len(trades_df) == 0:
                print("⚠️ Aucun trade généré avec les meilleurs paramètres")
                return trades_df
            
            return trades_df
            
        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération des trades: {e}")
            import traceback
            traceback.print_exc()
            return pl.DataFrame()

    def print_trades_summary(self, trades_df: pl.DataFrame):
        """
        Affiche un résumé des trades.
        
        Args:
            trades_df: DataFrame des trades
        """
        if len(trades_df) == 0:
            print("Aucun trade à afficher")
            return
        
        print("\n" + "="*70)
        print("RÉSUMÉ DES TRADES")
        print("="*70)
        
        # Stats générales
        total_trades = len(trades_df)
        buy_trades = (trades_df['action'] == 'BUY').sum()
        sell_trades = (trades_df['action'] == 'SELL').sum()
        
        print(f"\n📊 Nombre total de trades: {total_trades}")
        print(f"  - Achats: {buy_trades}")
        print(f"  - Ventes: {sell_trades}")
        
        # Stats de performance
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] < 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\n💰 Performance:")
        print(f"  - Trades gagnants: {winning_trades} ({win_rate:.2f}%)")
        print(f"  - Trades perdants: {losing_trades} ({100-win_rate:.2f}%)")
        
        avg_win = trades_df.filter(pl.col('pnl') > 0)['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df.filter(pl.col('pnl') < 0)['pnl'].mean() if losing_trades > 0 else 0
        
        print(f"  - Gain moyen: {avg_win:.4f}")
        print(f"  - Perte moyenne: {avg_loss:.4f}")
        
        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
            print(f"  - Profit Factor: {profit_factor:.2f}")
        
        # Top 5 meilleurs et pires trades
        print(f"\n🏆 Top 5 meilleurs trades:")
        best_trades = trades_df.sort('pnl', descending=True).head(5)
        for row in best_trades.iter_rows(named=True):
            print(f"  {row['timestamp']}: {row['action']:4s} @ {row['price']:.2f}$ → PnL: {row['pnl']:.4f}")
        
        print(f"\n📉 Top 5 pires trades:")
        worst_trades = trades_df.sort('pnl').head(5)
        for row in worst_trades.iter_rows(named=True):
            print(f"  {row['timestamp']}: {row['action']:4s} @ {row['price']:.2f}$ → PnL: {row['pnl']:.4f}")

    def export_trades_to_csv(self, trades_df: pl.DataFrame, filename: str = "trades_history.csv"):
        """
        Export les trades vers un fichier CSV.
        
        Args:
            trades_df: DataFrame des trades
            filename: Nom du fichier de sortie
        """
        if len(trades_df) == 0:
            print("Aucun trade à exporter")
            return
        
        trades_df.write_csv(filename)
        print(f"✅ Trades exportés vers {filename}")


# Exemple d'utilisation
if __name__ == "__main__":
    from financeta.clients.yfinance import YahooFinanceClient
    import polars as pl
    import numpy as np
    
    # Définir la fonction de backtest
    
    
    # Récupérer les données
    yh = YahooFinanceClient()
    df_init = yh.get_price("AAPL", from_date="2025-09-26", to_date="2025-10-04", 
                           interval="1m", postclean=True)
    
    # Charger la configuration
    from financeta.ta_clients.optimization_strategy import OptimizationStrategyClient
    opt_strategy = OptimizationStrategyClient()
    opt_strategy.print_strategies()
    
    from financeta.ta_clients.optimization import OptimizationClient
    INDICATOR_CLASSES  = opt_strategy.get_seed_indicator_config()
    opt_client = OptimizationClient(INDICATOR_CLASSES)
    
    
    strat_name = "simple_strategy"
    opt_config = opt_strategy.get_seed_optimization_config()
    backtest_strategy = opt_strategy.get_strategy_fct(strat_name)
    
    
    # Lancer l'optimisation
    ta_client = TAClient()
    study = opt_client.optimize(opt_config, backtest_strategy, df_init, ta_client)
    
    # Sauvegarder les résultats
    opt_client.save_results("results")
    
    # Récupérer les meilleurs indicateurs
    best_indicators = opt_client.get_best_indicators()
    print(f"\nMeilleurs indicateurs: {best_indicators}")