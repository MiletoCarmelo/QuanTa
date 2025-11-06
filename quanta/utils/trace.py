"""
Traces de visualisation - éléments graphiques de base
"""

class Trace:
    """Classe de base pour les traces de visualisation."""
    pass


class Candlesticks(Trace):
    """Représente les chandeliers japonais."""
    
    def __init__(self):
        pass


class Volume(Trace):
    """Représente le volume."""
    
    def __init__(self):
        pass


class Line(Trace):
    """Ligne générique pour tracer n'importe quelle colonne."""
    
    def __init__(self, column: str, name: str = None, color: str = None, width: float = 1.5):
        self.column = column
        self.name = name or column
        self.color = color
        self.width = width


# Exemple d'utilisation
if __name__ == "__main__":
    # Les traces définissent COMMENT afficher les données
    candlesticks = Candlesticks()
    volume = Volume()
    custom_line = Line('my_custom_indicator', name='Mon Indicateur', color='purple')