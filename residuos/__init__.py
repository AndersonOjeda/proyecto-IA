"""Proyecto 4: detección de residuos y clasificación de basura."""
CLASSES = ('plastico', 'papel', 'carton', 'vidrio', 'metal', 'organico')
LABELS = ('Plástico', 'Papel', 'Cartón', 'Vidrio', 'Metal', 'Restos orgánicos')

def group(material):
    return 'Orgánico' if material == 'organico' else 'Inorgánico (gestión)'
