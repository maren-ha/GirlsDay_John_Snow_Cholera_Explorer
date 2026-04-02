import numpy as np
import pandas as pd

from src.data_schema import DATA_DIR

# Zufallszahlengenerator setzen für Reproduzierbarkeit
np.random.seed(42)

# Anzahl der Personen im Datensatz
anzahl_personen = 500

# Alter generieren (schwerpunktmäßig jüngere und mittelalte Personen)
alter = np.random.choice(range(1, 80), size=anzahl_personen, p=np.linspace(1, 2, 79)[::-1] / np.sum(np.linspace(1, 2, 79)))

# Geschlecht zufällig zuweisen
geschlecht = np.random.choice(["Männlich", "Weiblich"], size=anzahl_personen)

# Haushaltsgröße (größere Haushalte häufiger in ärmeren Gegenden)
haushaltsgroeße = np.random.choice(range(1, 8), size=anzahl_personen, p=[0.05, 0.1, 0.15, 0.25, 0.2, 0.15, 0.1])

# Berufe zufällig zuweisen
berufe = np.random.choice(["Arbeiter", "Händler", "Hausfrau", "Schüler", "Schreiber"], size=anzahl_personen)

# Positionen der Pumpen
pumpen = {
    "Pumpe A": (2, 3),
    "Pumpe B": (5, 7),
    "Pumpe C": (8, 2),
    "Pumpe D": (3, 6)
}

# Wohnorte der Personen innerhalb eines 10x10-Rasters
wohn_x = np.random.uniform(0, 10, anzahl_personen)
wohn_y = np.random.uniform(0, 10, anzahl_personen)

# Entfernungen zu jeder Pumpe berechnen
entfernungen = {pumpe: np.sqrt((wohn_x - koord[0])**2 + (wohn_y - koord[1])**2) for pumpe, koord in pumpen.items()}

# Nächstgelegene Pumpe bestimmen
naechste_pumpe = np.array([min(pumpen.keys(), key=lambda p: entfernungen[p][i]) for i in range(anzahl_personen)])

# Rohkost-Konsum generieren
rohkost = np.random.choice(["Oft", "Manchmal", "Selten"], size=anzahl_personen, p=[0.3, 0.4, 0.3])

# Gesundheitsstatus bestimmen
gesundheitsstatus = []
for i in range(anzahl_personen):
    grundrisiko = np.exp(-min(entfernungen[p][i] for p in pumpen)) * 2  # Höheres Basisrisiko
    
    if naechste_pumpe[i] == "Pumpe B":  # Pumpe B ist kontaminiert
        grundrisiko *= 4  # Stärkerer Effekt
    
    # Wechselwirkungseffekt: Alter & Haushaltsgröße gemeinsam beeinflussen Risiko
    altersfaktor = 1.5 if alter[i] < 10 or alter[i] > 60 else 1.0
    haushaltsfaktor = 1.5 if haushaltsgroeße[i] > 4 else 1.0
    interaktionseffekt = altersfaktor * haushaltsfaktor * 1.3  # Kombinierte Wirkung
    
    risiko = grundrisiko * interaktionseffekt * np.random.uniform(0.8, 1.2)
    
    if risiko > 3.0:
        gesundheitsstatus.append("Tod")
    elif risiko > 1.8:
        gesundheitsstatus.append("Schwere Krankheit")
    elif risiko > 0.6:
        gesundheitsstatus.append("Leichte Krankheit")
    else:
        gesundheitsstatus.append("Keine Krankheit")

# DataFrame erstellen
daten = pd.DataFrame({
    "ID": range(1, anzahl_personen + 1),
    "Alter": alter,
    "Geschlecht": geschlecht,
    "Haushaltsgröße": haushaltsgroeße,
    "Beruf": berufe,
    "Wohnort X": wohn_x,
    "Wohnort Y": wohn_y,
    "Nächstgelegene Pumpe": naechste_pumpe,
    "Rohkost-Konsum": rohkost,
    **{f"Entfernung zu {pumpe}": entfernungen[pumpe] for pumpe in pumpen},
    "Gesundheitsstatus": gesundheitsstatus
})

# Realistischere Berufszuweisung
def beruf_zuweisen(alter, geschlecht):
    """Weist einen realistischen Beruf basierend auf Alter und Geschlecht zu."""
    if alter < 14:
        return "Schüler"
    elif alter > 65:
        return "Rentner"
    else:
        if geschlecht == "Weiblich":
            return np.random.choice(["Hausfrau", "Händler", "Diener"])
        else:
            return np.random.choice(["Arbeiter", "Schreiber", "Händler"])

daten["Beruf"] = [beruf_zuweisen(a, g) for a, g in zip(daten["Alter"], daten["Geschlecht"])]

# Datensatz speichern
daten.to_csv(DATA_DIR / 'cholera_datensatz_de.csv', index=False)
