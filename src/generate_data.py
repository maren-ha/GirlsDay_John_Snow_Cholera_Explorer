import numpy as np
import pandas as pd

# set seed for reproducibility
np.random.seed(42)

# Number of individuals in the dataset
number_of_individuals = 500

# Generate ages (focusing on younger and middle-aged individuals)
ages = np.random.choice(range(1, 80), size=number_of_individuals, p=np.linspace(1, 2, 79)[::-1] / np.sum(np.linspace(1, 2, 79)))

# Assign gender randomly
genders = np.random.choice(["Male", "Female"], size=number_of_individuals)

# Household size (larger households more common in poorer areas)
household_sizes = np.random.choice(range(1, 8), size=number_of_individuals, p=[0.05, 0.1, 0.15, 0.25, 0.2, 0.15, 0.1])

# Assign occupations randomly
occupations = np.random.choice(["Laborer", "Merchant", "Housewife", "Student", "Clerk"], size=number_of_individuals)

# # Define pump locations (simplified, using arbitrary coordinates)
pumps = {
    "Pump A": (2, 3),
    "Pump B": (5, 7),
    "Pump C": (8, 2),
    "Pump D": (3, 6)
}

# Generate random home locations for individuals within a 10x10 grid
home_x = np.random.uniform(0, 10, number_of_individuals)
home_y = np.random.uniform(0, 10, number_of_individuals)

# Calculate distances to each pump
distances = {pump: np.sqrt((home_x - coord[0])**2 + (home_y - coord[1])**2) for pump, coord in pumps.items()}

# Determine the closest pump
nearest_pump = np.array([min(pumps.keys(), key=lambda p: distances[p][i]) for i in range(number_of_individuals)])

# Generate a fake but somewhat realistic risk factor (e.g., raw vegetable consumption)
raw_veggies = np.random.choice(["Often", "Sometimes", "Rarely"], size=number_of_individuals, p=[0.3, 0.4, 0.3])

# Introduce some randomness in disease severity but link it primarily to the closest pump

# Gesundheitsstatus bestimmen
health_status = []
for i in range(number_of_individuals):
    base_risk = np.exp(-min(distances[p][i] for p in pumps)) * 2  # Höheres Basisrisiko

    if nearest_pump[i] == "Pump B":  # Assume Pump B is contaminated
        base_risk *= 4  # Increase risk significantly

    # Interaction effect: Age & Household Size jointly influence risk
    age_factor = 1.5 if ages[i] < 10 or ages[i] > 60 else 1.0
    household_factor = 1.5 if household_sizes[i] > 4 else 1.0
    interaction_effect = age_factor * household_factor * 1.3  # Combined effect

    risk = base_risk * interaction_effect * np.random.uniform(0.8, 1.2)

    if risk > 3.0:
        health_status.append("Death")
    elif risk > 1.8:
        health_status.append("Severe Illness")
    elif risk > 0.6:
        health_status.append("Mild Illness")
    else:
        health_status.append("No Illness")

# create DataFrame
data = pd.DataFrame({
    "ID": range(1, number_of_individuals + 1),
    "Age": ages,
    "Gender": genders,
    "Household Size": household_sizes,
    "Occupation": occupations,
    "Home Location X": home_x,
    "Home Location Y": home_y,
    "Nearest Pump": nearest_pump,
    "Raw Vegetable Consumption": raw_veggies,
    **{f"Distance to {pump}": distances[pump] for pump in pumps},
    "Health Status": health_status
})

# More realistic occupation assignment
def assign_occupation(age, gender):
    """Assign a realistic occupation based on age and gender."""
    if age < 14:
        return "Student"
    elif age > 65:
        return "Retiree"
    else:
        if gender == "Female":
            return np.random.choice(["Housewife", "Merchant", "Servant"])
        else:
            return np.random.choice(["Laborer", "Clerk", "Merchant"])

data["Occupation"] = [assign_occupation(a, g) for a, g in zip(data["Age"], data["Gender"])]

# Datensatz speichern
data.to_csv('../data/cholera_dataset.csv', index=False)
