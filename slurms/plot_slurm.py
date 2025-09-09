import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Specify the directory path
directory_path = "versions/"
#directory_path = "pooling"
# Initialize dictionaries to store file contents
file_contents = {}  # Stores the entire content of each file as a string
file_lines = {}     # Stores the content of each file as a list of lines
file_names = []
# Iterate through all files in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    # Check if it's a file (not a directory)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            content = file.read()  # Read the entire file as a string
            lines = content.splitlines()  # Split the content into lines
            
            # Store the content in the dictionaries
            file_contents[filename] = content
            file_lines[filename] = lines
            file_names.append(filename)


verlauf = pd.DataFrame(columns=["Filename","epoch", "train_loss", "train", "dev","Pa_acc","dev_par_acc"])
for name in file_names:
    save = False
    if ".err" in name:
        continue
    for line in file_lines[name]:
        if "Epoch " in line:
            pattern = r"[-+]?\d*\.\d+|\d+"
            numbers = re.findall(pattern, line)
            numbers = [float(num) for num in numbers]
            if len(numbers) < 2:
                break
            verlauf = pd.concat([verlauf, pd.DataFrame({"Filename":[name],"epoch": [numbers[0]], "train_loss": [numbers[1]], "train": [numbers[2]], "dev": [numbers[3]],"Pa_acc": [0],"dev_par_acc": [0]})], ignore_index=True)
            if numbers[0]==6:
                save = True
        elif save == True and ("Paraphrase detection accuracy:" in line):
            numbers = re.findall(pattern, line)  # Alle Zahlen finden
            numbers = [float(num) for num in numbers] 
            verlauf.loc[verlauf["Filename"] == name, "Pa_acc"] = numbers[0]

        elif save == True and ("dev paraphrase acc" in line):
            numbers = re.findall(pattern, line)
            numbers = [float(num) for num in numbers] 
            verlauf.loc[verlauf["Filename"] == name, "dev_par_acc"] = numbers[0]


from matplotlib.lines import Line2D
plt.style.use('seaborn-v0_8-paper')
# Definiere Marker für die verschiedenen Typen
markers = {
    "train_loss": "o",  # Kreis
    "train": "s",       # Quadrat
    "dev": "D"          # Raute
}

# Generiere eine Farbpalette für die verschiedenen Filenames
unique_names = verlauf["Filename"].unique()
colors = sns.color_palette("husl", len(unique_names))  # Farbpalette mit eindeutigen Farben

# Erstelle den Plot
plt.figure(figsize=(12, 8))  # Größeres Diagramm für bessere Lesbarkeit

verlauf = verlauf.drop(columns=["train_loss", "train"])

max_bl = 0
for i, name in enumerate(unique_names):
    # Filtere den DataFrame für den aktuellen "Filename"
    temp = verlauf[verlauf["Filename"] == name]
    # Plotte Linien und Scatter-Punkte für jeden Typ
    for col, marker in markers.items():
        if "baseline" in name:
            max_bl = max(temp["dev"].max(), max_bl)
        if "train_loss" in col or "train" in col:
            continue
        plt.plot(temp["epoch"], temp[col], label=f"{name.split('-')[0]} - {col.replace('_', ' ').title()}", color=colors[i], alpha=0.6)
        plt.scatter(temp["epoch"], temp[col], color=colors[i], marker=marker, edgecolor="black", s=50, alpha=0.8)

# Benutzerdefinierte Legende erstellen
legend_elements = []
for i, name in enumerate(unique_names):
    for col, marker in markers.items():
        if "train_loss" in col or "train" in col:
            continue    
        legend_elements.append(Line2D([0], [0], color=colors[i], marker=marker, label=f"{name.split('-')[0]} - {col.replace('_', ' ').title()}", 
                                       markersize=10, linestyle='-', alpha=0.8))
        
# BASELINE
plt.hlines(y=0.870, xmin=1, xmax=6, color='r', linestyle='--', label='BASELINE', alpha=0.3)  # Horizontale Linie bei y=0.870
plt.hlines(y=max_bl, xmin=1, xmax=6, color='r', linestyle='--', label='NEW BASELINE', alpha=0.3)  # Horizontale Linie bei y=0.870
plt.fill_between(x=[1, 6], y1=0.870, y2=max_bl, color='red', alpha=0.1, label='Wide Baseline')



# Diagramm-Details hinzufügen
plt.title("Validation Dev over Epochs", fontsize=18)  # Titel auf Englisch
plt.xlabel("Epochs", fontsize=16)  # X-Achse auf Englisch
plt.ylabel("Values", fontsize=16)  # Y-Achse auf Englisch

# Schriftgröße der Achsen-Ticks anpassen
plt.xticks(fontsize=14)  # Schriftgröße der X-Achsen-Ticks
plt.yticks(fontsize=14)  # Schriftgröße der Y-Achsen-Ticks

plt.legend(handles=legend_elements, fontsize=12, loc="best")  # Benutzerdefinierte Legende
plt.grid(True, linestyle="--", alpha=0.7)  # Gitterlinien
plt.tight_layout()  # Optimiert die Abstände im Plot

# Bild speichern
output_dir = "Bilder"
os.makedirs(output_dir, exist_ok=True)  # Ordner erstellen, falls er nicht existiert
if directory_path == "pooling":
    plt.savefig(os.path.join(output_dir, "qqp_pooling_plot.png"), dpi=300)  # Speichere das Bild mit hoher Auflösung
elif directory_path == "versions/":
    plt.savefig(os.path.join(output_dir, "qqp_changes.png"), dpi=300)
# Zeige den Plot an
plt.show()