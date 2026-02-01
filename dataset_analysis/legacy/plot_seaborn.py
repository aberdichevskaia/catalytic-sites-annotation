import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

sns.set(style="whitegrid")  # Set Seaborn style

# Load JSON dataset for EC number and protein information
with open('/home/iscb/wolfson/annab4/DB/protein_table_modified.json', 'r') as f:
    protein_data = json.load(f)

# Load CSV dataset for cluster and sequence info
dataset_dir = '/home/iscb/wolfson/annab4/DB/splitted_by_EC_number_fixed'
dataset = pd.read_csv(os.path.join(dataset_dir, 'final_dataset_with_component_ids.csv'))

# Load labels from test/train/validate files
def load_labels(file_path):
    labels = []
    sequences = []
    with open(file_path, 'r') as file:
        seq_id = None
        for line in file:
            if line.startswith('>'):
                seq_id = line.strip()[1:]
            else:
                parts = line.strip().split()
                sequences.append(seq_id)
                labels.append(int(parts[3]))
    return sequences, labels

def merge_labels(files):
    sequences, labels = [], []
    for file in files:
        seqs, lbls = load_labels(file)
        sequences.extend(seqs)
        labels.extend(lbls)
    return pd.DataFrame({'Sequence_ID': sequences, 'Label': labels})

label_data = merge_labels([
    os.path.join(dataset_dir, 'test.txt'), 
    os.path.join(dataset_dir, 'train.txt'), 
    os.path.join(dataset_dir, 'validate.txt')
])

# 1. Dataset statistics
# Unique proteins
unique_proteins = len(protein_data)

# Cluster stats
clusters_X = dataset['Cluster_1'].nunique()
clusters_Y = dataset['Cluster_2'].nunique()

# Fraction of positive/negative labels
positive_fraction = label_data['Label'].mean()
negative_fraction = 1 - positive_fraction

# Display statistics
print(f"Unique proteins: {unique_proteins}")
print(f"Clusters level 1: {clusters_X}")
print(f"Clusters level 2: {clusters_Y}")
print(f"Positive labels fraction: {positive_fraction:.5f}")
print(f"Negative labels fraction: {negative_fraction:.5f}")

# 2. Plots
# Create directory for plots
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

# Histogram and CDF: Catalytic sites per unique protein
protein_labels = label_data.copy()
protein_labels['Protein'] = protein_labels['Sequence_ID'].str.split('_').str[0]
unique_protein_labels = protein_labels[protein_labels['Protein'].isin(protein_data.keys())]
catalytic_sites_per_protein = unique_protein_labels.groupby('Protein')['Label'].sum()

# Histogram
plt.figure(figsize=(8, 6))
sns.histplot(catalytic_sites_per_protein, bins=np.arange(0, 11), color='lightgreen', kde=False)
plt.title('Catalytic Sites Per Unique Protein')
plt.xlabel('Number of Catalytic Sites')
plt.ylabel('Frequency')
plt.xticks(range(11))  # Show all numbers up to 10
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'catalytic_sites_per_unique_protein_seaborn.png'))
plt.close()

# CDF
sorted_data = np.sort(catalytic_sites_per_protein)
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
plt.figure(figsize=(8, 6))
sns.lineplot(x=sorted_data, y=cdf, marker='o', linestyle='-', color='blue')
plt.title('Cumulative Distribution of Catalytic Sites Per Unique Protein')
plt.xlabel('Number of Catalytic Sites')
plt.ylabel('Cumulative Probability')
plt.xticks(range(11))  # Show all numbers up to 10
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'catalytic_sites_cdf_per_unique_protein_seaborn.png'))
plt.close()

# Pie charts: EC number distribution
def get_ec_level(ec_number, level):
    if ec_number == "not found":
        return None
    parts = ec_number.split('.')
    if len(parts) < level:
        return None
    if all(part.isdigit() for part in parts[:level]):
        return '.'.join(parts[:level])

ec_level_1 = [get_ec_level(protein_data[protein]['EC_number'], 1) for protein in protein_data]
ec_level_1 = [ec for ec in ec_level_1 if ec is not None]  # Remove None and 'Unknown' values

ec_descriptions = {
    '1': 'EC 1: Oxidoreductases',
    '2': 'EC 2: Transferases',
    '3': 'EC 3: Hydrolases',
    '4': 'EC 4: Lyases',
    '5': 'EC 5: Isomerases',
    '6': 'EC 6: Ligases',
    '7': 'EC 7: Translocases'
}

ec_counts = pd.Series(ec_level_1).value_counts()

plt.figure(figsize=(10, 10))
wedges, texts, autotexts = plt.pie(ec_counts, autopct=lambda p: f'{p:.1f}%', startangle=140, pctdistance=1.2, colors=sns.color_palette("muted", len(ec_counts)), wedgeprops={'linewidth': 0})
for autotext in autotexts:
    #autotext.set_color('white')  # Set percentage text to white
    autotext.set_horizontalalignment('center')
    autotext.set_verticalalignment('center')
    #autotext.set_rotation(-20)
    autotext.set_fontweight("bold")
    
plt.legend([f'{ec_descriptions[label]}' for label in sorted(ec_counts.index)], loc="upper left", bbox_to_anchor=(0.0, 0.0))
plt.title('Enzymes classes distribution', fontsize=30)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'ec_level_1_distribution_seaborn.png'))
plt.close()

# Bar plot: Amino acid distribution
amino_acids = []
amino_acid_labels = []

for file in [
    os.path.join(dataset_dir, 'test.txt'), 
    os.path.join(dataset_dir, 'train.txt'), 
    os.path.join(dataset_dir, 'validate.txt')
]:
    with open(file, 'r') as f:
        seq_id = None
        for line in f:
            if line.startswith('>'):
                seq_id = line.strip()
            else:
                parts = line.strip().split()
                amino_acids.append(parts[2])  # Amino acid column
                amino_acid_labels.append(int(parts[3]))  # Label column

amino_acid_df = pd.DataFrame({'AminoAcid': amino_acids, 'Label': amino_acid_labels})
amino_acid_df = amino_acid_df[amino_acid_df['AminoAcid'] != 'X']  # Remove only ambiguous 'X', keep 'Y' (tyrosine)

# Separate distributions for catalytic and non-catalytic sites
catalytic_distribution = amino_acid_df[amino_acid_df['Label'] == 1]['AminoAcid'].value_counts(normalize=True)
total_distribution = amino_acid_df['AminoAcid'].value_counts(normalize=True)

# Combine into a single DataFrame for plotting
overall_distribution = pd.DataFrame({
    'Total': total_distribution,
    'Catalytic': catalytic_distribution
}).fillna(0).reset_index().melt(id_vars='AminoAcid', var_name='Site Type', value_name='Proportion')

plt.figure(figsize=(14, 8))
sns.barplot(x='AminoAcid', y='Proportion', hue='Site Type', data=overall_distribution, dodge=True, palette=['lightblue', 'orange'])
plt.title('Amino Acid Distribution (Proportions Across Sites)')
plt.xlabel('Amino Acid')
plt.ylabel('Proportion')
plt.xticks(rotation=90)
plt.legend(title='Site Type', loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'amino_acid_distribution_by_site_type_seaborn.png'))
plt.close()
