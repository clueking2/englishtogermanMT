import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


############################### Table #######################################
#create a dataframe
data = {
    'Model': ['Unigram', 'Bigram', 'Custom Unigram', 'Custom Bigram'],
    'BLEU Score': [0.4044, 1.4795, 3.1071, 2.6473],
    'Accuracy': [0.1121, 0.2176, 0.2801, 0.2533],
    'F1 Score': [0.1071, 0.2460, 0.2056, 0.3116]
}
df = pd.DataFrame(data)

#plot table with matplotlib
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.scale(1, 2)  # Scale rows/columns
plt.title("Model Scores Table")
plt.tight_layout()
plt.show()



############################### Line Chart #######################################
#data
models = ['Unigram', 'Bigram', 'Custom Unigram', 'Custom Bigram']
bleu_scores = [0.4044, 1.4795, 3.1071, 2.6473]
accuracy_scores = [0.1121, 0.2176, 0.2801, 0.2533]
f1_scores = [0.1071, 0.2460, 0.2056, 0.3116]

#plotting
plt.figure(figsize=(10, 6))
plt.plot(models, bleu_scores, marker='o', label='BLEU Score')
plt.plot(models, accuracy_scores, marker='o', label='Accuracy')
plt.plot(models, f1_scores, marker='o', label='F1 Score')

plt.title('Model Performance Comparison')
plt.xlabel('Model Variant')
plt.ylabel('Score')
plt.ylim(0, max(bleu_scores) + 0.5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



############################### Radar Chart #######################################
#data for radar chart
labels = ['BLEU', 'Accuracy', 'F1 Score']
num_vars = len(labels)

models_b = {
    'Unigram': [0.4044, 0.1121, 0.1071],
    'Bigram': [1.4795, 0.2176, 0.2460],
    'Custom Unigram': [3.1071, 0.2801, 0.2056],
    'Custom Bigram': [2.6473, 0.2533, 0.3116]
}

#setup
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

#plot each model
for model_name, stats in models_b.items():
    stats += stats[:1]  # Close the loop
    ax.plot(angles, stats, label=model_name)
    ax.fill(angles, stats, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title('Model Performance (Radar Chart)', size=16)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.tight_layout()
plt.show()

