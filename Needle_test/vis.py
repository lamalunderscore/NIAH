import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import os
import re
import glob

if __name__ == '__main__':
    json_files = glob.glob("LongAlign/Needle_test/results/*.json")

    if not os.path.exists('vis'):
        os.makedirs('vis')

    for file in json_files:
        data = []
        
        with open(file, 'r') as f:
            json_data = json.load(f)
            
            for k, v in json_data.items():
                try:
                    # Extract numbers from Huggingface_XXXX_YY format
                    parts = k.split('_')
                    if len(parts) == 3:
                        context_length = int(parts[1])
                        document_depth = float(parts[2])
                        score = v['score']
                        
                        data.append({
                            "Score": score,
                            "Document_Depth": document_depth,
                            "Context_Length": context_length
                        })
                except Exception as e:
                    print(f"Error processing key {k}: {str(e)}")

        df = pd.DataFrame(data)
        
        if df.empty:
            print("DataFrame is empty!")
            continue

        pivot_table = pd.pivot_table(df, 
                                   values='Score',
                                   index='Document_Depth',
                                   columns='Context_Length',
                                   aggfunc='mean')

        plt.figure(figsize=(17.5, 8))
        sns.heatmap(pivot_table,
                   fmt="g",
                   cmap=LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]),
                   cbar_kws={'label': 'Score'},
                   vmin=1,
                   vmax=10)
        
        plt.title('Pressure Testing\nFact Retrieval Across Context Lengths ("Needle In A HayStack")')
        plt.xlabel('Token Limit')
        plt.ylabel('Depth Percent')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"vis/{os.path.splitext(os.path.basename(file))[0]}.png")
        plt.close()
