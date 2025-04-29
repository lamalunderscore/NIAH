import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import os
import glob
from pathlib import Path
import yaml

CONF_FILE = "config.yaml"

if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / CONF_FILE
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    parent_dir = Path(config["parent_dir"])
    results_glob = parent_dir / config["eval"]["save_dir"] / "*.json"
    print(results_glob)
    vis_path = parent_dir / config["vis"]["save_dir"]

    result_files = glob.glob(str(results_glob))
    os.makedirs(vis_path, exist_ok=True)

    for file in result_files:
        data = []

        with open(file, "r") as f:
            json_data = json.load(f)

            for k, v in json_data.items():
                try:
                    # Extract document and context length
                    parts = k.split("_")
                    if len(parts) == 3:
                        context_length = int(parts[1])
                        document_depth = float(parts[2])
                        score = v["score"]

                        data.append(
                            {
                                "Score": score,
                                "Document_Depth": document_depth,
                                "Context_Length": context_length,
                            }
                        )
                except Exception as e:
                    print(f"Error processing key {k}: {str(e)}")

        df = pd.DataFrame(data)

        if df.empty:
            print("DataFrame is empty!")
            continue

        pivot_table = pd.pivot_table(
            df,
            values="Score",
            index="Document_Depth",
            columns="Context_Length",
            aggfunc="mean",
        )

        plt.figure(figsize=(17.5, 8))
        sns.heatmap(
            pivot_table,
            fmt="g",
            cmap=LinearSegmentedColormap.from_list(
                "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
            ),
            cbar_kws={"label": "Score"},
            vmin=1,
            vmax=10,
        )

        plt.title(
            'Pressure Testing\nFact Retrieval Across Context Lengths ("Needle In A HayStack")'
        )
        plt.xlabel("Token Limit")
        plt.ylabel("Depth Percent")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        png_name = f"{os.path.splitext(os.path.basename(file))[0]}.png"
        save_path = vis_path / png_name

        plt.savefig(save_path)
        plt.close()
