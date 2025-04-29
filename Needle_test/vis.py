import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
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

    parent_dir = Path(config["parent_dir"]).resolve()
    results_glob = parent_dir / config["eval"]["save_dir"] / "*.json"
    vis_path = parent_dir / config["vis"]["save_dir"]

    result_files = glob.glob(str(results_glob))
    os.makedirs(vis_path, exist_ok=True)

    for file_path in result_files:
        data = []

        with open(file_path, "r") as f:
            json_data = json.load(f)

            for k, v in json_data["detailed_results"].items():
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

        # Binary colormap: red for 0, green for 10
        binary_cmap = ListedColormap(["#F0496E", "#0CD79F"])

        plt.figure(figsize=(9.0, 8))
        ax = sns.heatmap(
            pivot_table,
            fmt="g",
            cmap=binary_cmap,
            cbar=False,  # Disable default colorbar
            vmin=0,
            vmax=10,
        )

        # Custom binary legend
        legend_elements = [
            Patch(
                facecolor="#F0496E", edgecolor="black", label="Unsuccessful"
            ),
            Patch(facecolor="#0CD79F", edgecolor="black", label="Successful"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            frameon=True,
            fontsize="medium",
        )
        file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        k = file_name_no_ext.split("_")[2][1:]
        plt.title(f"Needle In A HayStack - K={k}", fontweight="bold")

        plt.xlabel("Token Limit", fontweight="bold")
        plt.ylabel("Depth Percent", fontweight="bold")
        plt.xticks(rotation=45, fontweight="bold")
        plt.yticks(rotation=0, fontweight="bold")
        plt.tight_layout()

        png_name = f"{file_name_no_ext}.png"
        save_path = vis_path / png_name

        plt.savefig(save_path, dpi=600)
        plt.close()
