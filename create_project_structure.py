import os

# 定义项目目录结构
project_structure = {
    "config": ["default.yaml"],
    "data": ["raw", "processed", "generated"],
    "src": {
        "__init__.py": "",
        "utils": ["fairness_metrics.py", "logger.py", "loader.py"],
        "detection": ["auto_sensitive.py"],
        "df_training": ["trainer.py", "loss.py"],
        "causal": ["causal_graph.py", "intervention.py", "generator.py"],
        "models": ["mlp.py", "tabular_baseline.py"]
    },
    "notebooks": ["0_data_overview.ipynb", "1_run_sensitivity.ipynb", "2_run_df_training.ipynb", "3_run_causal_mitigation.ipynb"],
    "experiments": ["run_baseline.py", "run_with_auto_df.py", "run_with_auto_df_causal.py"],
    "results": {
        "logs": [],
        "plots": [],
        "models": []
    }
}

# 创建目录和文件
def create_structure(base_path, structure):
    for item, content in structure.items():
        full_path = os.path.join(base_path, item)
        if isinstance(content, dict):
            os.makedirs(full_path, exist_ok=True)
            create_structure(full_path, content)
        elif isinstance(content, list):
            os.makedirs(full_path, exist_ok=True)
            for file in content:
                if file.endswith(".py") or file.endswith(".ipynb") or file.endswith(".yaml"):
                    open(os.path.join(full_path, file), "w").close()
        else:
            open(full_path, "w").close()

# 主函数
if __name__ == "__main__":
    base_path = r"e:\UIUC\Algorithm Fairness and Justice\intersectional-fairness"
    create_structure(base_path, project_structure)