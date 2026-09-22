"""
模型对比结果表格生成（论文格式）

读取交叉验证结果，生成适合论文使用的表格格式
输出: output/tables/comparison_table/

表格内容：
- Model: 模型名称
- Accuracy (Mean ± Std): 准确率均值±标准差
- F1-Score (Macro)
- Recall (Macro)
- Precision (Macro)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import get_config, get_logger


def load_cv_results(config, task: str = 'PP') -> dict:
    """加载交叉验证结果"""
    cv_dir = Path(config.paths['output_dir']) / 'models' / 'cross_validation' / task
    json_path = cv_dir / 'cv_results.json'

    if not json_path.exists():
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_comparison_results(config, task: str = 'PP') -> pd.DataFrame:
    """加载model_comparison_Acc的结果"""
    comparison_dir = Path(config.base_dir) / 'output' / 'comparison_Acc' / task
    csv_path = comparison_dir / 'comparison_results.csv'

    if not csv_path.exists():
        return None

    return pd.read_csv(csv_path)


def format_mean_std(values: list, decimals: int = 4) -> str:
    """格式化为 mean ± std 形式"""
    mean = np.mean(values)
    std = np.std(values)
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def generate_cv_table(config, output_dir: Path):
    """生成交叉验证结果表格（论文格式）"""
    logger = get_logger('comparison_table')

    output_dir.mkdir(parents=True, exist_ok=True)

    for task in ['PP', 'PE']:
        logger.info(f"生成 {task} 任务交叉验证表格...")

        cv_results = load_cv_results(config, task)
        if cv_results is None:
            logger.warning(f"未找到 {task} 任务的交叉验证结果")
            continue

        # 构建表格数据
        table_data = []
        for model_name, data in cv_results.items():
            accs = data['accuracy']
            mean = np.mean(accs)
            std = np.std(accs)

            table_data.append({
                'Model': model_name,
                'Accuracy (Macro)': f"{mean:.4f} ± {std:.4f}",
                'Mean': mean,
                'Std': std,
                'N_folds': len(accs)
            })

        # 按均值排序
        df = pd.DataFrame(table_data)
        df = df.sort_values('Mean', ascending=False)

        # 保存完整版CSV（含原始数值）
        csv_path = output_dir / f'cv_results_{task}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"保存: {csv_path}")

        # 保存论文版CSV（只含格式化结果）
        paper_df = df[['Model', 'Accuracy (Macro)', 'N_folds']].copy()
        paper_csv_path = output_dir / f'cv_results_{task}_paper.csv'
        paper_df.to_csv(paper_csv_path, index=False)
        logger.info(f"保存: {paper_csv_path}")

        # 生成LaTeX表格
        latex_content = generate_latex_table(
            df[['Model', 'Accuracy (Macro)']],
            caption=f"Cross-validation results for {task} task (3×5-fold, Macro Recall)",
            label=f"tab:cv_{task.lower()}"
        )
        latex_path = output_dir / f'cv_results_{task}.tex'
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        logger.info(f"保存: {latex_path}")


def generate_comparison_table(config, output_dir: Path):
    """生成模型对比结果表格（论文格式，4个指标）"""
    logger = get_logger('comparison_table')

    output_dir.mkdir(parents=True, exist_ok=True)

    for task in ['PP', 'PE']:
        logger.info(f"生成 {task} 任务模型对比表格...")

        df = load_comparison_results(config, task)
        if df is None:
            logger.warning(f"未找到 {task} 任务的对比实验结果")
            continue

        # 选择论文需要的列
        columns_map = {
            'Model': 'Model',
            'Accuracy': 'Accuracy',
            'Accuracy (Macro)': 'Balanced Acc',
            'F1-Score (Macro)': 'F1 (Macro)',
            'Recall (Macro)': 'Recall (Macro)',
            'Precision (Macro)': 'Precision (Macro)'
        }

        # 检查哪些列存在
        available_cols = [col for col in columns_map.keys() if col in df.columns]
        paper_df = df[available_cols].copy()
        paper_df.columns = [columns_map[col] for col in available_cols]

        # 格式化数值（保留4位小数）
        numeric_cols = paper_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            paper_df[col] = paper_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

        # 保存CSV
        csv_path = output_dir / f'comparison_{task}.csv'
        paper_df.to_csv(csv_path, index=False)
        logger.info(f"保存: {csv_path}")

        # 生成LaTeX表格
        latex_content = generate_latex_table(
            paper_df,
            caption=f"Model comparison results for {task} task",
            label=f"tab:comparison_{task.lower()}"
        )
        latex_path = output_dir / f'comparison_{task}.tex'
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        logger.info(f"保存: {latex_path}")


def generate_combined_table(config, output_dir: Path):
    """生成PP和PE合并的对比表格（论文主表格）"""
    logger = get_logger('comparison_table')

    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载PP和PE的交叉验证结果
    pp_results = load_cv_results(config, 'PP')
    pe_results = load_cv_results(config, 'PE')

    if pp_results is None or pe_results is None:
        logger.warning("缺少PP或PE的交叉验证结果，无法生成合并表格")
        return

    # 获取所有模型名称
    all_models = list(pp_results.keys())

    # 构建合并表格
    table_data = []
    for model in all_models:
        pp_accs = pp_results.get(model, {}).get('accuracy', [])
        pe_accs = pe_results.get(model, {}).get('accuracy', [])

        row = {
            'Model': model,
            'PP Accuracy': format_mean_std(pp_accs) if pp_accs else 'N/A',
            'PE Accuracy': format_mean_std(pe_accs) if pe_accs else 'N/A',
            'PP_Mean': np.mean(pp_accs) if pp_accs else 0,
            'PE_Mean': np.mean(pe_accs) if pe_accs else 0,
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    # 按PP均值排序
    df = df.sort_values('PP_Mean', ascending=False)

    # 保存完整版
    csv_path = output_dir / 'cv_results_combined.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"保存: {csv_path}")

    # 保存论文版（只含格式化结果）
    paper_df = df[['Model', 'PP Accuracy', 'PE Accuracy']].copy()
    paper_csv_path = output_dir / 'cv_results_combined_paper.csv'
    paper_df.to_csv(paper_csv_path, index=False)
    logger.info(f"保存: {paper_csv_path}")

    # 生成LaTeX表格
    latex_content = generate_latex_table(
        paper_df,
        caption="Cross-validation results for PP and PE tasks (3×5-fold, Macro Recall)",
        label="tab:cv_combined"
    )
    latex_path = output_dir / 'cv_results_combined.tex'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    logger.info(f"保存: {latex_path}")


def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """生成LaTeX表格代码"""
    n_cols = len(df.columns)
    col_format = 'l' + 'c' * (n_cols - 1)

    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex.append(r"\toprule")

    # 表头
    headers = ' & '.join(df.columns)
    latex.append(f"{headers} \\\\")
    latex.append(r"\midrule")

    # 数据行
    for _, row in df.iterrows():
        row_str = ' & '.join(str(v) for v in row.values)
        latex.append(f"{row_str} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    return '\n'.join(latex)


def main():
    """主函数"""
    config = get_config()
    logger = get_logger('comparison_table')

    logger.info("=" * 60)
    logger.info("生成论文格式对比表格")
    logger.info("=" * 60)

    # 输出目录
    output_dir = Path(config.base_dir) / 'output' / 'tables' / 'comparison_table'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成各类表格
    logger.info("\n>>> 生成交叉验证结果表格...")
    generate_cv_table(config, output_dir)

    logger.info("\n>>> 生成模型对比表格（4指标）...")
    generate_comparison_table(config, output_dir)

    logger.info("\n>>> 生成PP/PE合并表格...")
    generate_combined_table(config, output_dir)

    logger.info("\n>>> 完成!")
    logger.info(f"结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
