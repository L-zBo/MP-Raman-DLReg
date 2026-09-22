"""
实验环境配置表生成

输出: output/tables/environment.csv, environment.tex
"""

import sys
import platform
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import get_config, get_logger


def get_environment_info() -> list:
    """获取实验环境信息"""
    info = []

    # 操作系统
    info.append({
        'Category': 'System',
        'Item': 'Operating System',
        'Value': f'{platform.system()} {platform.release()}'
    })

    info.append({
        'Category': 'System',
        'Item': 'Python Version',
        'Value': platform.python_version()
    })

    # PyTorch
    try:
        import torch
        info.append({
            'Category': 'Framework',
            'Item': 'PyTorch Version',
            'Value': torch.__version__
        })

        if torch.cuda.is_available():
            info.append({
                'Category': 'Hardware',
                'Item': 'GPU',
                'Value': torch.cuda.get_device_name(0)
            })
            info.append({
                'Category': 'Hardware',
                'Item': 'CUDA Version',
                'Value': torch.version.cuda
            })
            info.append({
                'Category': 'Hardware',
                'Item': 'GPU Memory',
                'Value': f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB'
            })
        else:
            info.append({
                'Category': 'Hardware',
                'Item': 'GPU',
                'Value': 'N/A (CPU only)'
            })
    except ImportError:
        info.append({
            'Category': 'Framework',
            'Item': 'PyTorch Version',
            'Value': 'Not installed'
        })

    # NumPy
    info.append({
        'Category': 'Framework',
        'Item': 'NumPy Version',
        'Value': np.__version__
    })

    # Scikit-learn
    try:
        import sklearn
        info.append({
            'Category': 'Framework',
            'Item': 'Scikit-learn Version',
            'Value': sklearn.__version__
        })
    except ImportError:
        pass

    # CPU信息
    import os
    cpu_count = os.cpu_count()
    info.append({
        'Category': 'Hardware',
        'Item': 'CPU Cores',
        'Value': str(cpu_count)
    })

    # 内存
    try:
        import psutil
        mem = psutil.virtual_memory()
        info.append({
            'Category': 'Hardware',
            'Item': 'System Memory',
            'Value': f'{mem.total / 1e9:.1f} GB'
        })
    except ImportError:
        pass

    return info


def get_training_config(config) -> list:
    """获取训练配置信息"""
    info = []

    training = config.training
    model = config.model['architecture']

    info.append({
        'Category': 'Training',
        'Item': 'Batch Size',
        'Value': str(training['batch_size'])
    })

    info.append({
        'Category': 'Training',
        'Item': 'Epochs',
        'Value': str(training['epochs'])
    })

    info.append({
        'Category': 'Training',
        'Item': 'Learning Rate',
        'Value': str(training['learning_rate'])
    })

    info.append({
        'Category': 'Training',
        'Item': 'Weight Decay',
        'Value': str(training['weight_decay'])
    })

    info.append({
        'Category': 'Training',
        'Item': 'Optimizer',
        'Value': 'AdamW'
    })

    info.append({
        'Category': 'Training',
        'Item': 'LR Scheduler',
        'Value': training['scheduler']['type']
    })

    info.append({
        'Category': 'Training',
        'Item': 'Early Stopping Patience',
        'Value': str(training['early_stopping']['patience'])
    })

    info.append({
        'Category': 'Model',
        'Item': 'Input Length',
        'Value': str(model['input_len'])
    })

    info.append({
        'Category': 'Model',
        'Item': 'Conv Channels',
        'Value': str(model['conv_channels'])
    })

    info.append({
        'Category': 'Model',
        'Item': 'LSTM Hidden Size',
        'Value': str(model['lstm_hidden'])
    })

    info.append({
        'Category': 'Model',
        'Item': 'Dropout',
        'Value': str(model['dropout'])
    })

    info.append({
        'Category': 'Model',
        'Item': 'Number of Classes',
        'Value': str(model['num_classes'])
    })

    return info


def generate_environment_table(config, output_dir: Path):
    """生成实验环境配置表"""
    logger = get_logger('environment')

    output_dir.mkdir(parents=True, exist_ok=True)

    # 环境信息
    env_info = get_environment_info()

    # 训练配置
    train_info = get_training_config(config)

    # 合并
    all_info = env_info + train_info

    df = pd.DataFrame(all_info)
    df.to_csv(output_dir / 'environment.csv', index=False)

    # LaTeX
    latex_content = generate_environment_latex(df)
    with open(output_dir / 'environment.tex', 'w', encoding='utf-8') as f:
        f.write(latex_content)

    logger.info(f"实验环境配置表已保存到: {output_dir}")
    return df


def generate_environment_latex(df: pd.DataFrame) -> str:
    """生成LaTeX表格"""
    latex = []
    latex.append(r'\begin{table}[htbp]')
    latex.append(r'\centering')
    latex.append(r'\caption{Experimental Environment and Configuration}')
    latex.append(r'\label{tab:environment}')
    latex.append(r'\begin{tabular}{lll}')
    latex.append(r'\toprule')
    latex.append(r'Category & Item & Value \\')
    latex.append(r'\midrule')

    current_category = None
    for _, row in df.iterrows():
        cat_col = row['Category'] if row['Category'] != current_category else ''
        current_category = row['Category']
        latex.append(f"{cat_col} & {row['Item']} & {row['Value']} \\\\")

    latex.append(r'\bottomrule')
    latex.append(r'\end{tabular}')
    latex.append(r'\end{table}')

    return '\n'.join(latex)


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('environment')

    logger.info("=" * 60)
    logger.info("生成实验环境配置表")
    logger.info("=" * 60)

    output_dir = Path(config.paths['output_dir']) / 'tables' / 'environment'
    df = generate_environment_table(config, output_dir)

    print("\n实验环境配置:")
    print(df.to_string(index=False))
