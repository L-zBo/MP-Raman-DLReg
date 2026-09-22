"""
模型性能对比可视化 - 重定向到训练脚本

此脚本直接调用 training/model_comparison_Acc.py 来训练所有对比模型并生成可视化图表。
training/model_comparison_Acc.py 已经包含了完整的训练和可视化功能。
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入并运行训练脚本的主函数
from training.model_comparison_Acc import main

if __name__ == '__main__':
    print("=" * 60)
    print("模型对比可视化")
    print("=" * 60)
    print("正在调用 training/model_comparison_Acc.py...")
    print("该脚本会训练所有对比模型并自动生成可视化图表")
    print("=" * 60)
    print()

    # 运行主函数
    main()
