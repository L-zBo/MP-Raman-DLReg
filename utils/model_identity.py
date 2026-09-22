"""
Centralized display names for the primary model.
"""

PRIMARY_MODEL_NAME = 'RADAR-Net'
PRIMARY_MODEL_FULL_NAME = 'Raman Abundance-guided Dual-head Attentive Recurrent Network'

RESNET_VARIANT_NAME = 'ResNet-50 (1D Bottleneck [3,4,6,3])'

# Best candidate settings discovered so far.
# These are written here so future experiments can reuse them directly instead
# of depending on chat history or scattered CSV files.
BEST_CANDIDATE_CONFIGS = {
    'SMART-NIR': {
        'pp_best': {
            'tag': 'smart_nir_pp_final',
            'source': 'output/comparison_final/final_comparison_results.csv',
            'model_kwargs': {
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 4,
                'patch_size': 16,
                'branch_channels': 16,
                'dropout': 0.15,
            },
            'train_kwargs': {
                'neg_ratio_train': 0.25,
                'neg_ratio_val': 0.25,
                'epochs': 80,
                'lr': 1.5e-04,
                'weight_decay': 4e-04,
                'batch_size': 64,
                'patience': 16,
                'label_smoothing': 0.02,
                'use_weighted_sampler': False,
            },
        },
        'pe_best': {
            'tag': 'smart_nir_pe_final',
            'source': 'output/comparison_final/final_comparison_results.csv',
            'model_kwargs': {
                'd_model': 48,
                'num_heads': 4,
                'num_layers': 3,
                'patch_size': 16,
                'branch_channels': 12,
                'dropout': 0.15,
            },
            'train_kwargs': {
                'epochs': 50,
                'lr': 2e-04,
                'weight_decay': 4e-04,
                'batch_size': 64,
                'patience': 12,
                'label_smoothing': 0.02,
                'use_weighted_sampler': True,
            },
        },
    },
    'ConvTran': {
        'pp_best': {
            'tag': 'convtran_pp_final',
            'source': 'output/comparison_final/final_comparison_results.csv',
            'model_kwargs': {
                'emb_size': 96,
                'num_heads': 4,
                'num_layers': 2,
                'dim_ff': 192,
                'patch_size': 16,
                'conv_expansion': 3,
                'dropout': 0.32,
            },
            'train_kwargs': {
                'neg_ratio_train': 0.50,
                'neg_ratio_val': 0.50,
                'epochs': 95,
                'lr': 5e-05,
                'weight_decay': 0.0012,
                'batch_size': 64,
                'patience': 20,
                'label_smoothing': 0.06,
                'use_weighted_sampler': True,
            },
        },
        'pe_best': {
            'tag': 'convtran_pe_final',
            'source': 'output/comparison_final/final_comparison_results.csv',
            'model_kwargs': {
                'emb_size': 64,
                'num_heads': 4,
                'num_layers': 2,
                'dim_ff': 160,
                'patch_size': 16,
                'conv_expansion': 2,
                'dropout': 0.15,
            },
            'train_kwargs': {
                'epochs': 65,
                'lr': 2e-04,
                'weight_decay': 3e-04,
                'batch_size': 64,
                'patience': 14,
                'label_smoothing': 0.02,
                'use_weighted_sampler': True,
            },
        },
    },
    'MambaHSI': {
        'pp_best': {
            'tag': 'mambahsi_pp_final',
            'source': 'output/comparison_final/final_comparison_results.csv',
            'model_kwargs': {
                'd_model': 96,
                'patch_size': 16,
                'num_layers': 4,
                'num_groups': 8,
                'dropout': 0.15,
            },
            'train_kwargs': {
                'neg_ratio_train': 0.25,
                'neg_ratio_val': 0.25,
                'epochs': 90,
                'lr': 8e-05,
                'weight_decay': 8e-04,
                'batch_size': 64,
                'patience': 18,
                'label_smoothing': 0.05,
                'use_weighted_sampler': False,
            },
        },
        'pe_best': {
            'tag': 'mambahsi_pe_final',
            'source': 'output/comparison_final/final_comparison_results.csv',
            'model_kwargs': {
                'd_model': 96,
                'patch_size': 16,
                'num_layers': 4,
                'num_groups': 8,
                'dropout': 0.10,
            },
            'train_kwargs': {
                'epochs': 50,
                'lr': 1.5e-04,
                'weight_decay': 2e-04,
                'batch_size': 64,
                'patience': 12,
                'label_smoothing': 0.02,
                'use_weighted_sampler': True,
            },
        },
    },
}

# Final comparison-model replacement decision currently adopted for manuscript
# discussion and future experiment consolidation.
FINAL_COMPARISON_REPLACEMENTS = {
    '1D-CNN': 'SMART-NIR',
    '1D-Transformer': 'ConvTran',
    'PLS-DA': 'MambaHSI',
}

FINAL_DROPPED_CANDIDATES = ['RS-MLP', 'LITE']

FINAL_RESNET_CHOICE = 'ResNet-50'

FINAL_RESNET_CONFIG = {
    'pp_best': {
        'tag': 'resnet50_pp_final',
        'source': 'output/comparison_final/final_comparison_results.csv',
        'model_kwargs': {
            'dropout': 0.25,
        },
        'train_kwargs': {
            'neg_ratio_train': 0.50,
            'neg_ratio_val': 0.25,
            'epochs': 100,
            'lr': 5e-05,
            'weight_decay': 0.0012,
            'batch_size': 64,
            'patience': 20,
            'label_smoothing': 0.06,
            'use_weighted_sampler': False,
        },
    },
    'pe_best': {
        'tag': 'resnet50_pe_final',
        'source': 'output/comparison_final/final_comparison_results.csv',
        'model_kwargs': {
            'dropout': 0.0,
        },
        'train_kwargs': {
            'epochs': 80,
            'lr': 1.0e-04,
            'weight_decay': 8.0e-04,
            'batch_size': 64,
            'patience': 18,
            'label_smoothing': 0.03,
            'use_weighted_sampler': True,
        },
    },
}

FINAL_COMPARISON_MODEL_ORDER = [
    PRIMARY_MODEL_NAME,
    'XGBoost',
    'SVM',
    'Random Forest',
    'LightGBM',
    FINAL_RESNET_CHOICE,
    'SMART-NIR',
    'ConvTran',
    'MambaHSI',
]
