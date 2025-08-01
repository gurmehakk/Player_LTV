"""
Create comprehensive summary visualization showing all key results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_final_summary():
    """Create final summary visualization with all key metrics"""
    
    # Load the detailed metrics
    metrics_df = pd.read_csv('output/detailed_precision_recall_metrics.csv')
    
    # Create comprehensive summary plot
    fig = plt.figure(figsize=(20, 12))
    
    # Define layout - 3 rows, 3 columns with specific subplot arrangements
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    # 1. AUC Comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    auc_data = metrics_df[['target', 'auc_pr', 'auc_roc']].copy()
    auc_data['target'] = auc_data['target'].str.replace('_', ' ').str.title()
    
    x = np.arange(len(auc_data))
    width = 0.35
    
    ax1.bar(x - width/2, auc_data['auc_pr'], width, label='AUC-PR', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, auc_data['auc_roc'], width, label='AUC-ROC', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Classification Target')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('AUC Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(auc_data['target'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Precision at Different Recall Levels (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    recall_levels = [50, 70, 80, 90, 95]
    precision_cols = [f'precision_at_{level}' for level in recall_levels]
    
    for i, (_, row) in enumerate(metrics_df.iterrows()):
        target = row['target'].replace('_', ' ').title()
        precisions = [row[col] for col in precision_cols]
        ax2.plot(recall_levels, precisions, marker='o', linewidth=2, label=target)
    
    ax2.set_xlabel('Recall Level (%)')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Recall Trade-off')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.4)
    
    # 3. F1 Scores at Different Recall Levels (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    f1_cols = [f'f1_at_{level}' for level in recall_levels]
    
    for i, (_, row) in enumerate(metrics_df.iterrows()):
        target = row['target'].replace('_', ' ').title()
        f1_scores = [row[col] for col in f1_cols]
        ax3.plot(recall_levels, f1_scores, marker='s', linewidth=2, label=target)
    
    ax3.set_xlabel('Recall Level (%)')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score vs Recall Trade-off')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 0.4)
    
    # 4. Metrics at 90% Recall Heatmap (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    metrics_90 = metrics_df[['target', 'precision_at_90', 'f1_at_90', 'threshold_at_90']].copy()
    metrics_90['target'] = metrics_90['target'].str.replace('_', ' ').str.title()
    metrics_90 = metrics_90.set_index('target')
    metrics_90.columns = ['Precision', 'F1 Score', 'Threshold']
    
    sns.heatmap(metrics_90, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax4, 
                cbar_kws={'label': 'Score'})
    ax4.set_title('Metrics at 90% Recall')
    ax4.set_ylabel('')
    
    # 5. Business Value Matrix (middle middle)
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Create business value assessment
    business_value = []
    for _, row in metrics_df.iterrows():
        target = row['target'].replace('_', ' ').title()
        precision_90 = row['precision_at_90']
        auc_pr = row['auc_pr']
        
        # Calculate business value score (combination of precision at high recall and overall AUC)
        value_score = (precision_90 * 0.6 + auc_pr * 0.4)  # Weight high recall precision more
        business_value.append({
            'target': target,
            'precision_90': precision_90,
            'auc_pr': auc_pr,
            'value_score': value_score
        })
    
    bv_df = pd.DataFrame(business_value)
    
    scatter = ax5.scatter(bv_df['precision_90'], bv_df['auc_pr'], 
                         s=bv_df['value_score']*1000, alpha=0.6, 
                         c=bv_df['value_score'], cmap='viridis')
    
    for i, row in bv_df.iterrows():
        ax5.annotate(row['target'], (row['precision_90'], row['auc_pr']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax5.set_xlabel('Precision at 90% Recall')
    ax5.set_ylabel('AUC-PR')
    ax5.set_title('Business Value Assessment\\n(Size = Overall Value Score)')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Value Score')
    
    # 6. Threshold Analysis (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    threshold_data = []
    
    for _, row in metrics_df.iterrows():
        target = row['target'].replace('_', ' ').title()
        for recall_level in recall_levels:
            threshold = row[f'threshold_at_{recall_level}']
            precision = row[f'precision_at_{recall_level}']
            threshold_data.append({
                'target': target,
                'recall_level': recall_level,
                'threshold': threshold,
                'precision': precision
            })
    
    threshold_df = pd.DataFrame(threshold_data)
    
    # Create threshold vs precision plot
    for target in threshold_df['target'].unique():
        target_data = threshold_df[threshold_df['target'] == target]
        ax6.plot(target_data['threshold'], target_data['precision'], 
                marker='o', linewidth=2, label=target)
    
    ax6.set_xlabel('Classification Threshold')
    ax6.set_ylabel('Precision')
    ax6.set_title('Threshold vs Precision')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log')
    
    # 7. Summary Table (bottom span)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')
    
    # Create summary table
    summary_data = []
    for _, row in metrics_df.iterrows():
        target = row['target'].replace('_', ' ').title()
        summary_data.append([
            target,
            f"{row['auc_pr']:.3f}",
            f"{row['auc_roc']:.3f}",
            f"{row['precision_at_90']:.1%}",
            f"{row['f1_at_90']:.3f}",
            f"{row['threshold_at_90']:.3f}",
            "✅" if row['precision_at_90'] > 0.15 else "⚠️" if row['precision_at_90'] > 0.05 else "❌"
        ])
    
    table = ax7.table(cellText=summary_data,
                     colLabels=['Target', 'AUC-PR', 'AUC-ROC', 'Precision@90%', 'F1@90%', 'Threshold@90%', 'Viability'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(7):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                if j == 6:  # Viability column
                    if summary_data[i-1][6] == "✅":
                        table[(i, j)].set_facecolor('#E8F5E8')
                    elif summary_data[i-1][6] == "⚠️":
                        table[(i, j)].set_facecolor('#FFF3E0')
                    else:
                        table[(i, j)].set_facecolor('#FFEBEE')
                else:
                    table[(i, j)].set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')
    
    ax7.set_title('LTV Classification Model Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Add overall title
    fig.suptitle('LTV Classification Analysis - Complete Results Summary', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add explanatory text
    fig.text(0.02, 0.02, 
            '✅ Good (>15% precision at 90% recall)  ⚠️ Fair (5-15% precision)  ❌ Poor (<5% precision)\\n' +
            'Best Model: Low Spender (22.5% precision at 90% recall, suitable for broad targeting campaigns)',
            fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.08)
    plt.savefig('output/complete_ltv_analysis_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📊 Complete summary visualization saved to 'output/complete_ltv_analysis_summary.png'")
    
    # Create a simple metrics table for easy reference
    simple_summary = pd.DataFrame({
        'Model': [row['target'].replace('_', ' ').title() for _, row in metrics_df.iterrows()],
        'AUC-PR': [f"{row['auc_pr']:.3f}" for _, row in metrics_df.iterrows()],
        'AUC-ROC': [f"{row['auc_roc']:.3f}" for _, row in metrics_df.iterrows()],
        'Precision_at_90%_Recall': [f"{row['precision_at_90']:.1%}" for _, row in metrics_df.iterrows()],
        'F1_at_90%_Recall': [f"{row['f1_at_90']:.3f}" for _, row in metrics_df.iterrows()],
        'Threshold_at_90%_Recall': [f"{row['threshold_at_90']:.4f}" for _, row in metrics_df.iterrows()],
        'Business_Recommendation': [
            'Best for broad campaigns' if row['precision_at_90'] > 0.2 else
            'Good for targeted campaigns' if row['precision_at_90'] > 0.1 else
            'Limited practical use' for _, row in metrics_df.iterrows()
        ]
    })
    
    simple_summary.to_csv('output/ltv_classification_summary_table.csv', index=False)
    print("📋 Simple summary table saved to 'output/ltv_classification_summary_table.csv'")
    
    return simple_summary

if __name__ == "__main__":
    summary = create_final_summary()
    print("\\n📊 FINAL RESULTS SUMMARY:")
    print("=" * 60)
    print(summary.to_string(index=False))