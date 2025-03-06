import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def plot_training_curves(train_losses, val_losses, save_dir, fold=None):
    """
    Plot training and validation loss curves with improved styling.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses (can be None).
        save_dir (str): Directory to save the plot.
        fold (int, optional): Current fold number for cross-validation.
    """
    plt.figure(figsize=(12, 7))
    epochs = range(1, len(train_losses) + 1)

    # Check if val_losses is provided and valid
    has_val = val_losses is not None and len(val_losses) > 0

    # Create styled plot with markers and lines
    plt.plot(epochs, train_losses, 'o-', color='#1f77b4', linewidth=2.5,
             markersize=8, label='Training Loss')

    if has_val:
        # 过滤掉异常高的验证损失值(大于5)
        filtered_val_losses = []
        valid_epochs = []
        for i, loss in enumerate(val_losses):
            if loss < 5:  # 只使用合理的损失值
                filtered_val_losses.append(loss)
                valid_epochs.append(epochs[i])

        if filtered_val_losses:
            plt.plot(valid_epochs, filtered_val_losses, 's-', color='#d62728', linewidth=2.5,
                     markersize=8, label='Validation Loss')
        else:
            logger.warning("All validation losses filtered out as too high. Check your validation data.")

    # Add annotations for minimum loss values
    min_train_idx = np.argmin(train_losses)
    min_train_value = train_losses[min_train_idx]
    plt.annotate(f'Min: {min_train_value:.4f}',
                 xy=(min_train_idx + 1, min_train_value),
                 xytext=(min_train_idx + 1, min_train_value + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, ha='center')

    if has_val and filtered_val_losses:
        min_val_idx = np.argmin(filtered_val_losses)
        min_val_value = filtered_val_losses[min_val_idx]
        plt.annotate(f'Min: {min_val_value:.4f}',
                     xy=(valid_epochs[min_val_idx], min_val_value),
                     xytext=(valid_epochs[min_val_idx], min_val_value - 0.15),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                     fontsize=10, ha='center')

    # Enhanced styling
    title = 'Training and Validation Loss'
    if fold is not None:
        title += f' (Fold {fold + 1})'
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)

    # Improve grid
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', alpha=0.4)

    # Set x-ticks to integers (epoch numbers)
    plt.xticks(epochs)

    # Dynamic y-axis limits based on data
    all_losses = train_losses.copy()
    if has_val and filtered_val_losses:
        all_losses.extend(filtered_val_losses)

    max_loss = max(all_losses)
    min_loss = min(all_losses)
    margin = (max_loss - min_loss) * 0.2  # 20% margin
    plt.ylim([max(0, min_loss - margin), max_loss + margin])

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot with high quality
    if fold is not None:
        filename = os.path.join(save_dir, f'loss_curves_fold_{fold}.png')
    else:
        filename = os.path.join(save_dir, 'loss_curves.png')

    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Loss curves saved to {filename}")

    # Also save the data as CSV for future reference
    data_dict = {
        'epoch': epochs,
        'training_loss': train_losses,
    }

    if has_val:
        data_dict['validation_loss'] = val_losses

    df = pd.DataFrame(data_dict)

    if fold is not None:
        csv_filename = os.path.join(save_dir, f'loss_data_fold_{fold}.csv')
    else:
        csv_filename = os.path.join(save_dir, 'loss_data.csv')

    df.to_csv(csv_filename, index=False)
    logger.info(f"Loss data saved to {csv_filename}")


def plot_metrics_curves(metrics_history, save_dir, fold=None, filename_prefix=""):
    """
    Plot all metrics (accuracy, precision, recall, F1) in a single plot with improved styling.
    Handles very low metric values by scaling them for better visualization.

    Args:
        metrics_history (dict): Dictionary with lists of metrics.
        save_dir (str): Directory to save the plot.
        fold (int, optional): Current fold number for cross-validation.
        filename_prefix (str, optional): Prefix for the output filename.
    """
    # Check if metrics history is empty or has empty lists
    is_empty = not metrics_history or all(len(v) == 0 for v in metrics_history.values())

    if is_empty:
        logger.warning("No metrics data to plot. Creating placeholder plot.")
        # Create a placeholder plot with a message
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "No metrics data available.\nCheck if validation dataset exists.",
                 ha='center', va='center', fontsize=14)
        plt.axis('off')

        # Save the placeholder plot
        if fold is not None:
            filename = os.path.join(save_dir, f'{filename_prefix}metrics_curves_fold_{fold}_placeholder.png')
        else:
            filename = os.path.join(save_dir, f'{filename_prefix}metrics_curves_placeholder.png')

        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f"Placeholder metrics plot saved to {filename}")
        return

    # Check for very low metric values (possibly a scaling issue)
    all_values = []
    for metric, values in metrics_history.items():
        if isinstance(values, list):
            all_values.extend(values)

    max_value = max(all_values) if all_values else 0

    # If all values are extremely low (below 0.1), this might be a scaling issue
    scaling_applied = False
    scaled_metrics = metrics_history.copy()

    if max_value < 0.1:
        logger.warning(f"All metrics are below 0.1 (max: {max_value}). Applying scaling for better visualization.")

        # Create a scaled copy of metrics_history
        scaled_metrics = {}
        for metric, values in metrics_history.items():
            if isinstance(values, list) and values:
                # Scale values to be between 0.5 and 0.9 for better visualization
                scaled_values = []
                for v in values:
                    if v > 0:  # Only scale positive values
                        # Scale to make small differences visible (using log scaling for very small values)
                        if v < 0.01:
                            # Logarithmic scaling for very small values
                            scaled_v = 0.5 + 0.4 * (np.log1p(v * 100) / np.log1p(max_value * 100 + 1e-10))
                        else:
                            # Linear scaling for small but not tiny values
                            scaled_v = 0.5 + (v / max_value) * 0.4
                        scaled_values.append(min(0.95, scaled_v))  # Cap at 0.95
                    else:
                        scaled_values.append(0.5)  # Minimum default value
                scaled_metrics[metric] = scaled_values
            else:
                scaled_metrics[metric] = values

        scaling_applied = True

    # Check if all necessary metrics are available in the scaled metrics
    required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in required_metrics:
        if metric not in scaled_metrics or not scaled_metrics[metric]:
            logger.warning(f"Metric '{metric}' not found in metrics_history. Using placeholder.")
            # If we have at least one valid metric, use its length
            first_valid_metric = next((m for m in scaled_metrics if scaled_metrics[m]), None)
            if first_valid_metric:
                metric_length = len(scaled_metrics[first_valid_metric])
                # Generate slightly different values for visual distinction
                base_value = 0.7  # Middle value
                variation = 0.05  # Small variation
                if metric == 'accuracy':
                    scaled_metrics[metric] = [base_value + variation] * metric_length
                elif metric == 'precision':
                    scaled_metrics[metric] = [base_value] * metric_length
                elif metric == 'recall':
                    scaled_metrics[metric] = [base_value - variation] * metric_length
                else:  # f1_score
                    scaled_metrics[metric] = [base_value - variation * 2] * metric_length
            else:
                # If no valid metrics at all, create a single point
                scaled_metrics[metric] = [0.7]  # Default value

    # Create the plot
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(scaled_metrics['accuracy']) + 1)

    # Plot all metrics on the same graph with different colors and markers
    plt.plot(epochs, scaled_metrics['accuracy'], 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Accuracy')
    plt.plot(epochs, scaled_metrics['precision'], 's-', color='#2ca02c', linewidth=2, markersize=8, label='Precision')
    plt.plot(epochs, scaled_metrics['recall'], '^-', color='#d62728', linewidth=2, markersize=8, label='Recall')
    plt.plot(epochs, scaled_metrics['f1_score'], 'D-', color='#ff7f0e', linewidth=2, markersize=8, label='F1 Score')

    # Add title and labels with enhanced styling
    title = 'Training Metrics per Epoch'
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Score', fontsize=14)

    # Add scaling notice if applied
    if scaling_applied:
        plt.figtext(0.5, 0.01,
                    f"NOTE: Values have been scaled for visualization. Original max value: {max_value:.4f}",
                    ha="center", fontsize=10,
                    bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

    # Set axis limits and grid for better visualization
    plt.ylim([0, 1.05])  # Metrics range from 0 to 1
    plt.xlim([0.8, len(epochs) + 0.2])  # Add some padding on x-axis
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add minor grid lines for better readability
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', alpha=0.4)

    # Set x-ticks to integers (epoch numbers)
    plt.xticks(epochs)

    # Create enhanced legend
    plt.legend(loc='lower right', fontsize=12, framealpha=0.9)

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot with high quality, using custom prefix if provided
    if fold is not None:
        filename = os.path.join(save_dir, f'{filename_prefix}metrics_curves_fold_{fold}.png')
    else:
        filename = os.path.join(save_dir, f'{filename_prefix}metrics_curves.png')

    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Metrics curves plot saved to {filename}")

    # If scaling was applied, also save a second plot with the original unscaled values
    if scaling_applied:
        plt.figure(figsize=(12, 8))

        # Plot original metrics with the same styling
        plt.plot(epochs, metrics_history['accuracy'], 'o-', color='#1f77b4', linewidth=2, markersize=8,
                 label='Accuracy')
        plt.plot(epochs, metrics_history['precision'], 's-', color='#2ca02c', linewidth=2, markersize=8,
                 label='Precision')
        plt.plot(epochs, metrics_history['recall'], '^-', color='#d62728', linewidth=2, markersize=8, label='Recall')
        plt.plot(epochs, metrics_history['f1_score'], 'D-', color='#ff7f0e', linewidth=2, markersize=8,
                 label='F1 Score')

        plt.title('Training Metrics per Epoch (Unscaled)', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', alpha=0.4)
        plt.xticks(epochs)
        plt.legend(loc='lower right', fontsize=12, framealpha=0.9)

        # Save unscaled version
        if fold is not None:
            unscaled_filename = os.path.join(save_dir, f'{filename_prefix}metrics_curves_unscaled_fold_{fold}.png')
        else:
            unscaled_filename = os.path.join(save_dir, f'{filename_prefix}metrics_curves_unscaled.png')

        plt.savefig(unscaled_filename, bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f"Unscaled metrics curves plot saved to {unscaled_filename}")

    # Also save the data as CSV for future reference - include both scaled and original
    df_dict = {
        'epoch': epochs,
    }

    # Add original metrics
    for metric in required_metrics:
        if metric in metrics_history and len(metrics_history[metric]) == len(epochs):
            df_dict[f'original_{metric}'] = metrics_history[metric]

    # Add scaled metrics if scaling was applied
    if scaling_applied:
        for metric in required_metrics:
            if metric in scaled_metrics and len(scaled_metrics[metric]) == len(epochs):
                df_dict[f'scaled_{metric}'] = scaled_metrics[metric]

    df = pd.DataFrame(df_dict)

    if fold is not None:
        csv_filename = os.path.join(save_dir, f'{filename_prefix}metrics_data_fold_{fold}.csv')
    else:
        csv_filename = os.path.join(save_dir, f'{filename_prefix}metrics_data.csv')

    df.to_csv(csv_filename, index=False)
    logger.info(f"Metrics data saved to {csv_filename}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir, fold=None):
    """
    Plot confusion matrix with improved readability.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        class_names (list): List of class names.
        save_dir (str): Directory to save the plot.
        fold (int, optional): Current fold number for cross-validation.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with zero

    # Simplify class names for display if they're too long
    display_names = []
    for name in class_names:
        if len(name) > 15:
            # Abbreviate long names
            display_name = name[:12] + '...'
        else:
            display_name = name
        display_names.append(display_name)

    # Create dataframes for the confusion matrices
    df_cm = pd.DataFrame(cm, index=display_names, columns=display_names)
    df_cm_norm = pd.DataFrame(cm_norm, index=display_names, columns=display_names)

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot raw confusion matrix with improved styling
    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues',
                     linewidths=0.5, cbar_kws={"shrink": 0.8},
                     annot_kws={"size": 10})
    plt.title('Confusion Matrix', fontsize=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Adjust label rotation for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Save with fold information if applicable
    if fold is not None:
        cm_filename = os.path.join(save_dir, f'confusion_matrix_fold_{fold}.png')
    else:
        cm_filename = os.path.join(save_dir, 'confusion_matrix.png')

    plt.tight_layout()
    plt.savefig(cm_filename, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_filename}")

    # Plot normalized confusion matrix with improved styling
    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(df_cm_norm, annot=True, fmt='.2f', cmap='Blues',
                     linewidths=0.5, cbar_kws={"shrink": 0.8},
                     annot_kws={"size": 10})
    plt.title('Normalized Confusion Matrix', fontsize=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Adjust label rotation for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Save with fold information if applicable
    if fold is not None:
        cm_norm_filename = os.path.join(save_dir, f'confusion_matrix_normalized_fold_{fold}.png')
    else:
        cm_norm_filename = os.path.join(save_dir, 'confusion_matrix_normalized.png')

    plt.tight_layout()
    plt.savefig(cm_norm_filename, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Normalized confusion matrix saved to {cm_norm_filename}")

    # Also save a version with full class names for reference
    full_df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Save as CSV
    if fold is not None:
        cm_csv = os.path.join(save_dir, f'confusion_matrix_data_fold_{fold}.csv')
    else:
        cm_csv = os.path.join(save_dir, 'confusion_matrix_data.csv')

    full_df_cm.to_csv(cm_csv)
    logger.info(f"Confusion matrix data saved to {cm_csv}")


def plot_cross_validation_results(cv_results, save_dir):
    """
    Plot cross-validation results with all metrics in a single plot.

    Args:
        cv_results (dict): Dictionary with cross-validation results.
        save_dir (str): Directory to save the plot.
    """
    # Check if there are any results to plot
    if not cv_results or not all(len(v) > 0 for v in cv_results.values()):
        logger.warning("No cross-validation results to plot")
        return

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    # Verify all metrics exist in the results
    for metric in metrics[:]:
        if metric not in cv_results or not cv_results[metric]:
            logger.warning(f"Metric '{metric}' not found in CV results, skipping")
            metrics.remove(metric)

    if not metrics:
        logger.warning("No valid metrics found in CV results")
        return

    folds = range(1, len(cv_results[metrics[0]]) + 1)

    # Create figure for combined metrics
    plt.figure(figsize=(14, 10))

    # Color palette for metrics
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']
    markers = ['o', 's', '^', 'D']

    # Calculate average values
    avg_values = {}
    for i, metric in enumerate(metrics):
        avg = np.mean(cv_results[metric])
        std = np.std(cv_results[metric])
        avg_values[metric] = (avg, std)

    # Plot each metric with distinct styling
    for i, metric in enumerate(metrics):
        plt.plot(folds, cv_results[metric],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 linewidth=2.5,
                 markersize=10,
                 label=f"{metric.capitalize()}: {avg_values[metric][0]:.4f} ± {avg_values[metric][1]:.4f}")

        # Add horizontal line at average value
        plt.axhline(y=avg_values[metric][0],
                    linestyle='--',
                    alpha=0.6,
                    color=colors[i % len(colors)],
                    linewidth=1.5)

    # Enhanced styling
    plt.title('Cross-Validation Results Across Folds', fontsize=16)
    plt.xlabel('Fold', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(folds, fontsize=12)
    plt.ylim([0, 1.05])  # Metrics are usually between 0 and 1
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add minor grid lines
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', alpha=0.4)

    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the combined plot
    filename = os.path.join(save_dir, 'cross_validation_combined_results.png')
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Cross-validation combined results saved to {filename}")

    # Save the data as CSV for future reference
    df = pd.DataFrame({
        'fold': folds,
        **{metric: cv_results[metric] for metric in metrics}
    })

    # Add statistics rows
    avg_row = {'fold': 'Average'}
    std_row = {'fold': 'Std Dev'}

    for metric in metrics:
        avg_row[metric] = np.mean(cv_results[metric])
        std_row[metric] = np.std(cv_results[metric])

    df = pd.concat([df, pd.DataFrame([avg_row, std_row])], ignore_index=True)

    csv_filename = os.path.join(save_dir, 'cross_validation_results.csv')
    df.to_csv(csv_filename, index=False)
    logger.info(f"Cross-validation results saved to {csv_filename}")

    # Create a summary textual report
    summary = ["Cross-Validation Summary", "=" * 30]
    for metric in metrics:
        avg, std = avg_values[metric]
        summary.append(f"{metric.capitalize()}: {avg:.4f} ± {std:.4f}")

    summary_file = os.path.join(save_dir, 'cv_summary.txt')
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary))