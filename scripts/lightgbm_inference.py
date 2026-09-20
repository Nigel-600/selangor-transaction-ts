from lightgbm_script import *

import numpy as np
import pandas as pd
import lightgbm as lgb
import json
import os
import argparse
import logging
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Import helper functions from the original script
from lightgbm_script import *

def setup_logging(log_dir, study_name, model_type):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"inference_{study_name}_model{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_params(dir, study_name, model_type):
    """Load model and data parameters from JSON files"""
    model_param_path = os.path.join(dir, study_name, "params", f"best_model_params_{model_type}.json")
    data_param_path = os.path.join(dir, study_name, "params", f"best_data_params_{model_type}.json")
    
    with open(model_param_path, 'r') as f:
        model_params = json.load(f)
    
    with open(data_param_path, 'r') as f:
        data_params = json.load(f)
    
    return model_params, data_params

def run_inference(model_params, data_params, model_type, resid, use_panel, scaling=False, output_dir=None):
    """Run inference using loaded parameters and generate plots"""
    logger = logging.getLogger(__name__)
    
    # Determine individual based on model type
    if model_type == 0:
        individual = "Store.Code"
        entity_name = "Store"
        logger.info("Running inference for Store model")
    elif model_type == 1:
        individual = "Product.Name"
        entity_name = "Product"
        logger.info("Running inference for Product model")
    
    # Prepare data with the same parameters used in training
    logger.info(f"Preparing data with parameters: {data_params}")
    logger.info(f"Revenue max lag: {data_params['rev_max_lag']}")
    logger.info(f"Revenue max lag: {data_params['disc_max_lag']}")
    logger.info(f"Rolling window: {data_params['rev_roll_windows']} -> {ROLL_WINDOW_MAPPING[data_params['rev_roll_windows']]}")
    logger.info(f"Rolling window: {data_params['disc_roll_windows']} -> {ROLL_WINDOW_MAPPING[data_params['disc_roll_windows']]}")

    df_clean, feature_cols = prepare_data(
        rev_lags = list(range(1, data_params["rev_max_lag"] + 1)),
        disc_lags = list(range(0, data_params["disc_max_lag"] + 1)),
        rev_roll_windows = ROLL_WINDOW_MAPPING[data_params["rev_roll_windows"]],
        disc_roll_windows = ROLL_WINDOW_MAPPING[data_params["disc_roll_windows"]],
        model_type = model_type,
        scale = bool(scaling),
        resid_data = resid,
        use_panel_id=use_panel,
        drop_na = True
    )
    print(feature_cols)
    logger.info(f"Data prepared successfully")
    logger.info(f"Dataset shape: {df_clean.shape}")
    logger.info(f"Number of features: {len(feature_cols)}")
    logger.info(f"Features: {feature_cols[:5]}... (showing first 5)")
    
    # Train model with loaded parameters
    logger.info("Training model with loaded parameters...")
    if use_panel:
        train_data = lgb.Dataset(df_clean[feature_cols], label=df_clean['Revenue'], categorical_feature=individual)
    else:
        train_data = lgb.Dataset(df_clean[feature_cols], label=df_clean['Revenue'])
        
    params = {
        'objective': 'regression_l2',  # This will come from best_model_params
        'metric': 'regression_l2',  # Keep fixed or could come from best_model_params
        'data_sample_strategy': 'goss',  # Fixed
        'device': 'cpu',  # Fixed
        'learning_rate': 0.10,  # Could be from tuning or fixed
        'feature_fraction_seed': 42,  # Fixed
    }

    # Add all the relevant params from best_model_params
    # Option 1: Update with all best_model_params (overwrites duplicates)
    params.update(model_params)
    # Ensure numeric parameters are the correct type
    for key, value in params.items():
        if key in ['max_depth', 'num_leaves', 'min_child_samples', 'num_tree']:
            params[key] = int(value)
        elif key in ['feature_fraction', 'learning_rate']:
            params[key] = float(value)
    print(params)
    final_model = lgb.train(params, train_data)
    logger.info("Model training completed")
    
    # Make predictions
    logger.info("Making predictions...")
    print(df_clean[feature_cols].columns)
    predictions = final_model.predict(df_clean[feature_cols])
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((df_clean['Revenue'] - predictions) ** 2))
    mae = np.mean(np.abs(df_clean['Revenue'] - predictions))
    r2 = 1 - np.sum((df_clean['Revenue'] - predictions) ** 2) / np.sum((df_clean['Revenue'] - np.mean(df_clean['Revenue'])) ** 2)
    
    logger.info("="*50)
    logger.info("INFERENCE RESULTS")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info("="*50)
    
    # Add predictions to dataframe
    df_results = df_clean.copy()
    df_results['Predicted_Revenue'] = predictions
    df_results['Residuals'] = df_results['Revenue'] - predictions
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions
        predictions_path = os.path.join(output_dir, f"predictions_model{model_type}.csv")
        df_results.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to: {predictions_path}")
        
        # Save metrics
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'model_type': model_type,
            'individual': individual,
            'n_samples': len(df_results),
            'n_features': len(feature_cols)
        }
        
        metrics_path = os.path.join(output_dir, f"metrics_model{model_type}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4, cls=NumpyEncoder)
        logger.info(f"Metrics saved to: {metrics_path}")
        
        # Save feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': final_model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        importance_path = os.path.join(output_dir, f"feature_importance_model{model_type}.csv")
        importance.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to: {importance_path}")
        
        # ========== PLOTTING SECTION ==========
        logger.info("Generating entity-level plots...")
        

        
        # Get unique entities
        unique_entities = df_results[individual].unique()
        
        # Create PDF for actual vs predicted
        pdf_predictions_path = os.path.join(output_dir, f"{entity_name.lower()}_predictions_model{model_type}.pdf")
        with PdfPages(pdf_predictions_path) as pdf:
            for entity in unique_entities:
                # Index using the individual column directly
                entity_data = df_results[df_results[individual] == entity].sort_values('Sales.Date')
                
                # Skip if no data
                if len(entity_data) == 0:
                    continue
                
                # Create figure
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                fig.suptitle(f'{entity_name}: {entity}', fontsize=14, fontweight='bold')
                
                # Plot 1: Actual vs Predicted
                ax1 = axes[0]
                ax1.plot(entity_data['Sales.Date'], entity_data['Revenue'], 
                        'b-', label='Actual Revenue', linewidth=2, alpha=0.7)
                ax1.plot(entity_data['Sales.Date'], entity_data['Predicted_Revenue'], 
                        'r--', label='Predicted Revenue', linewidth=2, alpha=0.7)
                ax1.set_ylabel('Revenue')
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)
                ax1.set_title('Actual vs Predicted Revenue Over Time')
                
                # Rotate x-axis labels for better readability
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Plot 2: Residuals
                ax2 = axes[1]
                ax2.bar(entity_data['Sales.Date'], entity_data['Residuals'], 
                       color='gray', alpha=0.6, width=0.8)
                ax2.axhline(y=0, color='r', linestyle='-', linewidth=1)
                ax2.set_ylabel('Residuals (Actual - Predicted)')
                ax2.set_xlabel('Date')
                ax2.grid(True, alpha=0.3)
                ax2.set_title('Prediction Residuals Over Time')
                
                # Rotate x-axis labels
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Add statistics box
                entity_rmse = np.sqrt(np.mean(entity_data['Residuals'] ** 2))
                entity_mae = np.mean(np.abs(entity_data['Residuals']))
                entity_r2 = 1 - np.sum(entity_data['Residuals'] ** 2) / np.sum((entity_data['Revenue'] - np.mean(entity_data['Revenue'])) ** 2) if len(entity_data) > 1 else np.nan
                
                stats_text = f'RMSE: {entity_rmse:.2f}\nMAE: {entity_mae:.2f}\nR²: {entity_r2:.3f}'
                ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        
        logger.info(f"✅ Saved predictions plot to: {pdf_predictions_path}")
        
        # Create separate PDF for residuals only
        pdf_residuals_path = os.path.join(output_dir, f"{entity_name.lower()}_residuals_model{model_type}.pdf")
        with PdfPages(pdf_residuals_path) as pdf:
            for entity in unique_entities:
                # Index using the individual column directly
                entity_data = df_results[df_results[individual] == entity].sort_values('Sales.Date')
                
                # Skip if no data
                if len(entity_data) == 0:
                    continue
                
                # Create figure for residuals only
                fig, ax = plt.subplots(figsize=(12, 6))
                fig.suptitle(f'{entity_name}: {entity} - Prediction Residuals', fontsize=14, fontweight='bold')
                
                # Plot residuals
                ax.bar(entity_data['Sales.Date'], entity_data['Residuals'], 
                       color='gray', alpha=0.7, width=0.8)
                ax.axhline(y=0, color='r', linestyle='-', linewidth=2)
                ax.axhline(y=np.std(entity_data['Residuals']), color='orange', 
                          linestyle='--', linewidth=1, alpha=0.7, label='+1 Std')
                ax.axhline(y=-np.std(entity_data['Residuals']), color='orange', 
                          linestyle='--', linewidth=1, alpha=0.7, label='-1 Std')
                
                ax.set_ylabel('Residuals')
                ax.set_xlabel('Date')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
                
                # Rotate x-axis labels
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Add statistics
                stats_text = f'Residual Std: {np.std(entity_data["Residuals"]):.2f}\n'
                stats_text += f'Max Underprediction: {entity_data["Residuals"].min():.2f}\n'
                stats_text += f'Max Overprediction: {entity_data["Residuals"].max():.2f}'
                
                ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        
        logger.info(f"✅ Saved residuals plot to: {pdf_residuals_path}")
        
        # Create entity-level metrics summary
        logger.info("Generating entity-level summary statistics...")
        
        # Calculate entity-level metrics
        entity_metrics = []
        for entity in unique_entities:
            entity_data = df_results[df_results[individual] == entity]
            if len(entity_data) > 0:
                entity_rmse = np.sqrt(np.mean(entity_data['Residuals'] ** 2))
                entity_mae = np.mean(np.abs(entity_data['Residuals']))
                entity_r2 = 1 - np.sum(entity_data['Residuals'] ** 2) / np.sum((entity_data['Revenue'] - np.mean(entity_data['Revenue'])) ** 2) if len(entity_data) > 1 else np.nan
                
                metrics_row = {
                    'entity': entity,
                    'rmse': entity_rmse,
                    'mae': entity_mae,
                    'r2': entity_r2,
                    'mean_revenue': np.mean(entity_data['Revenue']),
                    'std_revenue': np.std(entity_data['Revenue']),
                    'n_observations': len(entity_data)
                }
                entity_metrics.append(metrics_row)
        
        entity_metrics_df = pd.DataFrame(entity_metrics)
        
        # Save entity metrics to CSV
        entity_metrics_path = os.path.join(output_dir, f"{entity_name.lower()}_entity_metrics_model{model_type}.csv")
        entity_metrics_df.to_csv(entity_metrics_path, index=False)
        logger.info(f"✅ Saved entity metrics to: {entity_metrics_path}")
        
        # Create summary PDF
        pdf_summary_path = os.path.join(output_dir, f"{entity_name.lower()}_summary_stats_model{model_type}.pdf")
        with PdfPages(pdf_summary_path) as pdf:
            if len(entity_metrics_df) > 0:
                # Figure 1: RMSE distribution
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'{entity_name}-Level Prediction Summary Statistics', fontsize=16, fontweight='bold')
                
                # RMSE histogram
                ax1 = axes[0, 0]
                ax1.hist(entity_metrics_df['rmse'], bins=20, edgecolor='black', alpha=0.7)
                ax1.set_xlabel('RMSE')
                ax1.set_ylabel('Number of Entities')
                ax1.set_title('Distribution of  Across Entities')
                ax1.grid(True, alpha=0.3)
                
                # MAE histogram
                ax2 = axes[0, 1]
                ax2.hist(entity_metrics_df['mae'], bins=20, edgecolor='black', alpha=0.7, color='green')
                ax2.set_xlabel('MAE')
                ax2.set_ylabel('Number of Entities')
                ax2.set_title('Distribution of MAE Across Entities')
                ax2.grid(True, alpha=0.3)
                
                # R² histogram
                ax3 = axes[1, 0]
                r2_valid = entity_metrics_df['r2'].dropna()
                if len(r2_valid) > 0:
                    ax3.hist(r2_valid, bins=20, edgecolor='black', alpha=0.7, color='orange')
                ax3.set_xlabel('R²')
                ax3.set_ylabel('Number of Entities')
                ax3.set_title('Distribution of R² Across Entities')
                ax3.grid(True, alpha=0.3)
                
                # RMSE vs Mean Revenue scatter
                ax4 = axes[1, 1]
                ax4.scatter(entity_metrics_df['mean_revenue'], entity_metrics_df['rmse'], alpha=0.6)
                ax4.set_xlabel('Mean Revenue')
                ax4.set_ylabel('RMSE')
                ax4.set_title('RMSE vs Mean Revenue by Entity')
                ax4.grid(True, alpha=0.3)
                
                # Add trend line if enough points
                if len(entity_metrics_df) > 1:
                    z = np.polyfit(entity_metrics_df['mean_revenue'], entity_metrics_df['rmse'], 1)
                    p = np.poly1d(z)
                    ax4.plot(entity_metrics_df['mean_revenue'].sort_values(), 
                            p(entity_metrics_df['mean_revenue'].sort_values()), 
                            "r--", alpha=0.8, label='Trend line')
                    ax4.legend()
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        
        logger.info(f"✅ Saved summary statistics to: {pdf_summary_path}")
        # ========== END PLOTTING SECTION ==========
    
    return final_model, df_results, metrics
def main():
    parser = argparse.ArgumentParser(description="LightGBM Inference Script")
    parser.add_argument("--dir", type = str, required = True,
                        help = "Directory of Optuna study")
    parser.add_argument("--study_name", type=str, required=True, 
                       help="Name of optuna study")
    parser.add_argument("--model_type", type=int, required=True, choices=[0, 1],
                       help="Model type: 0 for Store model, 1 for Product model")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory to save log files")
    parser.add_argument("--residual", type=int, default=0, help = "To tune a model on residuals obtained by a statistical backend. 0 for raw revenue, 1 for residual revenue.")
    parser.add_argument("--transform", type=int, default=0, 
                        help = "To scale and translate each panel by sample mean and standard deviation. 0 for no scaling, 1 for scaling.")
    parser.add_argument("--use_panel_id", type=int, default=0, help = "To use Store.Code or Product.Name identifier when fitting the model. 0 for no, 1 for yes.")
    args = parser.parse_args()
    
    
    
    args = parser.parse_args()
    log_dir = os.path.join(args.dir, args.study_name, "logs")
    param_dir = os.path.join(args.dir, args.study_name, "params")
    output_dir = os.path.join(args.dir, args.study_name, "inference_results")
    # Validate model_type
    if args.model_type not in [0, 1]:
        raise ValueError("model_type must be either 0 or 1")
    
    # Setup logging
    logger = setup_logging(log_dir, args.study_name, args.model_type)
    
    logger.info("="*50)
    logger.info("STARTING INFERENCE")
    logger.info(f"Parameter directory: {param_dir}")
    logger.info(f"Model type: {'Store' if args.model_type == 0 else 'Product'}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*50)
    
    try:
        # Load parameters
        logger.info("Loading parameters...")
        model_params, data_params = load_params(args.dir, args.study_name, args.model_type)
        logger.info(f"Model parameters loaded: {model_params}")
        logger.info(f"Data parameters loaded: {data_params}")
        
        # Run inference
        
        model, results, metrics = run_inference(
            model_params=model_params,
            data_params=data_params,
            model_type=args.model_type,
            scaling=args.transform,
            resid=args.residual,
            use_panel=bool(args.use_panel_id),
            output_dir=output_dir
        )
        
        logger.info("="*50)
        logger.info("INFERENCE COMPLETED SUCCESSFULLY")
        logger.info(f"Log file saved in: {args.log_dir}")
        logger.info("="*50)
        
    except FileNotFoundError as e:
        logger.error(f"Parameter files not found: {e}")
        logger.error(f"Expected files: best_model_params_{args.model_type}.json and best_data_params_{args.model_type}.json in {os.path.join(args.dir, args.study_name, 'params')}")   
        raise
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()