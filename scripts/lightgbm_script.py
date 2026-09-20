import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
import optuna
import joblib
import copy
import json
import sys

import os
import logging
from datetime import datetime
import argparse
from functools import partial

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

MAX_POSSIBLE_LAG = 7 
ROLL_WINDOW_MAPPING = {
    "none": [],
    "mid": [3],
    "full": [7],
    "both": [3, 7]
}
LGB_MAX_DEPTH = 16


def setup_logging(log_dir, study_name, model_type):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{study_name}_model{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return logging.getLogger(__name__)




def read_data(individual = "Store.Code", resid = False, verbose = False):
    # Example unbalanced panel data
    if individual == "Store.Code":
        if not resid:
            dir = "data/panels/store/trimmed_store_panel.csv"
        else:
            dir = "data/panels/store/residuals_store_panel.csv"
    elif individual == "Product.Name":
        if not resid:
            dir = "data/panels/drink/combined_drink_panel.csv"
        else:
            dir = "data/panels/drink/residuals_drink_panel.csv"
    df = pd.read_csv(dir, index_col = False)
    # Sort by entity and time   

    df = df.sort_values([individual, 'Sales.Date']).reset_index(drop=True)

    df["Sales.Date"] = pd.to_datetime(
        df["Sales.Date"],
        format="%Y-%m-%d",
        errors="raise"
    )    
    if verbose:
        print(df.head())
        print(df.shape)
    return df
        
        
# Create lag features
def create_panel_lags(df, group_col, time_col, value_cols, lags=[1,2,3]):
    df = df.sort_values([group_col, time_col])
    for col in value_cols:
        for lag in lags:
            df[f'{col}.lag{lag}'] = df.groupby(group_col)[col].shift(lag)
    return df

# Create rolling features
def create_panel_rolling(df, group_col, value_cols, windows=[3,7]):
    
    if isinstance(windows, int):
        windows = [windows]
    for col in value_cols:
        for window in windows:
            df[f'{col}.roll_mean_{window}'] = (
                df.groupby(group_col)[col]
                .shift(1)
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    return df

        
def prepare_data(rev_lags = [1,2,3,4], disc_lags = [1,2,3,4], rev_roll_windows = [3,7], disc_roll_windows = [3,7], model_type = 0, scale = False, resid_data = False, use_panel_id = False, drop_na = True):
    if model_type == 0:
        individual = "Store.Code"
    elif model_type == 1:
        individual = f"Product.Name"
    df = read_data(individual, resid = resid_data)
    if scale:
        for indiv_value in df[individual].unique():
            mask = df[individual] == indiv_value
            
            indiv_rev = df.loc[mask, "Revenue"]
            df.loc[mask, "Revenue"] = (indiv_rev - indiv_rev.mean()) / indiv_rev.std(ddof=1)

            indiv_disc = df.loc[mask, "Discount"]
            df.loc[mask, "Discount"] = (indiv_disc - indiv_disc.mean()) / indiv_disc.std(ddof=1)
    if len(rev_lags) != 0:
        df = create_panel_lags(df, group_col = individual, time_col = 'Sales.Date', value_cols = ['Revenue'], lags=rev_lags)
    if len(disc_lags) > 1:
        df = create_panel_lags(df, group_col = individual, time_col = 'Sales.Date', value_cols = ['Discount'], lags=disc_lags[1:])
    if len(rev_roll_windows):
        df = create_panel_rolling(df, group_col = individual, value_cols = ['Revenue'], windows=rev_roll_windows)
    if len(disc_roll_windows):
        df = create_panel_rolling(df, group_col = individual, value_cols = ['Discount'], windows=disc_roll_windows)
    # df['day_of_week'] = df['Sales.Date'].dt.dayofweek
    
    # Aggregate stats
    # store_stats = df.groupby('Store.Code').agg({
    #     'Revenue': 'mean',
    #     'Discount': 'mean'
    # }).rename(columns={'Revenue': 'Revenue_mean', 'Discount': 'Discount_mean'})
    # df = df.merge(store_stats, on='Store.Code', how='left')
    forbidden_list =  ['Sales.Date', 'Revenue', 'City']
    if not use_panel_id:
        forbidden_list.append("Product.Name")
        forbidden_list.append("Store.Code")
    else:
        if model_type == 0:
            df['Store.Code'] = df['Store.Code'].astype("category")
            
        elif model_type == 1:
            df['Product.Name'] = df['Product.Name'].astype("category")
        
        
    feature_cols = [col for col in df.columns if col not in forbidden_list]

    if model_type == 0:
        # Drop rows with store codes 31, 38, and 39
        store_codes_to_drop = [31, 38, 39]
        df = df[~df['Store.Code'].isin(store_codes_to_drop)]
        
        
    if drop_na:
        df_clean = df.dropna()
    else:
        df_clean = df
    return df_clean, feature_cols
        

def get_features_by_pattern(feature_cols, revenue_max_lag, discount_max_lag, revenue_roll_wind, discount_roll_wind, use_panel):
    """
    Select features based on max_lag and rolling_window type
    """
    selected_features = []
    
    # Define patterns for different feature types
    rev_lag_patterns = [f"Revenue.lag{i}" for i in range(1, revenue_max_lag + 1)]
    
    disc_lag_patterns = [f"Discount.lag{i}" for i in range(0, discount_max_lag + 1)]
    disc_lag_patterns[0] = "Discount"
    window_patterns = []
    # Rolling window patterns based on selection
    if revenue_roll_wind == "mid":
        window_patterns.append("Revenue.roll_mean_3")
    elif revenue_roll_wind == "full":
        window_patterns.append("Revenue.roll_mean_7")
    elif revenue_roll_wind == "both":
        window_patterns.append("Revenue.roll_mean_3")
        window_patterns.append("Revenue.roll_mean_7")
    elif revenue_roll_wind == "none":
        pass
    
    if discount_roll_wind == "mid":
        window_patterns.append("Discount.roll_mean_3")
    elif discount_roll_wind == "full":
        window_patterns.append("Discount.roll_mean_7")
    elif discount_roll_wind == "both":
        window_patterns.append("Discount.roll_mean_3")
        window_patterns.append("Discount.roll_mean_7")
    elif discount_roll_wind == "none":
        pass
    
    # Select features that match the patterns
    for col in feature_cols:
        # Keep non-rolling, non-lag features (like original Revenue, etc.)
        if any(pattern in col for pattern in ["Store.Code", "Product.Name"]) and use_panel:
            selected_features.append(col)
        # Check Revenue lag features
        elif any(pattern in col for pattern in rev_lag_patterns):
            selected_features.append(col)
        # Check Discount and Discount lag features
        elif any(pattern in col for pattern in disc_lag_patterns):
            selected_features.append(col)
        # Check rolling window features
        elif any(pattern in col for pattern in window_patterns):
            selected_features.append(col)
    
    return selected_features



def objective(trial, full_df, all_features, use_panel_id, loss_fn):
    # model params
    feature_fraction = trial.suggest_float("feature_fraction", 0.5, 1.0)
    max_depth = trial.suggest_int("max_depth", 4, LGB_MAX_DEPTH)
    
    max_possible_leaves = 2 ** (max_depth - 1)
    min_leaves = min(16, max_possible_leaves)  # Adjust lower bound if needed
    num_leaves = trial.suggest_int("num_leaves", min_leaves, max_possible_leaves) 
    
    min_child_samples = trial.suggest_categorical('min_child_samples', [10, 20, 30, 40, 50])
    num_tree = trial.suggest_int('num_tree', 50, 250, log = True)
    
    lambda_l2 = trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True)  # ✓ No comma
    
    # data params
    rev_max_lag = trial.suggest_int("rev_max_lag", 1, 7)
    disc_max_lag = trial.suggest_int("disc_max_lag", 0, 7)
    rev_roll_windows = trial.suggest_categorical("rev_roll_windows" , ["none", "mid", "full", "both"])
    disc_roll_windows = trial.suggest_categorical("disc_roll_windows" , ["none", "mid", "full", "both"])
    
    
    

    selected_features = get_features_by_pattern(
        feature_cols = all_features, 
        revenue_max_lag = rev_max_lag,
        discount_max_lag=disc_max_lag,
        revenue_roll_wind=rev_roll_windows,
        discount_roll_wind=disc_roll_windows,
        use_panel = use_panel_id
    )
    if loss_fn == 0:
        train_obj = 'regression_l2'
        metric = 'rmse'
    elif loss_fn == 1:
        train_obj = 'regression_l1'
        metric = 'mae'
    params = {
        'objective': train_obj,
        'metric': metric,
        'data_sample_strategy': 'goss',
        'device' : 'cpu',
        'learning_rate' : 0.05,
        'feature_fraction_seed' : 42,
        'feature_fraction': feature_fraction,
        'max_depth' : max_depth,
        'num_leaves': num_leaves,
        'min_child_samples' : min_child_samples,
        'num_tree' : num_tree,        
        'lambda_l2' : lambda_l2,
    }

    X = copy.deepcopy(full_df[selected_features])
    y = copy.deepcopy(full_df["Revenue"])
    
    # C or Sub.Cat
    if use_panel_id:
        train_data_revenue = lgb.Dataset(X, label=y, categorical_feature=[selected_features[0]])
    else:
        train_data_revenue = lgb.Dataset(X, label=y)
    model_revenue = lgb.train(
        params = params,
        train_set = train_data_revenue,
    )
    
    preds = model_revenue.predict(X)
    if loss_fn == 0:
        loss = np.sqrt(np.mean((y - preds) ** 2)) # RMSE
    else:
        loss = np.mean(np.abs(y - preds))  # MAE

    
    # sys.exit()
    return loss


def tuning_procedure(data_to_tune = None, features_to_tune = None, study_name = 'LightGBM_tuning', storage = None, n_tune_trials = 100, use_panel_id = False, loss_func = 0):
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing tuning procedure for study: {study_name}")
    logger.info(f"Storage: {storage}")
    logger.info(f"Number of trials: {n_tune_trials}")
    
    study = optuna.create_study(
        study_name = study_name,
        direction='minimize',
        storage = storage,
        load_if_exists=False,
        sampler=optuna.samplers.TPESampler(seed=42)
        )
    
    logger.info("Optuna study created successfully")
    logger.info("Starting optimization...")
    print(data_to_tune.head(10))
    objective_with_data = partial(
        objective, 
        full_df=data_to_tune,
        all_features=features_to_tune,
        use_panel_id=use_panel_id,
        loss_fn=loss_func
    )
    
    
    study.optimize(objective_with_data, 
                   n_trials = n_tune_trials)
    
    logger.info(f"Optimization completed. Best value: {study.best_value}")
    
    lgbm_best_params = copy.deepcopy(study.best_params)
    data_param_names = ["rev_max_lag", "disc_max_lag", "rev_roll_windows", "disc_roll_windows"]
    
    model_param = {}
    data_param = {}
    
    for param_name, value in lgbm_best_params.items():
        if param_name in data_param_names:
            data_param[param_name] = value
        else:
            model_param[param_name] = value
    
    logger.info(f"Best model parameters: {model_param}")
    logger.info(f"Best data parameters: {data_param}")
    
    return model_param, data_param, study

def main():
    parser = argparse.ArgumentParser(description="LightGBM Hyperparameter Tuning")
    parser.add_argument("--study_name", type=str, default="LightGBM_tuning", help="Name of the Optuna study")
    parser.add_argument("--dir", type=str, default=r"optuna", help="Storage URL for Optuna study")
    parser.add_argument("--n_tune_trials", type=int, default=100, help="Number of tuning trials")
    parser.add_argument("--model_type", type=int, default=0, help="To tune a store or drink model. 0 for store model, 1 for drink model")
    parser.add_argument("--residual", type=int, default=0, help = "To tune a model on residuals obtained by a statistical backend. 0 for raw revenue, 1 for residual revenue.")
    parser.add_argument("--transform", type=int, default=0, help = "To scale and translate each panel by sample mean and standard deviation. 0 for no scaling, 1 for scaling.")
    parser.add_argument("--use_panel_id", type=int, default=0, help = "To use Store.Code or Product.Name identifier when fitting the model. 0 for no, 1 for yes.")
    parser.add_argument("--loss", type=int, default=0, help = "To use RMSE or MAE as the loss function in training/tuning. 0 for RMSE, 1 for MAE.")
    args = parser.parse_args()
    
    
    full_df_clean, all_feature_cols = prepare_data(
            rev_lags=list(range(1, MAX_POSSIBLE_LAG + 1)),
            disc_lags=list(range(0, MAX_POSSIBLE_LAG + 1)),
            rev_roll_windows= [3,7],
            disc_roll_windows=[3,7],
            model_type=args.model_type,  # You'll need to pass this appropriately
            scale=bool(args.transform),
            resid_data=bool(args.residual),  # You'll need to pass this appropriately
            use_panel_id=bool(args.use_panel_id),
            drop_na=True
        )
    print(all_feature_cols)
    # C or Sub.Cat
    log_dir = os.path.join(args.dir, args.study_name, "logs")
    logger = setup_logging(log_dir, args.study_name, args.model_type)
    
    # Setup logging

    
    # Log dataframe info
    logger.info("="*50)
    logger.info(f"Full dataframe definition established with shape: {full_df_clean.shape}")
    logger.info(f"Number of features: {len(all_feature_cols)}")
    logger.info(f"Columns: {all_feature_cols[:5]}...")  # Show first 5 features as sample
    logger.info(f"Date range: {full_df_clean['Sales.Date'].min()} to {full_df_clean['Sales.Date'].max()}")

    # Determine individual column based on model_type
    individual_col = "Store.Code" if args.model_type == 0 else "Product.Name"
    logger.info(f"Number of unique {individual_col}: {full_df_clean[individual_col].nunique()}")

    # Save dataframe to disk as CSV
    data_save_path = os.path.join(args.dir, f"{args.study_name}/data", "full_prepared_data.csv")
    os.makedirs(os.path.join(args.dir, f"{args.study_name}/data"), exist_ok=True)
    full_df_clean.to_csv(data_save_path, index=False)
    logger.info(f"Full dataframe saved to: {data_save_path}")

    # Optionally save feature list as well
    features_save_path = os.path.join(args.dir, f"{args.study_name}/data", "feature_cols.txt")
    with open(features_save_path, 'w') as f:
        for feature in all_feature_cols:
            f.write(f"{feature}\n")
    logger.info(f"Feature list saved to: {features_save_path}")
    
    logger.info("="*50)
    logger.info(f"Starting LightGBM Tuning Session")
    logger.info(f"Study Name: {args.study_name}")
    logger.info(f"Model Type: {'Store' if args.model_type == 0 else 'Drink'}")
    logger.info(f"Number of Trials: {args.n_tune_trials}")
    logger.info(f"Base Directory: {args.dir}")
    logger.info("="*50)

    
    db_file_storage = os.path.join(args.dir, args.study_name, "tune_results")
    logger.info(f"Creating directory: {db_file_storage}")
    os.makedirs(db_file_storage, exist_ok=False)
    logger.info(f"Directory created successfully")
    db_path = os.path.join(db_file_storage, f"{args.study_name}_{args.model_type}.db")
    db_file_storage = r"sqlite:///" + db_path
    
    
    logger.info(f"SQLite database will be created at: {db_path}")
    logger.info("Starting hyperparameter tuning procedure...")
    try:
        # =================== TUNING PROCEDURE ===================
        # Add informative logging about the model configuration
        # Detailed configuration logging
        model_config = {
            "model_type": "Store.Code" if args.model_type == 0 else "Product.Name",
            "residuals_training": args.residual,
            "n_trials": args.n_tune_trials,
            "features_count": len(all_feature_cols),
            "data_shape": full_df_clean.shape
        }

        logger.info(f"Starting optimization with configuration:")
        logger.info(f"  - Model type: {model_config['model_type']}")
        logger.info(f"  - Training on residuals: {model_config['residuals_training']}")
        logger.info(f"  - Number of trials: {model_config['n_trials']}")
        logger.info(f"  - Features: {model_config['features_count']}")
        logger.info(f"  - Data samples: {model_config['data_shape'][0]}")
        logger.info(f"  - Loss: {args.loss} (0 : RMSE, 1 : MAE)")

        best_model_params, best_data_params, study = tuning_procedure(
                    full_df_clean, 
                    all_feature_cols,
                    study_name = args.study_name,
                    storage = db_file_storage,
                    n_tune_trials = args.n_tune_trials,
                    use_panel_id=bool(args.use_panel_id),
                    loss_func=args.loss
                                      
                )
        
        logger.info("="*50)
        logger.info("TUNING COMPLETED SUCCESSFULLY")
        logger.info(f"Best Loss {args.loss}: {study.best_value:.4f}")
        logger.info(f"Best trial number: {study.best_trial.number}")
        logger.info("Best parameters found:")
        logger.info(f"  Model parameters: {best_model_params}")
        logger.info(f"  Data parameters: {best_data_params}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Error during tuning procedure: {str(e)}", exc_info=True)
        raise
    logger.info("Preparing final dataset with best parameters...")

    feature_cols = get_features_by_pattern(
        all_feature_cols,
        revenue_max_lag=best_data_params["rev_max_lag"],
        discount_max_lag=best_data_params["disc_max_lag"],
        revenue_roll_wind=ROLL_WINDOW_MAPPING[best_data_params["rev_roll_windows"]],
        discount_roll_wind = ROLL_WINDOW_MAPPING[best_data_params["disc_roll_windows"]],
        use_panel=bool(args.use_panel_id)
    )
    df_clean = copy.deepcopy(full_df_clean[feature_cols])
    df_clean["Revenue"] = full_df_clean["Revenue"]

    ftrs_and_rev = copy.deepcopy(feature_cols)
    ftrs_and_rev.append("Revenue")
    logger.info(f"Features: {ftrs_and_rev}")
    print(best_model_params, best_data_params)    
    logger.info(f"Saving the data used by model:")
    os.makedirs(os.path.join(args.dir, args.study_name, "model_data"))
    df_clean[ftrs_and_rev].to_csv(os.path.join(args.dir, args.study_name, "model_data/best_data.csv"), index=False)
    param_dir = os.path.join(args.dir, args.study_name, "params")
    logger.info(f"Creating parameters directory: {param_dir}")
    os.makedirs(param_dir, exist_ok = True)
    
    model_param_dir = os.path.join(param_dir, f"best_model_params_{args.model_type}.json")
    data_param_dir = os.path.join(param_dir, f"best_data_params_{args.model_type}.json")
    with open(model_param_dir, "w") as f:
        json.dump(best_model_params, f, indent=4)
        logger.info(f"Model parameters saved to: {model_param_dir}")
        
    with open(data_param_dir, "w") as f:
        json.dump(best_data_params, f, indent=4)
        logger.info(f"Data parameters saved to: {data_param_dir}")
    
    logger.info("Training final model...")
    if bool(args.use_panel_id):
        train_data = lgb.Dataset(df_clean[feature_cols], label=df_clean['Revenue'], categorical_feature=['Store.Code'] if args.model_type == 0 else 1)
    else:
        train_data = lgb.Dataset(df_clean[feature_cols], label=df_clean['Revenue'])
    logger.info(f"With the features {feature_cols}")
    # Your base params structure
    params = {
        'objective': 'regression_l2' if args.loss == 0 else 'regression_l1',  # This will come from best_model_params
        'metric': 'regression_l2' if args.loss == 0 else 'regression_l1',  # Keep fixed or could come from best_model_params
        'data_sample_strategy': 'goss',  # Fixed
        'device': 'cpu',  # Fixed
        'learning_rate': 0.05,  # Could be from tuning or fixed
        'feature_fraction_seed': 42,  # Fixed
    }

    # Add all the relevant params from best_model_params
    # Option 1: Update with all best_model_params (overwrites duplicates)
    params.update(best_model_params)


    final_model = lgb.train(params, train_data) # num_boost_round is the same as num_iterations
    print(df_clean.shape)
    logger.info("Final model training completed")
    
    logger.info("Generating feature importance plot...")
    lgb.plot_importance(final_model, max_num_features=15)
    plot_path = os.path.join(args.dir, args.study_name, f"feature_importance_model{args.model_type}.png")
    plt.savefig(plot_path)
    logger.info(f"Feature importance plot saved to: {plot_path}")
    plt.show()

    logger.info("="*50)
    logger.info(f"STUDY {args.study_name} COMPLETED")
    logger.info(f"Log file saved in: {log_dir}")
    logger.info("="*50)
    
    # joblib.dump(study, f"optuna/{args.study_name}.pkl")
    study_path = os.path.join(args.dir, args.study_name, f"{args.study_name}.pkl")
    joblib.dump(study, study_path)
    logger.info(f"Study saved to: {study_path}")
    
    return study, final_model


if __name__ == "__main__":
    study, final_model = main()
    
    