import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from joblib import dump, load
import os


#=====================================================
# GLOBALS FOR CONFIGURATION
#=====================================================
# Seed for Random State
SEED = 7
np.random.seed(SEED)

# Fixed CV splitter for reproducibility and fair RMSE comparisons
CV_SPLITTER = KFold(n_splits=10, shuffle=True, random_state=SEED)

# Switch for Overall Model Ceation vs. Prediction Evaluation
create_model_pieces_mode = False

#=====================================================
# CURRENTLY: RE-RUNNING MODEL BY MODEL WITH SEED AS A FEATURE
#=====================================================

# Model Creation and Testing Globals
set_model = 8

# These variables are now set in the routine
# retrain_same_params = True         # Extract best params from existing model .joblib and reuse
# run_feature_evaluation = True      # Routine to check effect of individual feature addition or removal on RMSE


# Model numbers for set_model:
#   1: TARGET POSS - GBR
#   2: TARGET FGA - GBR
#   3: TARGET FG% - GBR
#   4: TARGET 3PTA - GBR
#   5: TARGET 3PT% - GBR
#   6: TARGET FTA - GBR
#   7: TARGET FT% - GBR
#   8: TARGET 2PT% - GBR

#=====================================================
# MODEL DEPENDENCY METADATA
#=====================================================
MODEL_DEPENDENCIES = {
    'TARGET FGA': ['PRED POSS'],
    'TARGET FG%': ['PRED POSS', 'PRED FGA'],
    'TARGET 3PTA': ['PRED POSS', 'PRED FGA'],
    'TARGET 3PT%': ['PRED POSS', 'PRED FGA'],
    'TARGET FTA': ['PRED POSS', 'PRED FGA'],
    'TARGET FT%': ['PRED POSS', 'PRED FGA'],
    'TARGET 2PT%': ['PRED POSS', 'PRED FGA']
}

#=====================================================
# SIMPLE HELPER FUNCTIONS
#=====================================================
def get_filename_for_target(target_variable):
    target_variable_for_file = target_variable.lower().replace(' ', '_')
    file_string = str('model_*.joblib').replace('*', target_variable_for_file)
    if '%' in file_string:
        file_string = file_string.replace('%', 'pct')
    return file_string

def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))

def inv_logit(x):
    return 1 / (1 + np.exp(-x))

def generate_preds(model, target_variable, feature_list, combined_df):
    X = combined_df.loc[:, feature_list].values
    predicted_y = model.predict(X)

    if '%' in target_variable:

        predicted_y = inv_logit(predicted_y) * 100 

    combined_df[target_variable] = predicted_y
    return combined_df

def print_model_info(model, feature_list, rmse):
    print('RMSE:')
    print(np.round(rmse, 4))
    print()

    if isinstance(model, GradientBoostingRegressor):
        print('Feature importances:')
        sorted_feats = sorted(
            zip(feature_list, model.feature_importances_),
            key=lambda x: -x[1]
        )

    elif isinstance(model, Ridge):
        print('Coefficient magnitudes:')
        sorted_feats = sorted(
            zip(feature_list, np.abs(model.coef_)),
            key=lambda x: -x[1]
        )

    else:
        return

    for i in range(0, len(sorted_feats), 4):
        row = sorted_feats[i:i+4]
        print(" | ".join(f"{f:<20}: {v:>7.4f}" for f, v in row))

def create_target_variable_in_df(combined, target_variable):
    if target_variable in combined.columns:
        
        return combined

    if target_variable == 'TARGET FGA':
        combined['TARGET FGA'] = (
            (combined['PPG'] - combined['FTPG'] - 3 * combined['3PPG']) / 2.0
            + combined['3PPG']
        ) / combined['FG%']

    elif target_variable == 'TARGET 2PT%':
        combined['TARGET 2PT%'] = combined['2PT MADE'] / combined['2PT ATT']

    elif target_variable[7:] in combined.columns:
        combined[target_variable] = combined[target_variable[7:]]

    else:
        print('[WARN] No handling specified for creating target variable column in DataFrame.')

    return combined

def add_preds_to_combined(combined, pred_variable, visited=None):
    if visited is None:
        visited = set()

    if pred_variable in visited:
        raise RuntimeError(f'[ERROR] Circular dependency detected for {pred_variable}')

    visited.add(pred_variable)

    target_name = pred_variable.replace('PRED', 'TARGET')
    filename = get_filename_for_target(target_name)

    model_dict = load(filename)
    model = model_dict['model']
    feature_list = model_dict['feature_list']

    for feature in feature_list:
        if feature in combined.columns:
            continue
        if 'PRED' in feature:
            print(f'Creating column for dependency {feature}')
            combined = add_preds_to_combined(combined, feature, visited)

    combined = generate_preds(model, pred_variable, feature_list, combined)
    return combined

def backfill_2pt_stats(df):
    fg_pct = df['FG%']
    if fg_pct.max() > 1:
        fg_pct = fg_pct / 100

    missing_2pt_made = df['2PT MADE'].isna()
    missing_2pt_att = df['2PT ATT'].isna()

    df.loc[missing_2pt_made, '2PT MADE'] = (
        df.loc[missing_2pt_made, 'PPG']
        - df.loc[missing_2pt_made, 'FTPG']
        - 3 * df.loc[missing_2pt_made, '3PPG']
    ) / 2

    fg_made = df['2PT MADE'] + df['3PPG']
    fg_att = fg_made / fg_pct.replace(0, np.nan)

    df.loc[missing_2pt_att, '2PT ATT'] = (
        fg_att.loc[missing_2pt_att]
        - df.loc[missing_2pt_att, '3PTA']
    )

    df['2PT MADE'] = df['2PT MADE'].clip(lower=0)
    df['2PT ATT'] = df['2PT ATT'].clip(lower=0)

    return df

#=====================================================
# ADJUST CROSS VALIDATION PARAMS HERE
#=====================================================
def create_train_model(model, training_df, feature_list, target_variable, freeze_params=False):
    """
    Train a model (GBR or Ridge) with GridSearchCV, handle '%' targets via logit transformation,
    and calculate RMSE in the original units of the target variable.
    """
    # Extract features and target
    X = training_df.loc[:, feature_list].values
    y_original = training_df[target_variable].values  # Keep original for RMSE

    # Transform target if it's a percentage
    if '%' in target_variable:
        y_scaled = y_original / 100           # 0-1 scale
        y = logit(y_scaled)                   # logit transformation
    else:
        y = y_original.copy()

    # Set up hyperparameter grid
    if isinstance(model, GradientBoostingRegressor):
        params = {
            'learning_rate': [0.01, 0.02],
            'n_estimators': [200, 250, 300, 350],
            'max_depth': [2, 3, 4]
        }
    elif isinstance(model, Ridge):
        params = {
            'alpha': [0.005, 0.01, 0.05, 0.1, 0.5]
        }
    else:
        raise ValueError(f'Unsupported model type: {type(model)}')

    # Optionally reuse previous best parameters
    if retrain_same_params:
        filename = get_filename_for_target(target_variable)
        if os.path.isfile(filename):
            model_dict = load(filename)
            params = {k: [v] for k, v in model_dict['params'].items()}

    # Freeze parameters if requested
    if freeze_params:
        model.set_params(**{k: v[0] for k, v in params.items()})
        model.fit(X, y)

        oof_preds = cross_val_predict(model, X, y, cv=CV_SPLITTER)

        # Convert predictions back to original scale for RMSE
        if '%' in target_variable:
            y_pred_original = inv_logit(oof_preds) * 100
            rmse = np.sqrt(np.mean((y_pred_original - y_original) ** 2))
            rmse_logit = np.sqrt(np.mean((oof_preds - y) ** 2))
        else:
            y_pred_original = oof_preds
            rmse = np.sqrt(np.mean((y_pred_original - y_original) ** 2))
            rmse_logit = rmse

        # Store predictions in DataFrame in model output scale
        pred_col = target_variable.replace('TARGET', 'PRED')

        if '%' in target_variable:
            training_df[pred_col] = inv_logit(oof_preds) * 100
            training_df[pred_col + '_LOGIT'] = oof_preds
        else:
            training_df[pred_col] = oof_preds

        return model, rmse, training_df, model.get_params(), rmse_logit

    # Grid search for best hyperparameters
    cv = GridSearchCV(
        model,
        params,
        cv=CV_SPLITTER,
        scoring='neg_mean_squared_error',
        verbose=False
    )

    cv.fit(X, y)
    best_model = cv.best_estimator_

    # Out-of-fold predictions
    oof_preds = cross_val_predict(best_model, X, y, cv=CV_SPLITTER)

    # Convert predictions back to original scale for RMSE
    if '%' in target_variable:
        y_pred_original = inv_logit(oof_preds) * 100
        rmse = np.sqrt(np.mean((y_pred_original - y_original) ** 2))
        rmse_logit = np.sqrt(np.mean((oof_preds - y) ** 2))
    else:
        y_pred_original = oof_preds
        rmse = np.sqrt(np.mean((y_pred_original - y_original) ** 2))
        rmse_logit = rmse

    # Store predictions in DataFrame in model output scale (logit or raw)
    pred_col = target_variable.replace('TARGET', 'PRED')

    if '%' in target_variable:
        training_df[pred_col] = inv_logit(oof_preds) * 100
        training_df[pred_col + '_LOGIT'] = oof_preds
    else:
        training_df[pred_col] = oof_preds

    print("Best params:", cv.best_params_)

    return best_model, rmse, training_df, cv.best_params_, rmse_logit

def create_train_gbr_model(training_df, feature_list, target_variable, freeze_params=False):
    model = GradientBoostingRegressor(random_state=SEED)
    best_model, rmse, training_df, best_params, rmse_logit = create_train_model(model, training_df, feature_list, target_variable, freeze_params)
    
    return best_model, rmse, training_df, best_params, rmse_logit

def create_train_ridge_model(training_df, feature_list, target_variable, freeze_params=False):
    model = Ridge(random_state=SEED)
    best_model, rmse, training_df, best_params, rmse_logit = create_train_model(model, training_df, feature_list, target_variable, freeze_params)
    
    return best_model, rmse, training_df, best_params, rmse_logit

def prune_zero_importance_features(model, feature_list, threshold=1e-4):
    if not hasattr(model, 'feature_importances_'):
        return feature_list

    zero_importance_features = [f for f, imp in zip(feature_list, model.feature_importances_)
        if imp <= threshold]
    if zero_importance_features:
        print('Zero Importance Features')
        print(zero_importance_features)

    return [
        f for f, imp in zip(feature_list, model.feature_importances_)
        if imp > threshold
    ]

#=====================================================
# FEATURE LIST LOGIC
#=====================================================
def get_model_feature_list(target_variable, combined_df):
    feature_list = []

    if target_variable == "TARGET POSS":
        feature_list.extend(['Seed', 'T1_TEMPO', 'T2_TEMPO', 'T1_FGA', 'T2_FGA', 'T1_ORPG', 'T2_ORPG', 'T1_FTA', 'T2_FTA', 'T1_OPPG', 'T2_OPPG', 'T1_TOF', 'T2_TOF'])

    elif target_variable == 'TARGET FGA':
        feature_list.append('PRED POSS')

        feature_list.extend(['Seed', 'T1_APG', 'T1_FGA', 'T2_TEMPO', 'T1_ORPG', 'T2_DRPG', 'T2_OPPG', 'T2_FPG', 'T1_TPG', 'T2_TOF', 'T2_ORPG', 'T1_FTA', 'T2_TM', 'T2_RM'])

    elif target_variable == 'TARGET FG%':
        feature_list.extend(['PRED POSS', 'PRED FGA'])

        feature_list.extend(['Seed', 'T1_FG%', 'T2_3PD%', 'T2_BPG', 'T1_FB', 'T1_Bench', 'T2_RM', 'T1_SM', 'T1_3PTA', 'T1_3PPG', 'T1_APG', 'T2_SPG', 'T2_ORPG', 'T2_FGA'])

    elif target_variable == 'TARGET 3PTA':
        feature_list.extend(['PRED FGA'])

        feature_list.extend(['Seed', 'T1_3PTA', 'T2_3PD%', 'T1_3PPG', 'T2_TPG', 'T2_APG', 'T1_ORPG', 'T2_OPPG', 'T1_3PD%', 'T2_PPG', 'T2_FTA', 'T1_FTA', 'T2_TEMPO', 'T2_SPG', 'T2_FTPG', 'T2_DFG%'])
        
    elif target_variable == 'TARGET 3PT%':
        feature_list.extend(['PRED FGA'])

        feature_list.extend(['Seed', 'T1_3PT%', 'T2_3PD%', 'T2_RM', 'T1_FT%', 'T1_A:T', 'T2_BPG', 'T1_EFG%', 'T2_APG', 'T2_TOF', 'T1_ORPG', 'T1_TEMPO', 'T1_FG%', 'T1_RPG'])

    elif target_variable == 'TARGET FTA':
        feature_list.extend(['PRED POSS', 'PRED FGA'])

        feature_list.extend(['Seed', 'T1_FTA', 'T1_FTPG', 'T2_FPG', 'T2_APG', 'T1_ORPG', 'T2_BPG', 'T1_FPG', 'T2_3PD%', 'T1_2PT FGM', 'T2_3PT%', 'T1_TM', 'T2_OPPG', 'T1_OPPG'])
        
    elif target_variable == 'TARGET FT%':
        feature_list.extend(['PRED FGA'])

        feature_list.extend(['Seed', 'T1_FT%', 'T1_Bench', 'T2_RM', 'T1_TM', 'T1_TOF', 'T2_SM', 'T2_DFG%', 'T1_APG', 'T2_BPG', 'T2_DRPG', 'T1_SPG', 'T1_2PT FGM'])

    elif target_variable == 'TARGET 2PT%':
        feature_list.extend(['PRED POSS', 'PRED FGA'])

        feature_list.extend(['Seed', 'T1_FGA', 'T1_2PT FGM', 'T1_3PPG', 'T1_3PTA', 'T2_DFG%', 'T1_EFG%', 'T1_APG', 'T2_RM'])

    else:
        # Get all inclusive feature list
        feature_list.append('Seed')
        for column in combined_df.columns:
            if ('T1_' in column) or ('T2_' in column):
                feature_list.append(column)

    return feature_list

#=====================================================
# FUNCTION TO CREATE AND DUMP THE MODEL
#=====================================================
def create_and_dump_model_using_feature_list(combined, target_variable, model_type):

    feature_list = get_model_feature_list(target_variable, combined)

    for dep in MODEL_DEPENDENCIES.get(target_variable, []):
        combined = add_preds_to_combined(combined, dep)

    if model_type == 'RIDGE':
        model, rmse, combined, params, rmse_logit = create_train_ridge_model(
            combined, feature_list, target_variable
        )
    else:   
        model, rmse, combined, params, rmse_logit = create_train_gbr_model(
            combined, feature_list, target_variable
        )

    print(target_variable + ' ' + model_type + ' MODEL')
    print_model_info(model, feature_list, rmse)

    dump({'model': model,
        'feature_list': feature_list,
        'cross_val_rmse': rmse,
        'cross_val_rmse_logit': rmse_logit if '%' in target_variable else None,
        'params': params},
        get_filename_for_target(target_variable))

def calc_rmse_diff_before_after_feature_change(combined, target_variable, before_rmse, new_feature_list):
    model, after_rmse, _, _, _ = create_train_gbr_model(
        combined, new_feature_list, target_variable, freeze_params=True
    )
    return np.round(after_rmse - before_rmse, 4)

def evaluate_features_to_remove(combined, target_variable):
    model_dict = load(get_filename_for_target(target_variable))
    model = model_dict['model']
    before_rmse = model_dict['cross_val_rmse']
    feature_list = model_dict['feature_list']

    # Extract feature importance (GBR only)
    if hasattr(model, 'feature_importances_'):
        importances = dict(zip(feature_list, model.feature_importances_))
    else:
        raise ValueError("Feature removal evaluation requires a tree-based model")

    results = []

    for feature in feature_list:
        new_feature_list = [f for f in feature_list if f != feature]
        temp_combined = combined.copy()

        # Ensure dependencies exist
        for dep in MODEL_DEPENDENCIES.get(target_variable, []):
            if dep not in temp_combined.columns:
                temp_combined = add_preds_to_combined(temp_combined, dep)

        rmse_diff = calc_rmse_diff_before_after_feature_change(
            temp_combined,
            target_variable,
            before_rmse,
            new_feature_list
        )

        results.append({
            'feature': feature,
            'importance': importances.get(feature, 0.0),
            'rmse_diff': rmse_diff
        })

    # Convert to DataFrame for clean sorting/printing
    results_df = pd.DataFrame(results)

    # Sort: lowest importance first, then best RMSE impact
    results_df = results_df.sort_values(
        by=['importance', 'rmse_diff'],
        ascending=[True, True]
    )

    print(f'\nFeature removal impact for {target_variable}')
    print(results_df.to_string(
        index=False,
        formatters={
            'importance': '{:.6f}'.format,
            'rmse_diff': '{:+.4f}'.format
        }
    ))

    print('\nRecommended removal candidates:')
    print(
        results_df[
            (results_df['importance'] < 0.01) &
            (results_df['rmse_diff'] <= 0)
        ][['feature', 'importance', 'rmse_diff']]
        .to_string(index=False)
    )

def evaluate_features_to_add(combined, target_variable):
    model_dict = load(get_filename_for_target(target_variable))
    before_rmse = model_dict['cross_val_rmse']

    # Current model feature list (after pruning zero-importance features)
    base_feature_list = prune_zero_importance_features(
        model_dict['model'], model_dict['feature_list']
    )

    # Build candidate feature pool (ALL possible features not already included)
    candidate_features = []
    for col_name in combined.columns:
        if ("T1_" in col_name or "T2_" in col_name) and col_name not in base_feature_list:
            candidate_features.append(col_name)

    features_to_add = []

    for feature in candidate_features:
        new_feature_list = base_feature_list + [feature]
        temp_combined = combined.copy()

        # Ensure dependencies exist
        for dep in MODEL_DEPENDENCIES.get(target_variable, []):
            if dep not in temp_combined.columns:
                temp_combined = add_preds_to_combined(temp_combined, dep)

        rmse_diff = calc_rmse_diff_before_after_feature_change(
            temp_combined,
            target_variable,
            before_rmse,
            new_feature_list
        )

        if rmse_diff < 0:
            print(f'Feature Added: {feature}, RMSE Diff: {rmse_diff}')
            features_to_add.append((feature, rmse_diff))

    print('\nFeatures to add (best first):')
    print([f for f, _ in sorted(features_to_add, key=lambda x: x[1])])

#=====================================================
# OVERALL MODEL CLASS TO LOAD CONTAINING ALL OF THE
# MODELS IN SEQUENCE, WITH NESTED FEATURES FOR EACH
# COMPONENT:
    # MODEL FEATURE LIST ATTRIBUTE, 
    # PARAMETERS ATTRIBUTE, 
    # RMSE ATTRIBUTE
# CLASS METHODS TO PULL FEATURES AND PREDICT
# OVERALL SCORE USING MONTE CARLO SIMULATION FOR EACH
# TARGET VARIABLE IN THE OVERALL MODEL, DRAWING VALUES
# FROM A NORMAL DISTRIBUTION, PREDICTED TARGET VALUE 
# AS THE MEAN AND RMSE AS THE STDEV (N=1000)
#=====================================================
class componentModel:
    def __init__(self, target_variable):
        filename = self._get_filename_for_target(target_variable)
        model_dict = load(filename)
        
        self.target_variable = target_variable
        self.model = model_dict['model']
        self.feature_list = model_dict['feature_list']
        self.rmse = model_dict['cross_val_rmse']
        self.rmse_logit = model_dict['cross_val_rmse_logit']
        self.params = model_dict['params']
    
    def _get_filename_for_target(self, target_variable):
        target_variable_for_file = target_variable.lower().replace(' ', '_')
        file_string = str('model_*.joblib').replace('*', target_variable_for_file)
        if '%' in file_string:
            file_string = file_string.replace('%', 'pct')
        return file_string

    def _inv_logit(self, x):
        return 1 / (1 + np.exp(-x))
    
    def generate_preds(self, input, noise_scale=1.0, covariance_matrix=None):
        # Resolve feature names (PRED -> TARGET)
        feature_list_copy = [
            f.replace('PRED', 'TARGET') if 'PRED' in f else f
            for f in self.feature_list
        ]

        if isinstance(input, pd.Series):
            X = input[feature_list_copy].values.reshape(1, -1)
            raw_pred = self.model.predict(X)[0]

            if '%' in self.target_variable:
                noisy_logit = raw_pred + noise_scale * np.random.normal(0, self.rmse_logit)
                return float(self._inv_logit(noisy_logit) * 100)

            return float(raw_pred + noise_scale * np.random.normal(0, self.rmse))

        elif isinstance(input, pd.DataFrame):
            X = input.loc[:, feature_list_copy].values
            raw_preds = self.model.predict(X)

            if '%' in self.target_variable:
                noisy_logits = raw_preds + noise_scale * np.random.normal(
                    0, self.rmse_logit, size=len(raw_preds)
                )
                preds = self._inv_logit(noisy_logits) * 100
            else:
                preds = raw_preds + noise_scale * np.random.normal(
                    0, self.rmse, size=len(raw_preds)
                )

            return pd.Series(preds, index=input.index, 
                             name=self.target_variable.replace('TARGET', 'PRED') if 'TARGET' in self.target_variable else self.target_variable)

        else:
            raise TypeError(
                "generate_preds expects a pandas Series or DataFrame"
            )
        
class overallModel:
    def __init__(self, all_data):
        self.target_poss_model = componentModel('TARGET POSS')
        self.target_fga_model = componentModel('TARGET FGA')
        self.target_2ptpct_model = componentModel('TARGET 2PT%')
        self.target_3pta_model = componentModel('TARGET 3PTA')
        self.target_3ptpct_model = componentModel('TARGET 3PT%')
        self.target_fta_model = componentModel('TARGET FTA')
        self.target_ftpct_model = componentModel('TARGET FT%')

        all_data = self.create_target_variable(all_data, 'TARGET FGA')
        all_data = self.create_target_variable(all_data, 'TARGET 2PT%')
        all_data = self.create_target_variable(all_data, 'TARGET 3PTA')
        all_data = self.create_target_variable(all_data, 'TARGET 3PT%')
        all_data = self.create_target_variable(all_data, 'TARGET FTA')
        all_data = self.create_target_variable(all_data, 'TARGET FT%')

        preds = {
            'PRED POSS': self.target_poss_model.generate_preds(all_data, 0.0),
            'PRED FGA': self.target_fga_model.generate_preds(all_data, 0.0),
            'PRED 2PT%': self.target_2ptpct_model.generate_preds(all_data, 0.0),
            'PRED 3PTA': self.target_3pta_model.generate_preds(all_data, 0.0),
            'PRED 3PT%': self.target_3ptpct_model.generate_preds(all_data, 0.0),
            'PRED FTA': self.target_fta_model.generate_preds(all_data, 0.0),
            'PRED FT%': self.target_ftpct_model.generate_preds(all_data, 0.0),
        }

        all_data = pd.concat([all_data, pd.DataFrame(preds, index=all_data.index)], axis=1)

        residuals = pd.DataFrame({
            'POSS': all_data['TARGET POSS'] - all_data['PRED POSS'],
            'FGA': all_data['TARGET FGA'] - all_data['PRED FGA'],
            '2PT%': all_data['TARGET 2PT%'] - all_data['PRED 2PT%'],
            '3PTA': all_data['TARGET 3PTA'] - all_data['PRED 3PTA'],
            '3PT%': all_data['TARGET 3PT%'] - all_data['PRED 3PT%'],
            'FTA': all_data['TARGET FTA'] - all_data['PRED FTA'],
            'FT%': all_data['TARGET FT%'] - all_data['PRED FT%'],
        })

        self.Sigma = residuals.cov()
        self.R = residuals.corr()

    def _draw_conditional_error(self, drawn, target):
        """
        drawn: dict of already-drawn errors {var: value}
        target: variable name (string)
        """
        A = list(drawn.keys())
        a = np.array([drawn[v] for v in A])

        Sigma = self.Sigma

        Sigma_AA = Sigma.loc[A, A].values
        Sigma_BA = Sigma.loc[target, A].values
        Sigma_AB = Sigma.loc[A, target].values
        Sigma_BB = Sigma.loc[target, target]

        mean = Sigma_BA @ np.linalg.solve(Sigma_AA, a)
        var = Sigma_BB - Sigma_BA @ np.linalg.solve(Sigma_AA, Sigma_AB)

        return mean + np.random.normal(0, np.sqrt(max(var, 1e-8)))
    
    def create_target_variable(self, input, target_variable):
        if isinstance(input, pd.DataFrame):
            keys = input.columns
        else:
            keys = input.keys()
        
        if target_variable in keys:
            return input

        if target_variable == 'TARGET FGA':
            input['TARGET FGA'] = (
                (input['PPG'] - input['FTPG'] - 3 * input['3PPG']) / 2.0
                + input['3PPG']
            ) / input['FG%']

        elif target_variable == 'TARGET 2PT%':
            input['TARGET 2PT%'] = input['2PT MADE'] / input['2PT ATT']

        elif target_variable[7:] in input.columns:
            input[target_variable] = input[target_variable[7:]]

        else:
            print('[WARN] No handling specified for creating target variable column in DataFrame.')

        return input

    def generate_pred_points(self, row_series):
        errors = {}

        # POSS — root node
        errors['POSS'] = np.random.normal(
            0,
            np.sqrt(self.Sigma.loc['POSS', 'POSS'])
        )

        poss_mean = self.target_poss_model.generate_preds(row_series, noise_scale=0.0)
        row_series['TARGET POSS'] = poss_mean + errors['POSS']

        # FGA | POSS
        errors['FGA'] = self._draw_conditional_error(errors, 'FGA')
        fga_mean = self.target_fga_model.generate_preds(row_series, noise_scale=0.0)
        row_series['TARGET FGA'] = fga_mean + errors['FGA']

        # 3PTA | POSS, FGA
        errors['3PTA'] = self._draw_conditional_error(errors, '3PTA')
        threepta_mean = self.target_3pta_model.generate_preds(row_series, noise_scale=0.0)
        row_series['TARGET 3PTA'] = threepta_mean + errors['3PTA']

        # 3PT% | POSS, FGA
        errors['3PT%'] = self._draw_conditional_error(errors, '3PT%')
        row_series['TARGET 3PT%'] = (
            self.target_3ptpct_model.generate_preds(row_series, noise_scale=0.0)
            + errors['3PT%']
        )

        # FTA | POSS, FGA
        errors['FTA'] = self._draw_conditional_error(errors, 'FTA')
        row_series['TARGET FTA'] = (
            self.target_fta_model.generate_preds(row_series, noise_scale=0.0)
            + errors['FTA']
        )

        # FT% | POSS, FGA
        errors['FT%'] = self._draw_conditional_error(errors, 'FT%')
        row_series['TARGET FT%'] = (
            self.target_ftpct_model.generate_preds(row_series, noise_scale=0.0)
            + errors['FT%']
        )

        # Final points
        two_ptm = (row_series['TARGET FGA'] - row_series['TARGET 3PTA']) * row_series['TARGET 2PT%'] / 100
        points = (
            two_ptm * 2
            + row_series['TARGET 3PTA'] * row_series['TARGET 3PT%'] / 100 * 3
            + row_series['TARGET FTA'] * row_series['TARGET FT%'] / 100
        )

        return points
    
    def run_montecarlo_comparison(self, team1_series, team2_series, n):
        team1_wins = 0
        team2_wins = 0

        for _ in range(n):
            poss1 = self.target_poss_model.generate_preds(team1_series)
            poss2 = self.target_poss_model.generate_preds(team2_series)

            average_pred_poss = (poss1 + poss2) / 2
            team1_series['TARGET POSS'] = average_pred_poss
            team2_series['TARGET POSS'] = average_pred_poss

            team1_points = self.generate_pred_points(team1_series)
            team2_points = self.generate_pred_points(team2_series)

            if team1_points > team2_points: 
                team1_wins += 1
            elif team2_points > team1_points:
                team2_wins += 1
            # If equal, we discard the tie
        
        return team1_wins, team2_wins
    

#=====================================================
# MAIN ROUTINE
#=====================================================

# Initial import
structured_2025 = pd.read_excel('NCAA_Validation.xlsx', sheet_name='Structured 2025')
structured_2024 = pd.read_excel('NCAA_Validation.xlsx', sheet_name='Structured 2024')
structured_2023 = pd.read_excel('NCAA_Validation.xlsx', sheet_name='Structured 2023')

combined = pd.concat([structured_2023, structured_2024, structured_2025])
combined = backfill_2pt_stats(combined)

if create_model_pieces_mode:
    for mode in [False, True]:
        run_feature_evaluation = mode
        retrain_same_params = mode

        if set_model == 1:
            # Model 1
            # Target: TARGET POSS
            # GBR Params: learning_rate=0.02, max_depth=2, n_estimators=250
            # Best RMSE: 4.2475

            target_variable = 'TARGET POSS'

            if run_feature_evaluation:
                evaluate_features_to_add(combined, target_variable)
                evaluate_features_to_remove(combined, target_variable)
            else:
                feature_list = get_model_feature_list(target_variable, combined)

                model, rmse, combined, params, rmse_logit = create_train_gbr_model(combined, feature_list, target_variable)
                print(target_variable + ' MODEL')
                print_model_info(model, feature_list, rmse)

                dump({
                    'model': model,
                    'feature_list': feature_list,
                    'cross_val_rmse': rmse,
                    'cross_val_rmse_logit': rmse_logit if '%' in target_variable else None,
                    'params': params
                    }, 
                    get_filename_for_target(target_variable))

        elif set_model == 2:
            # Model 2
            # Target: TARGET FGA
            # GBR Params: learning_rate=0.02, max_depth=2, n_estimators=300
            # Best RMSE: 0.066

            target_variable = 'TARGET FGA'
            combined = create_target_variable_in_df(combined, target_variable)

            if run_feature_evaluation:
                evaluate_features_to_add(combined, target_variable)
                evaluate_features_to_remove(combined, target_variable)
            else:
                create_and_dump_model_using_feature_list(combined, target_variable, 'GBR')

        elif set_model == 3:
            # Model 3
            # Target: TARGET FG%
            # Model Params: learning_rate=0.02, max_depth=2, n_estimators=200 
            # Best RMSE: 6.9477

            target_variable = 'TARGET FG%'
            combined = create_target_variable_in_df(combined, target_variable)

            if run_feature_evaluation:
                evaluate_features_to_add(combined, target_variable)
                evaluate_features_to_remove(combined, target_variable)
            else:
                create_and_dump_model_using_feature_list(combined, target_variable, 'GBR')

        elif set_model == 4:
            # Model 4
            # Target: TARGET 3PTA
            # Model Params: learning_rate=0.01, max_depth=3, n_estimators=250
            # GBR with few features: 5.2618
            
            target_variable = 'TARGET 3PTA'
            combined = create_target_variable_in_df(combined, target_variable)
            
            if run_feature_evaluation:
                evaluate_features_to_add(combined, target_variable)
                evaluate_features_to_remove(combined, target_variable)
            else:
                create_and_dump_model_using_feature_list(combined, target_variable, 'GBR')

        elif set_model == 5:
            # Model 5
            # Target: TARGET 3PT%
            # Model Params: learning_rate=0.01, max_depth=3, n_estimators=250
            # Best RMSE: 10.5273
            
            target_variable = 'TARGET 3PT%'
            combined = create_target_variable_in_df(combined, target_variable)
            
            if run_feature_evaluation:
                evaluate_features_to_add(combined, target_variable)
                evaluate_features_to_remove(combined, target_variable)
            else:
                create_and_dump_model_using_feature_list(combined, target_variable, 'GBR')

        elif set_model == 6:
            # Model 6
            # Target: TARGET FTA
            # Model Params: learning_rate=0.02, max_depth=3, n_estimators=250
            # Best RMSE: 6.2971
            
            target_variable = 'TARGET FTA'
            combined = create_target_variable_in_df(combined, target_variable)

            if run_feature_evaluation:
                evaluate_features_to_add(combined, target_variable)
                evaluate_features_to_remove(combined, target_variable)
            else:
                create_and_dump_model_using_feature_list(combined, target_variable, 'GBR')

        elif set_model == 7:
            # Model 7
            # Target: TARGET FT%
            # Model Params: learning_rate=0.01, max_depth=2, n_estimators=200
            # Best RMSE: 12.8521
            
            target_variable = 'TARGET FT%'
            combined = create_target_variable_in_df(combined, target_variable)
            
            if run_feature_evaluation:
                evaluate_features_to_add(combined, target_variable)
                evaluate_features_to_remove(combined, target_variable)
            else:
                create_and_dump_model_using_feature_list(combined, target_variable, 'GBR')

        elif set_model == 8:
            # Model 8
            # Target: TARGET 2PT%
            # Model Params: learning_rate=0.01, max_depth=2, n_estimators=200
            # Best RMSE: 0.0968
            
            target_variable = 'TARGET 2PT%'
            combined = create_target_variable_in_df(combined, target_variable)
            
            if run_feature_evaluation:
                evaluate_features_to_add(combined, target_variable)
                evaluate_features_to_remove(combined, target_variable)
            else:
                create_and_dump_model_using_feature_list(combined, target_variable, 'GBR')

else:
    model = overallModel(combined)
    results_list = []

    n_correct = 0
    n_overall = 0
    n_confident_and_incorrect = 0
    n_close_and_incorrect = 0
    for i, row in combined.iterrows():
        if i % 2 == 0:
            team1_series = row
        else:
            team2_series = row

            team1_wins, team2_wins = model.run_montecarlo_comparison(team1_series, team2_series, 1000)

            # Append data for Team 1
            results_list.append({
                'team_name': team1_series['Team'],
                'team_seed': team1_series['Seed'],
                'team_score': team1_series['PPG'],
                'opponent_score': team1_series['OPPG'],
                'projected_wins_team1': team1_wins,
                'projected_wins_team2': team2_wins
            })

            # Append data for Team 2 (swap scores)
            results_list.append({
                'team_name': team2_series['Team'],
                'team_seed': team2_series['Seed'],
                'team_score': team1_series['OPPG'], 
                'opponent_score': team1_series['PPG'],
                'projected_wins_team1': team2_wins,
                'projected_wins_team2': team1_wins
            })

            # Check the vals against the final score 
            print(f'Game {(i // 2) + 1}')
            print(f'Team1 Score: {team1_series['PPG']}')
            print(f'Team2 Score: {team1_series['OPPG']}')
            print(f'Team1 Proj Wins: {team1_wins}')
            print(f'Team2 Proj Wins: {team2_wins}\n\n')

            n_overall += 1
            if ((team1_wins > team2_wins) and (team1_series['PPG'] > team1_series['OPPG'])) or ((team1_wins < team2_wins) and (team1_series['PPG'] < team1_series['OPPG'])):
                n_correct += 1
            else:
                diff = np.abs(team1_wins - team2_wins)
                if diff > 100:
                    n_confident_and_incorrect += 1
                else:
                    n_close_and_incorrect += 1

    print(f'\n\nNumber correct: {n_correct}')
    print(f'Number close and incorrect: {n_close_and_incorrect}')
    print(f'Number confident and incorrect: {n_confident_and_incorrect}')
    print(f'Overall total: {n_overall}')

    results_df = pd.DataFrame(results_list)
    results_df.to_csv('tournament_projections_v3.csv', index=False)
    print(f"Successfully exported {len(results_df)} rows to csv")
    


            
        

