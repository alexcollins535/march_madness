import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.metrics import log_loss

from joblib import load, dump
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.nonparametric.smoothers_lowess import lowess


#=====================================================
# GLOBALS FOR CONFIGURATION
#=====================================================
# Seed for Random State
SEED = 7
np.random.seed(SEED)

# Fixed CV splitter for reproducibility
CV_SPLITTER = KFold(n_splits=10, shuffle=True, random_state=SEED)

# Cached file for Monte Carlo Simulation results
MC_CACHE_FILE = r'cache_files\mc_meta_features_v4.csv'

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
        file_string = str(r'mc_model_components\model_*.joblib').replace('*', target_variable_for_file)
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
            
            # # Debugging
            # print(self.target_variable)
            # print(feature_list_copy)
            # print(input[feature_list_copy], '\n')

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
                'generate_preds expects a pandas Series or DataFrame'
            )
        
class overallMonteCarloModel:
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
        '''
        drawn: dict of already-drawn errors {var: value}
        target: variable name (string)
        '''
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
            new_data = pd.DataFrame({target_variable: input[target_variable[7:]]})
            input = pd.concat([input, new_data], axis=1)

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

        # 2PT% | POSS, FGA
        errors['2PT%'] = self._draw_conditional_error(errors, '2PT%')
        row_series['TARGET 2PT%'] = (
            self.target_2ptpct_model.generate_preds(row_series, noise_scale=0.0)
            + errors['2PT%']
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

def make_pipeline(model_class, model_params=None):
    if model_params is None:
        model_params = {}
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model_class(**model_params))
    ])

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

def load_data():
    # Initial import
    structured_2025 = pd.read_excel('NCAA_Validation.xlsx', sheet_name='Structured 2025')
    structured_2025['YEAR'] = 2025

    structured_2024 = pd.read_excel('NCAA_Validation.xlsx', sheet_name='Structured 2024')
    structured_2024['YEAR'] = 2024

    structured_2023 = pd.read_excel('NCAA_Validation.xlsx', sheet_name='Structured 2023')
    structured_2023['YEAR'] = 2023

    combined = pd.concat([structured_2023, structured_2024, structured_2025])
    combined = backfill_2pt_stats(combined)

    if os.path.exists(MC_CACHE_FILE):
        print('[INFO] Loading cached Monte Carlo features...\n')
        meta_win_df = pd.read_csv(MC_CACHE_FILE)

    else:
        print('[INFO] Running Monte Carlo simulations...')

        model = overallMonteCarloModel(combined)
        meta_win_dataset = []

        games_completed = 0
        for i, row in combined.iterrows():
            if i % 2 == 0:
                team1_series = row
            else:
                team2_series = row

                team1_wins, team2_wins = model.run_montecarlo_comparison(
                    team1_series, team2_series, 1000
                )

                for team_series, team_wins, opp_series, opp_wins in [
                    (team1_series, team1_wins, team2_series, team2_wins),
                    (team2_series, team2_wins, team1_series, team1_wins)
                ]:
                    mc_win_prob = team_wins / (team_wins + opp_wins)
                    mc_win_margin = team_wins - opp_wins

                    team_seed = team_series['Seed']
                    opp_seed = opp_series['Seed']

                    team_win = int(team_series['PPG'] > team_series['OPPG'])

                    team_proj_poss = model.target_poss_model.generate_preds(team_series, 0.0)
                    opp_proj_poss = model.target_poss_model.generate_preds(opp_series, 0.0)

                    ft_rate_diff = (
                        team_series['T1_FTPG'] / team_series['T1_FGA']
                        - team_series['T2_FTPG'] / team_series['T2_FGA']
                    )

                    three_pt_rate_diff = (
                        team_series['T1_3PTA'] / team_series['T1_FGA']
                        - team_series['T2_3PTA'] / team_series['T2_FGA']
                    )

                    meta_win_dataset.append({
                        'SEED': team_seed,
                        'SEED_DIFF': team_seed - opp_seed,
                        'MC_WIN_PROB': mc_win_prob,
                        'MC_WIN_MARGIN': mc_win_margin,
                        'PROJECTED_POSSESSIONS': (team_proj_poss + opp_proj_poss) / 2,
                        'FT_RATE_DIFF': ft_rate_diff,
                        '3PT_RATE_DIFF': three_pt_rate_diff,
                        'WIN': team_win
                    })

                games_completed += 1
                if games_completed % 5 == 0:
                    print(f'[INFO] {games_completed} games simulated.\n')

        meta_win_df = pd.DataFrame(meta_win_dataset)
        meta_win_df.to_csv(MC_CACHE_FILE, index=False)
        meta_win_df = pd.DataFrame(meta_win_dataset)

    # Join the new and existing data
    combined = combined.reset_index(drop=True)
    merged = pd.concat([combined, meta_win_df], axis=1)

    # Fix to data published in CSV
    merged['FT_RATE_DIFF'] = (
        merged['T1_FTPG'] / merged['T1_FGA']
        - merged['T2_FTPG'] / merged['T2_FGA']
    )
    merged['3PT_RATE_DIFF'] = (
        merged['T1_3PTA'] / merged['T1_FGA']
        - merged['T2_3PTA'] / merged['T2_FGA']
    )
    return merged 

def normalize_matchup_probs(probs, team1_idx, team2_idx):
    norms = np.empty_like(probs)
    p1 = probs[team1_idx]
    p2 = probs[team2_idx]
    total = p1 + p2
    normalized_p1 = p1 / total
    normalized_p2 = p2 / total
    norms[team1_idx] = normalized_p1
    norms[team2_idx] = normalized_p2
    return norms

def feature_ablation_with_refit(df, feature_list, target_variable, model_class, model_params, cv_splitter):
    X = df[feature_list]
    y = df[target_variable]

    # Baseline model with all features
    baseline_model = make_pipeline(model_class, {
        k.replace('model__', ''): v for k, v in model_params.items()
    })
    raw_preds = cross_val_predict(
        baseline_model,
        X,
        y,
        cv=cv_splitter,
        method='predict_proba'
    )[:, 1]

    team1_idx = np.arange(0, len(raw_preds), 2)
    team2_idx = np.arange(1, len(raw_preds), 2)
    baseline_probs = normalize_matchup_probs(raw_preds, team1_idx, team2_idx)

    baseline_ll = log_loss(y, baseline_probs)
    baseline_acc = ((baseline_probs > 0.5).astype(int) == y).mean()

    results = []

    for feature in feature_list:
        reduced_features = [f for f in feature_list if f != feature]
        X_reduced = df[reduced_features]

        model = make_pipeline(model_class, {
            k.replace('model__', ''): v for k, v in model_params.items()
        })
        raw_preds = cross_val_predict(
            model,
            X_reduced,
            y,
            cv=cv_splitter,
            method='predict_proba'
        )[:, 1]
        
        probs = normalize_matchup_probs(raw_preds, team1_idx, team2_idx)

        ll = log_loss(y, probs)
        acc = ((probs > 0.5).astype(int) == y).mean()

        results.append({
            'removed_feature': feature,
            'log_loss': ll,
            'accuracy': acc,
            'delta_log_loss': ll - baseline_ll,
            'delta_accuracy': acc - baseline_acc
        })

    return pd.DataFrame(results).sort_values('delta_log_loss').reset_index(drop=True)

def forward_avg_feature_addition_with_refit(df, base_features, target_variable, model_class, model_params, cv_splitter, min_non_null_frac=0.9):
    y = df[target_variable]

    # --------------------------------------------------
    # EXACT T1 / T2 pairing
    # --------------------------------------------------
    t1_stats = {}
    t2_stats = {}

    for col in df.columns:
        if col.startswith('T1_') and is_numeric_dtype(df[col]):
            stat = col[len('T1_'):]
            t1_stats[stat] = col
        elif col.startswith('T2_') and is_numeric_dtype(df[col]):
            stat = col[len('T2_'):]
            t2_stats[stat] = col

    paired_stats = sorted(set(t1_stats.keys()) & set(t2_stats.keys()))

    # Baseline with base_features
    X_base = df[base_features]
    baseline_model = make_pipeline(model_class, {
        k.replace('model__', ''): v for k, v in model_params.items()
    })
    raw_preds = cross_val_predict(
        baseline_model,
        X_base,
        y,
        cv=cv_splitter,
        method='predict_proba'
    )[:, 1]

    team1_idx = np.arange(0, len(raw_preds), 2)
    team2_idx = np.arange(1, len(raw_preds), 2)
    baseline_probs = normalize_matchup_probs(raw_preds, team1_idx, team2_idx)

    baseline_ll = log_loss(y, baseline_probs)
    baseline_acc = ((baseline_probs > 0.5).astype(int) == y).mean()

    results = []

    # --------------------------------------------------
    # Evaluate each AVG feature individually
    # --------------------------------------------------
    for stat in paired_stats:
        t1_col = t1_stats[stat]
        t2_col = t2_stats[stat]

        avg_values = (df[t1_col] + df[t2_col]) / 2
        if avg_values.notna().mean() < min_non_null_frac:
            continue

        avg_col_name = f'AVG_{stat}'
        X_aug = df[base_features].copy()
        X_aug[avg_col_name] = avg_values  # assign Series as a column

        model = make_pipeline(model_class, {
            k.replace('model__', ''): v for k, v in model_params.items()
        })
        raw_preds = cross_val_predict(
            model,
            X_aug,
            y,
            cv=cv_splitter,
            method='predict_proba'
        )[:, 1]
        
        probs = normalize_matchup_probs(raw_preds, team1_idx, team2_idx)

        ll = log_loss(y, probs)
        acc = ((probs > 0.5).astype(int) == y).mean()

        results.append({
            'added_feature': f'AVG_{stat}',
            'log_loss': ll,
            'accuracy': acc,
            'delta_log_loss': ll - baseline_ll,
            'delta_accuracy': acc - baseline_acc
        })

    return pd.DataFrame(results).sort_values('delta_log_loss').reset_index(drop=True)

def forward_diff_feature_addition_with_refit(df, base_features, target_variable, model_class, model_params, cv_splitter, min_non_null_frac=0.9):
    y = df[target_variable]

    # --------------------------------------------------
    # EXACT T1 / T2 pairing
    # --------------------------------------------------
    t1_stats = {}
    t2_stats = {}

    for col in df.columns:
        if col.startswith('T1_') and is_numeric_dtype(df[col]):
            stat = col[len('T1_'):]
            t1_stats[stat] = col
        elif col.startswith('T2_') and is_numeric_dtype(df[col]):
            stat = col[len('T2_'):]
            t2_stats[stat] = col

    paired_stats = sorted(set(t1_stats.keys()) & set(t2_stats.keys()))

    # --------------------------------------------------
    # Baseline model with current base_features
    # --------------------------------------------------
    X_base = df[base_features]
    baseline_model = make_pipeline(model_class, {
        k.replace('model__', ''): v for k, v in model_params.items()
    })
    raw_preds = cross_val_predict(
        baseline_model,
        X_base,
        y,
        cv=cv_splitter,
        method='predict_proba'
    )[:, 1]

    team1_idx = np.arange(0, len(raw_preds), 2)
    team2_idx = np.arange(1, len(raw_preds), 2)
    baseline_probs = normalize_matchup_probs(raw_preds, team1_idx, team2_idx)

    baseline_ll = log_loss(y, baseline_probs)
    baseline_acc = ((baseline_probs > 0.5).astype(int) == y).mean()

    results = []

    # --------------------------------------------------
    # Evaluate each DIFF feature individually
    # --------------------------------------------------
    for stat in paired_stats:
        t1_col = t1_stats[stat]
        t2_col = t2_stats[stat]

        diff_values = df[t1_col] - df[t2_col]
        if diff_values.notna().mean() < min_non_null_frac:
            continue

        diff_col_name = f'DIFF_{stat}'
        X_aug = df[base_features].copy()
        X_aug[diff_col_name] = diff_values  # assign Series as a column

        model = make_pipeline(model_class, {
            k.replace('model__', ''): v for k, v in model_params.items()
        })

        raw_preds = cross_val_predict(
            model,
            X_aug,
            y,
            cv=cv_splitter,
            method='predict_proba'
        )[:, 1]

        probs = normalize_matchup_probs(raw_preds, team1_idx, team2_idx)

        ll = log_loss(y, probs)
        acc = ((probs > 0.5).astype(int) == y).mean()

        results.append({
            'added_feature': f'DIFF_{stat}',
            't1_column': t1_col,
            't2_column': t2_col,
            'log_loss': ll,
            'accuracy': acc,
            'delta_log_loss': ll - baseline_ll,
            'delta_accuracy': acc - baseline_acc
        })

    return pd.DataFrame(results).sort_values('delta_log_loss').reset_index(drop=True)

def train_and_evaluate_model(meta_win_df, feature_list, target_variable):
    # ==============================
    # Initial Model Training
    # ==============================
    
    X = meta_win_df[feature_list]
    y = meta_win_df[target_variable]

    feature_names = X.columns
    
    estimator_class = LogisticRegression
    pipe = make_pipeline(estimator_class)

    # params = {'n_estimators': [150, 200, 250],
    #           'learning_rate': [0.01, 0.02, 0.03],
    #           'max_depth': [2, 3]}
    
    params = {'model__C': [1.0, 5.0, 10, 50, 100],
              'model__max_iter': [1000]}

    grid_search = GridSearchCV(
        pipe,
        param_grid=params,
        scoring='neg_log_loss',
        cv=CV_SPLITTER
    )

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    feature_weights = best_model.named_steps['model'].coef_
    print(f'Best Model Params: {best_params}')
    for name, weight in zip(feature_names, feature_weights[0]): 
        print(f'{name}: {np.round(weight, 4)}')
    print()

    # ==============================
    # Paired Probability Predictions
    # ==============================
    raw_preds = cross_val_predict(
        best_model,
        X,
        y,
        cv=CV_SPLITTER,
        method='predict_proba'
    )[:, 1]

    team1_idx = np.arange(0, len(raw_preds), 2)
    team2_idx = np.arange(1, len(raw_preds), 2)
    probs = normalize_matchup_probs(raw_preds, team1_idx, team2_idx)

    # ==============================
    # Model Evaluation
    # ==============================
    ll = log_loss(y, probs)
    print('Normalized Log Loss:', np.round(ll, 4))

    acc = ((probs > 0.5).astype(int) == y).mean()
    print('Normalized Accuracy:', np.round(acc, 4))

    chalk_acc = ((meta_win_df['SEED_DIFF'] < 0).astype(int) == y).mean()
    print('Chalk Accuracy:', np.round(chalk_acc, 4))
    print()

    # ==============================
    # Calibration Check
    # ==============================
    meta_win_df = meta_win_df.copy()
    meta_win_df['Pred Range'] = pd.cut(probs, bins=10)
    calibration = meta_win_df.groupby('Pred Range')['WIN'].agg(['mean', 'count'])
    calibration.columns = ['Win Rate', 'Sample Size']
    print('Calibration by Predicted Probability:')
    print(calibration)
    print()

    # ===============================
    # Feature Ablation (Frozen Model)
    # ===============================
    print('[INFO] Running feature abalation testing')
    ablation_results = feature_ablation_with_refit(
        df=meta_win_df,
        feature_list=feature_list,
        target_variable=target_variable,
        model_class=estimator_class,
        model_params=best_params,
        cv_splitter=CV_SPLITTER
    )

    # ===============================
    # Forward Feature Addition Test
    # ===============================
    print('[INFO] Running single feature addition testing')
    forward_results = forward_avg_feature_addition_with_refit(
        df=meta_win_df,
        base_features=feature_list,
        target_variable=target_variable,
        model_class=estimator_class,
        model_params=best_params,
        cv_splitter=CV_SPLITTER
    )

    # ===============================
    # Forward Diff Feature Addition Test
    # ===============================
    print('[INFO] Running diff feature addition testing')
    forward_diff_results = forward_diff_feature_addition_with_refit(
        df=meta_win_df,
        base_features=feature_list,
        target_variable=target_variable,
        model_class=estimator_class,
        model_params=best_params,
        cv_splitter=CV_SPLITTER
    )

    ablation_results['action'] = 'removed ' + ablation_results['removed_feature']
    forward_results['action'] = 'added ' + forward_results['added_feature']
    forward_diff_results['action'] = 'added ' + forward_diff_results['added_feature']

    merged = pd.concat([ablation_results, forward_results, forward_diff_results], axis=0)
    merged = merged.drop(['removed_feature', 'added_feature', 't1_column', 't2_column'], axis=1)
    sorted = merged.sort_values(by='delta_log_loss', ascending=True).reset_index(drop=True)
    sorted = sorted[sorted['delta_accuracy'] >= 0.0]
    print('\nTop 15 Feature Addition/Removal Results:')
    print(sorted.head(15))

    return best_model, best_params

# Additional features from file
def load_additional_features():
    df2023 = pd.read_excel('NCAA_Validation_Additional_Features.xlsx', sheet_name='2023')
    df2023['YEAR'] = 2023

    df2024 = pd.read_excel('NCAA_Validation_Additional_Features.xlsx', sheet_name='2024')
    df2024['YEAR'] = 2024

    df2025 = pd.read_excel('NCAA_Validation_Additional_Features.xlsx', sheet_name='2025')
    df2025['YEAR'] = 2025

    df = pd.concat([df2023, df2024, df2025])

    # Format columns, then drop
    df['Last Before 64'] = pd.to_datetime(df['Last Before 64'])
    df['64 Date'] = pd.to_datetime(df['64 Date'])

    df['REST_DAYS'] = np.where(df['Play In'], 1, (df['64 Date'] - df['Last Before 64']).dt.days)
    df['CT_WIN_PCT'] = df['Wins Conf'] / (df['Wins Conf'] + df['Losses Conf'])
    df['GAMES_LAST_WEEK'] = df['Conf Games Last Week'] + df['Play In']

    df = df.rename(columns={'Wins Conf': 'CT_WIN_TOTAL'})
    df = df[['Team', 'YEAR', 'REST_DAYS', 'CT_WIN_PCT', 'GAMES_LAST_WEEK', 'CT_WIN_TOTAL']]
    return df

def create_symmetric_matchup_features(df, diff=True, avg=True):
    """
    Automatically creates DIFF_ and/or AVG_ features for every
    matching T1_* / T2_* column pair in the dataframe.
    """

    ignore_stats = {'Team', 'Opp', 'YEAR'}

    t1_cols = {col[3:]: col for col in df.columns if col.startswith('T1_')}
    t2_cols = {col[3:]: col for col in df.columns if col.startswith('T2_')}

    shared_stats = sorted(set(t1_cols.keys()) & set(t2_cols.keys()))

    new_features = {}

    for stat in shared_stats:
        if stat in ignore_stats:
            continue

        t1_col = t1_cols[stat]
        t2_col = t2_cols[stat]

        if diff:
            new_features[f'DIFF_{stat}'] = df[t1_col] - df[t2_col]

        if avg:
            new_features[f'AVG_{stat}'] = (df[t1_col] + df[t2_col]) / 2

    new_df = pd.DataFrame(new_features, index=df.index)

    return pd.concat([df, new_df], axis=1)

def plot_residuals(model, df, feature_list, target_variable, cv_splitter):
    X = df[feature_list]
    y = df[target_variable]

    team1_idx = np.arange(0, len(y), 2)
    team2_idx = np.arange(1, len(y), 2)

    raw_preds = cross_val_predict(
        model, X, y, cv=cv_splitter, method='predict_proba'
    )[:, 1]

    probs = normalize_matchup_probs(raw_preds, team1_idx, team2_idx)
    residuals = y - probs

    n_features = len(feature_list)
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols

    fig = plt.figure(figsize=(6 * ncols, 4 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    for i, feature in enumerate(feature_list):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        feature_vals = df[feature].values

        ax.scatter(feature_vals, residuals, alpha=0.4, s=20)

        # Lowess smoothing line to reveal nonlinear patterns
        smoothed = lowess(residuals, feature_vals, frac=0.4)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2)

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel(feature)
        ax.set_ylabel('Residual (y - prob)')
        ax.set_title(f'Residuals vs {feature}')

    plt.suptitle('Residual Plots by Feature', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('residual_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('[INFO] Saved residual_plots.png')

def diagnose_zero_accumulation(df, feature_list):
    print('=' * 60)
    print('ZERO ACCUMULATION DIAGNOSTICS')
    print('=' * 60)

    for feature in feature_list:
        col = df[feature]
        n_total = len(col)
        n_zero = (col == 0).sum()
        n_null = col.isna().sum()
        pct_zero = n_zero / n_total * 100
        pct_null = n_null / n_total * 100

        if pct_zero > 5:  # Only flag features with notable zero accumulation
            print(f'\n--- {feature} ---')
            print(f'  Zeros:   {n_zero}/{n_total} ({pct_zero:.1f}%)')
            print(f'  Nulls:   {n_null}/{n_total} ({pct_null:.1f}%)')
            print(f'  Min:     {col.min():.4f}')
            print(f'  Max:     {col.max():.4f}')
            print(f'  Mean:    {col.mean():.4f}')

            # Check if zeros cluster by year
            if 'YEAR' in df.columns:
                print(f'  Zeros by year:')
                zero_by_year = df[df[feature] == 0].groupby('YEAR').size()
                total_by_year = df.groupby('YEAR').size()
                for year in total_by_year.index:
                    z = zero_by_year.get(year, 0)
                    t = total_by_year[year]
                    print(f'    {year}: {z}/{t} ({z/t*100:.1f}%)')

            # Check if zeros are associated with a win/loss pattern (potential data artifact)
            if 'WIN' in df.columns:
                zero_win_rate = df.loc[df[feature] == 0, 'WIN'].mean()
                nonzero_win_rate = df.loc[df[feature] != 0, 'WIN'].mean()
                print(f'  Win rate when zero:     {zero_win_rate:.3f}')
                print(f'  Win rate when non-zero: {nonzero_win_rate:.3f}')

            # Trace back to source T1/T2 columns if this is a DIFF or AVG feature
            prefix = feature.split('_')[0]  # DIFF or AVG
            stat = feature[len(prefix) + 1:]  # everything after DIFF_ or AVG_

            t1_col = f'T1_{stat}'
            t2_col = f'T2_{stat}'

            if t1_col in df.columns and t2_col in df.columns:
                t1_zeros = (df[t1_col] == 0).sum()
                t2_zeros = (df[t2_col] == 0).sum()
                t1_nulls = df[t1_col].isna().sum()
                t2_nulls = df[t2_col].isna().sum()
                print(f'  Source columns:')
                print(f'    {t1_col}: {t1_zeros} zeros, {t1_nulls} nulls')
                print(f'    {t2_col}: {t2_zeros} zeros, {t2_nulls} nulls')

                # Check if zeros in DIFF come from both teams having the same value
                # vs one team genuinely being zero
                if prefix == 'DIFF':
                    both_equal = (df[t1_col] == df[t2_col]).sum()
                    t1_zero_only = ((df[t1_col] == 0) & (df[t2_col] != 0)).sum()
                    t2_zero_only = ((df[t2_col] == 0) & (df[t1_col] != 0)).sum()
                    both_zero = ((df[t1_col] == 0) & (df[t2_col] == 0)).sum()
                    print(f'  DIFF zero breakdown:')
                    print(f'    Both teams equal (cancels to 0): {both_equal}')
                    print(f'    T1 zero only:                    {t1_zero_only}')
                    print(f'    T2 zero only:                    {t2_zero_only}')
                    print(f'    Both zero:                       {both_zero}')

    print('\n' + '=' * 60)

#=====================================================
# MAIN ROUTINE
#=====================================================
if __name__ == '__main__':

    df = load_data()

    # Add and merge additional features
    additional_df = load_additional_features()
    t1_additional = additional_df.add_prefix('T1_')
    t2_additional = additional_df.add_prefix('T2_')

    df = df.merge(t1_additional, left_on=['Team', 'YEAR'], right_on=['T1_Team', 'T1_YEAR'], how='inner')
    df = df.merge(t2_additional, left_on=['Team', 'YEAR'], right_on=['T2_Team', 'T2_YEAR'], how='inner')

    df = df.drop(columns=['T1_Team', 'T2_Team', 'T1_YEAR', 'T2_YEAR'])

    # Merge sanity check
    if len(df) % 2 != 0:
        raise ValueError('Merge broke matchup pairing.')

    # Create Additional Features
    new_features = pd.DataFrame({
        'T1_OR%': df['T1_ORPG'] / df['T1_RPG'],
        'T2_OR%': df['T2_ORPG'] / df['T2_RPG'],
    }, index=df.index)
    df = pd.concat([df, new_features], axis=1)

    df = create_symmetric_matchup_features(df)

    feature_list = ['SEED_DIFF', 'MC_WIN_PROB', 'FT_RATE_DIFF', 'DIFF_SM', 'DIFF_FT%', '3PT_RATE_DIFF', 'DIFF_FTA', 'DIFF_3PPG', 'DIFF_TM', 'DIFF_TPG']
    target_variable = 'WIN'

    diagnose_zero_accumulation(df, feature_list)

    model, params = train_and_evaluate_model(df, feature_list, target_variable)

    plot_residuals(model, df, feature_list, target_variable, CV_SPLITTER)

    dump({'model': model,
            'feature_list': feature_list,
            'params': params},
            r'mc_model_components\model_target_wins.joblib')


    #=====================================================
    # Export team-level predictions with model probabilities
    #=====================================================
    team_preds = []

    for i in range(0, len(df), 2):
        team1 = df.iloc[i]
        team2 = df.iloc[i + 1]

        # Prepare feature vectors for the model
        team1_features = pd.DataFrame([team1[feature_list].to_dict()])
        team2_features = pd.DataFrame([team2[feature_list].to_dict()])

        # Predict win probabilities
        team1_proba = model.predict_proba(team1_features)[0, 1]
        team2_proba = model.predict_proba(team2_features)[0, 1]

        # Normalize to sum to 1
        total = team1_proba + team2_proba
        team1_proba /= total
        team2_proba /= total

        # Record team-level info
        team_preds.append({
            'Team': team1['Team'],
            'Seed': team1['Seed'],
            'Points Scored': team1['PPG'],  # actual points per game
            'Opponent Points': team1['OPPG'],
            'Pred Win Prob': team1_proba,
            'Chalk Win Prob': 1.0 if team1['Seed'] < team2['Seed'] else (0.0 if team1['Seed'] > team2['Seed'] else 0.5) 
        })
        team_preds.append({
            'Team': team2['Team'],
            'Seed': team2['Seed'],
            'Points Scored': team2['PPG'],
            'Opponent Points': team2['OPPG'],
            'Pred Win Prob': team2_proba,
            'Chalk Win Prob': 1.0 if team2['Seed'] < team1['Seed'] else (0.0 if team2['Seed'] > team1['Seed'] else 0.5) 
        })

    # Create DataFrame
    predictions_df = pd.DataFrame(team_preds)

    # Export to CSV
    predictions_df.to_csv('historical_team_win_probabilities.csv', index=False)
    print("[INFO] Exported team predictions to 'historical_team_win_probabilities.csv'")