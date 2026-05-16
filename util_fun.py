'''
Utility functions for time series data processing.
Includes:
- Rolling window normalization dataset class.
- (create_sequences) Sequence creation function.
- Training functions for online and offline learning.
- Metrics calculation and plotting functions.
'''

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import torch
from torch.utils.data import Dataset
from collections import defaultdict
from torch.utils.data import DataLoader
import plotly.express as px
from plotly_resampler import FigureResampler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools


EPSILON = 1e-6  # For numerical stability in normalization

# window normalization
class RollingNormTimeSeriesDataset(Dataset):
    """
    Per-window normalization with last-value centering for H_bar.

    All features in X: per-feature min-max within the lookback window.
    H_bar in X and y: centered on the LAST observed H_bar value (H_bar[lookback_end]),
    scaled by the lookback std.

    Why last-value centering instead of min:
    - Using h_bar_min as zero-point anchors predictions to the lowest recent level.
      When a flood arrives the lookback is still calm → predictions are systematically
      shifted downward (model predicts "within lookback range" which is too low).
    - h_bar_last = H_bar[lookback_end] is always the current observed level, so
      y_scaled ≈ 0 means "stays at current level" and y_scaled > 0 means "rises".
      The model learns a centered, physically meaningful target regardless of absolute level.

    No data leakage: h_bar_ref and h_bar_scale both come only from the lookback X.

    Returns (per sample):
        x_scaled, y_scaled, h_bar_ref, h_bar_scale
    Unscale: y_true = y_scaled * h_bar_scale + h_bar_ref
    """
    def __init__(self, X_seq, y_seq, h_bar_index, epsilon=EPSILON, y_clip=5.0):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y_seq = torch.tensor(y_seq, dtype=torch.float32)
        self.h_bar_index = h_bar_index
        self.epsilon = epsilon
        self.y_clip = y_clip

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        x_raw = self.X_seq[idx]  # (lookback, n_features)
        y_raw = self.y_seq[idx]  # (horizon,)

        # Min-max normalize all features from lookback (no leakage)
        min_vals, _ = torch.min(x_raw, dim=0)
        max_vals, _ = torch.max(x_raw, dim=0)
        range_vals = max_vals - min_vals + self.epsilon
        x_scaled = (x_raw - min_vals) / range_vals

        # H_bar: last-value reference + lookback std scale (no leakage)
        h_bar_lookback = x_raw[:, self.h_bar_index]
        h_bar_ref   = h_bar_lookback[-1]                     # current observed level
        h_bar_scale = h_bar_lookback.std() + self.epsilon    # lookback variability

        x_scaled[:, self.h_bar_index] = (h_bar_lookback - h_bar_ref) / h_bar_scale
        y_scaled = (y_raw - h_bar_ref) / h_bar_scale
        y_scaled = torch.clamp(y_scaled, -self.y_clip, self.y_clip)

        return x_scaled, y_scaled, h_bar_ref, h_bar_scale


class ZScoreNormTimeSeriesDataset(Dataset):
    """
    Per-window z-score normalization. Mean and std come only from the lookback window.

    Why z-score over min-max for online learning:
    - Min-max maps the lookback to [0,1], so future values outside that range produce
      unbounded scaled targets (e.g. y_scaled = 10 during a flood).
    - Z-score centers on 0 with unit variance — a future spike is only a few stds away,
      keeping y_scaled in a stable range without hard clipping.

    Returns same 4-tuple interface as RollingNormTimeSeriesDataset.
    Unscale: y_true = y_scaled * h_bar_std + h_bar_mean
    """
    def __init__(self, X_seq, y_seq, h_bar_index, epsilon=EPSILON):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y_seq = torch.tensor(y_seq, dtype=torch.float32)
        self.h_bar_index = h_bar_index
        self.epsilon = epsilon

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        x_raw = self.X_seq[idx]  # (lookback, n_features)
        y_raw = self.y_seq[idx]  # (horizon,)

        mean_vals = torch.mean(x_raw, dim=0)
        std_vals  = torch.std(x_raw, dim=0) + self.epsilon
        x_scaled = (x_raw - mean_vals) / std_vals

        h_bar_mean = mean_vals[self.h_bar_index]
        h_bar_std  = std_vals[self.h_bar_index]
        y_scaled = (y_raw - h_bar_mean) / h_bar_std

        return x_scaled, y_scaled, h_bar_mean, h_bar_std


class PercentileNormTimeSeriesDataset(Dataset):
    """
    Per-window percentile-based min-max normalization (Plan 1.1).
    Uses [low_q, high_q] quantiles of the lookback instead of strict min/max,
    so a single outlier in the 48h window can't collapse the normalization range.
    y_scaled is clamped to [-y_clip, y_clip].

    Args:
        low_q / high_q: quantile bounds (default 0.05 / 0.95)
        y_clip: symmetric clamp bound on y_scaled (default 5.0)
    Returns same 4-tuple interface as RollingNormTimeSeriesDataset.
    """
    def __init__(self, X_seq, y_seq, h_bar_index, epsilon=EPSILON,
                 low_q=0.05, high_q=0.95, y_clip=5.0):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y_seq = torch.tensor(y_seq, dtype=torch.float32)
        self.h_bar_index = h_bar_index
        self.epsilon = epsilon
        self.low_q = low_q
        self.high_q = high_q
        self.y_clip = y_clip

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        x_raw = self.X_seq[idx]  # (lookback, n_features)
        y_raw = self.y_seq[idx]  # (horizon,)

        min_vals = torch.quantile(x_raw, self.low_q, dim=0)
        max_vals = torch.quantile(x_raw, self.high_q, dim=0)
        range_vals = max_vals - min_vals + self.epsilon
        x_scaled = (x_raw - min_vals) / range_vals

        h_bar_min = min_vals[self.h_bar_index]
        h_bar_range = range_vals[self.h_bar_index]
        y_scaled = (y_raw - h_bar_min) / h_bar_range
        y_scaled = torch.clamp(y_scaled, -self.y_clip, self.y_clip)

        return x_scaled, y_scaled, h_bar_min, h_bar_range


def create_sequences(data, lookback, horizon, target_value_index, step = None):
    ''' Create input-output sequences using sliding window approach.
    Args:
        data (np.array): The raw data array.
        lookback (int): Number of past time steps to use as input.
        horizon (int): Number of future time steps to predict.
        h_bar_index (int): Index of the H_bar feature in the data.
        step (int or None): step size for sliding window(the timestamps):
            - None/horizon: the original behavior - y[n] appears in X[n+1] --> the leakage
            - lookback+ horizon : no overlap between X and y windows, but more samples (under test).
    Returns:
        X (np.array): Input sequences of shape (num_samples, lookback, num_features).
        y (np.array): Output sequences of shape (num_samples, horizon).
    '''
    if step is None:
        step = horizon

    X, y = [], []
    print("Creating sequences...")
    print(f"Data length: {len(data)}, lookback: {lookback}, horizon: {horizon} , step: {step}")
    
    for i in range(0, len(data), step):
        
        historical_data = data[i:(i + lookback), :]
        future_y = data[(i + lookback):(i + lookback + horizon), target_value_index]
        if historical_data.shape[0] != lookback or future_y.shape[0] != horizon:
            print(f"End of sufficient data at index {i}.")
            break
        
        X.append(historical_data) # All 4 features for lookback
        y.append(future_y) # Just H_bar for horizon
    print(f"Total sequences created: {len(X)}\n",
          f"X shape: {np.array(X).shape}, y shape: {np.array(y).shape}")

    return np.array(X), np.array(y)
######################################################
################ Training Functions ###################
#######################################################
def train_model_online(model, optimizer, criterion,
                       dataset, df,
                       LOOKBACK, HORIZON, batch_size,
                       max_grad_norm=None,
                       use_amnesia_strategy=False, amnesia_threshold=2.5, amnesia_warmup_batches=10,
                       amnesia_new_lr=None,
                       amnesia_reset_lr=None,
                       silent=False):

    # --- Setup from original function ---
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_batches = len(train_loader)
    y_ground_truth, y_predictions = np.array([0.0]).reshape(1,1), np.array([0.0]).reshape(1,1)
    all_indices, batch_losses = np.array(['']), []

    # --- NEW: Setup for Amnesia Strategy ---
    running_avg_loss, ema_alpha = 0.0, 0.1
    amnesia_active = False # Flag to track if we're in a "spike" state

    if not silent:
        print(f"\nStarting 'online' training with {num_batches} batches...")
        if use_amnesia_strategy:
            print(f"  > Amnesia Strategy ENABLED (Threshold: {amnesia_threshold}x avg, "
                  f"Warmup: {amnesia_warmup_batches} batches)")
    
    model.train()
    for i, (x_batch_scaled, y_batch_scaled, h_min, h_range) in enumerate(train_loader):
        # --- Training Step ---
        y_pred_scaled = model(x_batch_scaled)
        
        # --- Data Unscaling & Storage (Original) ---
        h_range_u, h_min_u = h_range.unsqueeze(-1), h_min.unsqueeze(-1)
        y_pred_true = (y_pred_scaled * h_range_u + h_min_u).detach().cpu().numpy()
        y_unscaled =  (y_batch_scaled * h_range_u + h_min_u).detach().cpu().numpy()
        y_predictions = np.concatenate((y_predictions, y_pred_true), axis=1)
        y_ground_truth = np.concatenate((y_ground_truth, y_unscaled), axis=1)
        idx_start, idx_end = (i*HORIZON) + LOOKBACK, (i*HORIZON) + LOOKBACK + HORIZON
        all_indices = np.concatenate((all_indices, df.index[idx_start:idx_end]))

        # 2. Calculate loss (on scaled data)
        loss = criterion(y_pred_scaled, y_batch_scaled)
        current_loss = loss.item()
        batch_losses.append(current_loss)

        # --- NEW: Amnesia Strategy Logic ---
        if use_amnesia_strategy:
            if i == 0: running_avg_loss = current_loss
            else: running_avg_loss = (ema_alpha * current_loss) + ((1 - ema_alpha) * running_avg_loss)

            if i >= amnesia_warmup_batches:
                spike_threshold = running_avg_loss * amnesia_threshold
                
                # --- TRIGGER AMNESIA ---
                if current_loss > spike_threshold and not amnesia_active:
                    amnesia_active = True
                    optimizer.state = defaultdict(dict)
                    if not silent:
                        print(f"\n  *** AMNESIA TRIGGERED at batch {i+1} ***")
                        print(f"      Loss {current_loss:.6f} > Threshold ({spike_threshold:.6f})")
                    if amnesia_new_lr is not None:
                        if not silent:
                            print(f"      Setting LR to {amnesia_new_lr}")
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = amnesia_new_lr
                    running_avg_loss = current_loss

                # --- RESET AFTER SPIKE ---
                elif current_loss < running_avg_loss and amnesia_active:
                    amnesia_active = False
                    if not silent:
                        print(f"  *** AMNESIA RESET at batch {i+1}, Loss {current_loss:.6f}")
                    if amnesia_reset_lr is not None:
                        if not silent:
                            print(f"      Resetting LR to {amnesia_reset_lr}")
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = amnesia_reset_lr
        
        # 3. Backward pass and optimize (Original)
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        
        if not silent and (i+1) % 100 == 0:
            print(f"   Batch {i+1}/{num_batches}, Loss: {current_loss:.6f}")

    if not silent:
        print(f"   Batch {num_batches}/{num_batches}, Loss: {current_loss:.6f}")
        print("Training complete.")
    
    # Final cleanup: ensure LR is back to original if amnesia is still active
    if amnesia_active and amnesia_reset_lr is not None:
        print("Final LR reset.")
        for param_group in optimizer.param_groups:
            param_group['lr'] = amnesia_reset_lr
            
    return y_ground_truth[:,1:].flatten(), y_predictions[:,1:].flatten(), all_indices[1:].flatten(), batch_losses


def train_model_offline(x, y, x_test, y_test, testing_timestep, model, optimizer,criterion,batch_size=32,
                         max_epoch=5000, early_Stop_threshold=20):
    """
    Train a PyTorch model in offline (batch) mode on normalized data and return unnormalized test predictions.

    Args:
        x (array-like): Training inputs (samples,...,features).
        y (array-like): Training targets (samples,...).
        x_test (array-like): Test inputs.
        y_test (array-like): Test targets.
        testing_timestep (array-like or pd.Series): Timestamps for test samples.
        model (torch.nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (torch.nn.Module): Loss function.
        max_epoch (int): Maximum number of epochs.
        early_Stop_threshold (int): Patience (number of epochs without improvement before stopping).
    Returns:
        tuple: (y_true_unnormalized, y_pred_unnormalized, testing_timestep)
    """

    from torch.utils.data import TensorDataset, DataLoader as DataLoader

    # Convert to numpy arrays for stable broadcasting operations
    x = np.array(x)
    y = np.array(y)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # --- Compute min/max/range for X per-feature across all training time points ---
    num_features = x.shape[-1]
    x_flat = x.reshape(-1, num_features)
    x_min = x_flat.min(axis=0)
    x_max = x_flat.max(axis=0)
    x_range = x_max - x_min + EPSILON
    # Normalize training and test inputs (min-max)
    x_scaled = (x - x_min) / x_range
    x_test_scaled = (x_test - x_min) / x_range

    # --- Compute min/max/range for y (global across training targets) ---
    y_flat = y.reshape(-1)
    y_min = y_flat.min()
    y_max = y_flat.max()
    y_range = y_max - y_min + EPSILON

    y_scaled = (y - y_min) / y_range
    y_test_scaled = (y_test - y_min) / y_range

    # Convert to torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x_train_t = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32, device=device)


    train_dataset = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.MSELoss()
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    for epoch in range(1, max_epoch + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        ###################################################################
        # Validation on test set (scaled)
        ####################################################################
        model.eval()
        with torch.no_grad():
            y_test_pred_scaled = model(x_test_t)
            val_loss = criterion(y_test_pred_scaled, y_test_t).item()

        print(f"Epoch {epoch}/{max_epoch}  TrainLoss={epoch_train_loss:.6f}  ValLoss={val_loss:.6f}")

        # Early stopping logic
        if val_loss + 1e-12 < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # store best state (cpu)
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_Stop_threshold:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_Stop_threshold} epochs).")
                break

    # Load best model state
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        y_pred_scaled_final = model(x_test_t).cpu().numpy()

    # Unnormalize predictions and true values
    y_pred_unnorm = (y_pred_scaled_final * y_range) + y_min
    y_true_unnorm = (y_test_t.cpu().numpy() * y_range) + y_min

    # Flatten outputs to 1D if possible
    y_pred_out = y_pred_unnorm.reshape(-1)
    y_true_out = y_true_unnorm.reshape(-1)

    # Print short summary
    print("\nTest results (unnormalized):")
    print(f" y_true shape: {y_true_out.shape}, y_pred shape: {y_pred_out.shape} , timestamp length: {len(testing_timestep)}")

    return y_true_out, y_pred_out, testing_timestep
######################################################
################ Evaluating functions ##########
######################################################

def model_evaluate_with_norm(model,criterion , 
                            dataset, ds_timesteps,
                            lookback, horizon, batch_size=1 ):
    ''' Evaluate a trained model on normalized test data and return unnormalized predictions.
    Args:
        model (torch.nn.Module): Trained model.
        criterion (torch.nn.Module): Loss function.
        dataset (Dataset): Test Dataset for evaluation.
        ds_timesteps (pd.Index): Timestamps corresponding to the dataset samples.
        lookback (int): Lookback window size.
        horizon (int): Prediction horizon size.
        batch_size (int): Batch size for DataLoader.
    '''
    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        y_ground_truth, y_predictions = np.array([0.0]).reshape(1,1), np.array([0.0]).reshape(1,1)
        all_indices, batch_losses = np.array(['']), []

        for i, (x_batch_scaled, y_batch_scaled, h_min, h_range) in enumerate(test_loader):
            # get predictions
            y_pred_scaled = model(x_batch_scaled)
            
            # --- Data Unscaling & Storage (Original) ---
            h_range_u, h_min_u = h_range.unsqueeze(-1), h_min.unsqueeze(-1)
            y_pred_true = (y_pred_scaled * h_range_u + h_min_u).detach().cpu().numpy()
            y_unscaled =  (y_batch_scaled * h_range_u + h_min_u).detach().cpu().numpy()
            y_predictions = np.concatenate((y_predictions, y_pred_true), axis=1)
            y_ground_truth = np.concatenate((y_ground_truth, y_unscaled), axis=1)
            idx_start, idx_end = (i*horizon) + lookback, (i*horizon) + lookback + horizon
            all_indices = np.concatenate((all_indices, ds_timesteps[idx_start:idx_end]))

            # 2. Calculate loss (on scaled data)
            loss = criterion(y_pred_scaled, y_batch_scaled)
            current_loss = loss.item()
            batch_losses.append(current_loss)
        print("Testing complete.")
    return y_ground_truth[:,1:].flatten(), y_predictions[:,1:].flatten(), all_indices[1:].flatten()

#######################################################
################ Metrics & Plotting Functions ##########
#######################################################
def calculate_metrics_and_plot(result_dict, plot_title, plotly_theme='simple_white',
                               export_html=False, export_metrics=False, export_file_name=None,
                               methods_to_plot=None, use_seaborn_plot=True):
    ''' Calculate metrics and plot results from multiple methods.
    Args:
        result_dict (dict): Dictionary where keys are method names and values are tuples of
                            (y_true, y_pred, time_steps).
        plot_title (str): Title for the plot.
        plotly_theme (str): Plotly theme for the plot.
        export_html (bool): Whether to export the plot as an HTML file.
        export_metrics (bool): Whether to append the calculated metrics to a CSV file
                               next to the exported files.
        export_file_name (str): Filename for the exported HTML/PNG/CSV file (if exporting).
        methods_to_plot (list or None): List of method names (dictionary keys) to plot together.
                                         If None, all methods are plotted.
        use_seaborn_plot (bool): If True, use seaborn lineplot for static PNG; if False, use matplotlib directly.
    '''
    metrics_dict = {}
    df_all = []
    # distinct_palette: use the same colors of matplotlib 'tab10' for consistency
    distinct_palette = itertools.cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    for method_name, (y_true, y_pred, time_steps) in result_dict.items():
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics_dict[method_name] = (rmse, mae, r2)

        #plotting the values using plotly
        df_plot = pd.DataFrame({
            "Timestamp": time_steps,
            "True": y_true,
            f"Pred - {method_name}": y_pred
        })

        # Melt for compatibility with Plotly colors
        df_melted = df_plot.melt(
            id_vars="Timestamp",
            var_name="Legend",
            value_name="H_bar Value"
        )
        df_melted["method"] = method_name
        df_all.append(df_melted)

    #displaying the values
    print("\n\n============== FINAL COMPARISON ==============")
    for name, (rmse, mae, r2) in metrics_dict.items():
        print(f"{name:20s} | RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

    ######################################################
    ################ plotting the data ###################
    ######################################################
    df_all = pd.concat(df_all, ignore_index=True)

    df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"])
    df_all = df_all.sort_values(by="Timestamp")

    # Filter df_all based on methods_to_plot (for both Plotly and PNG)
    if methods_to_plot is None:
        methods_to_plot = list(result_dict.keys())
    
    # Filter to include selected methods + True
    selected_legends = ['True'] + [f'Pred - {m}' for m in methods_to_plot if m in result_dict]
    df_plot_data = df_all[df_all['Legend'].isin(selected_legends)].copy()

    # ---- Plot everything together with Plotly ----
    # 1. Create the base figure
    fig_base = px.line(
        df_plot_data,
        x="Timestamp",
        y="H_bar Value",
        color="Legend",
        title=plot_title,
        labels={"H_bar Value": "H̅ (Water Level)"}
    )
    for trace in fig_base.data:
        trace.line.color = next(distinct_palette) # Cycle distinct colors
        trace.line.width = 2      # Thinner


    fig_base.update_layout(
        template=plotly_theme,
        legend_title_text="Series",
        height=550
    )

    # Wrap the figure with FigureResampler
    fig = FigureResampler(fig_base, default_n_shown_samples=1000)
    fig.show()
    if export_html:
        if export_file_name is None:
            raise ValueError("export_file_name must be provided when export_html is True.")
        os.makedirs(os.path.dirname(export_file_name) or '.', exist_ok=True)
        fig_base.write_html(f"{export_file_name}.html")

    # --- Save a static PNG using seaborn or matplotlib ---
    # Determine base filename and directory
    if export_file_name is not None:
        base_name = os.path.splitext(os.path.basename(export_file_name))[0]
        target_dir = os.path.dirname(export_file_name) or '.'
    else:
        # sanitize plot_title to create a safe filename
        base_name = ''.join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in plot_title).strip()
        target_dir = '.'

    png_path = os.path.join(target_dir, base_name + '.png')
    os.makedirs(target_dir, exist_ok=True)

    try:
        if use_seaborn_plot:
            # Use seaborn lineplot for better-looking plots
            plt.figure(figsize=(12, 5))
            sns.set_style("darkgrid")
            sns.lineplot(data=df_plot_data, x='Timestamp', y='H_bar Value', hue='Legend', style='Legend', palette='tab20', linewidth=2, errorbar=None)
            plt.title(plot_title, fontsize=14)
            plt.xlabel('Timestamp', fontsize=12)
            plt.ylabel('H̅ (Water Level)', fontsize=12)
            plt.legend(title='Series', fontsize=10)
            plt.tight_layout()
        else:
            # Use matplotlib directly
            plt.figure(figsize=(12, 5))
            for legend in df_plot_data['Legend'].unique():
                sub = df_plot_data[df_plot_data['Legend'] == legend]
                plt.plot(sub['Timestamp'], sub['H_bar Value'], label=legend)
            plt.title(plot_title)
            plt.xlabel('Timestamp')
            plt.ylabel('H̅ (Water Level)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        if export_file_name is not None:
            plt.savefig(png_path, dpi=150)
            print(f"Saved PNG: {png_path}")
        plt.show()
    except Exception as e:
        print(f"Warning: could not save PNG to {png_path}: {e}")
            
    # --- Optionally export metrics to CSV (append mode) ---
    if export_metrics:
        # Determine base filename and directory
        if export_file_name is not None:
            base_name = os.path.splitext(os.path.basename(export_file_name))[0]
            target_dir = os.path.dirname(export_file_name) or '.'
        else:
            # sanitize plot_title to create a safe filename
            base_name = ''.join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in plot_title).strip()
            target_dir = '.'

        if export_file_name is None:
            raise ValueError("export_file_name must be provided when export_metrics is True.")
        os.makedirs(target_dir, exist_ok=True)
        csv_path = os.path.join(target_dir, base_name + '.csv')
        # Use the already-computed metrics_dict rather than recalculating
        rows = []
        for method_name, (rmse, mae, r2) in metrics_dict.items():
            rows.append({
                'export_time': pd.Timestamp.now(),
                'method': method_name,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })
        df_metrics = pd.DataFrame(rows)
        write_header = not os.path.exists(csv_path)
        try:
            df_metrics.to_csv(csv_path, mode='a', header=write_header, index=False)
            print(f"Appended metrics to CSV: {csv_path}")
        except Exception as e:
            print(f"Warning: could not write metrics CSV to {csv_path}: {e}")


def peak_analysis(result_dict, peak_quantile=0.90, n_peaks=5, window_hours=72,
                  export_file_name=None):
    """
    Peak-aware evaluation: computes metrics restricted to flood-peak timesteps
    and renders zoomed plots around the top-N peaks.

    Args:
        result_dict:    same dict as calculate_metrics_and_plot — {name: (y_true, y_pred, timestamps)}
        peak_quantile:  threshold quantile on y_true to define a "peak" timestep (default 0.90)
        n_peaks:        number of top peaks to zoom into (default 5)
        window_hours:   hours on each side of a peak to include in the zoom plot (default 72)
        export_file_name: base path for PNG/CSV export (optional)
    """
    # Use the first entry's y_true as reference for peak threshold
    ref_name   = next(iter(result_dict))
    y_true_ref, _, ts_ref = result_dict[ref_name]
    ts_ref = pd.to_datetime(ts_ref)

    threshold = np.quantile(y_true_ref, peak_quantile)
    print(f"Peak threshold ({int(peak_quantile*100)}th pct): {threshold:.4f}")

    # ── Per-method peak metrics ───────────────────────────────────────────────
    print(f"\n{'Method':<25} {'Peak-RMSE':>10} {'Peak-MAE':>10} {'Peak-R²':>8}")
    print("-" * 57)
    peak_metrics = {}
    for name, (y_true, y_pred, ts) in result_dict.items():
        mask = y_true >= threshold
        if mask.sum() == 0:
            continue
        p_rmse = root_mean_squared_error(y_true[mask], y_pred[mask])
        p_mae  = mean_absolute_error(y_true[mask], y_pred[mask])
        p_r2   = r2_score(y_true[mask], y_pred[mask])
        peak_metrics[name] = (p_rmse, p_mae, p_r2)
        print(f"{name:<25} {p_rmse:>10.4f} {p_mae:>10.4f} {p_r2:>8.4f}")

    # Export peak metrics CSV
    if export_file_name is not None:
        os.makedirs(os.path.dirname(export_file_name) or '.', exist_ok=True)
        rows = [{'method': n, 'peak_rmse': r, 'peak_mae': m, 'peak_r2': r2,
                 'peak_quantile': peak_quantile}
                for n, (r, m, r2) in peak_metrics.items()]
        pd.DataFrame(rows).to_csv(f"{export_file_name}_peak_metrics.csv", index=False)
        print(f"\nSaved peak metrics: {export_file_name}_peak_metrics.csv")

    # ── Find top-N peaks in reference y_true ─────────────────────────────────
    peak_mask  = y_true_ref >= threshold
    peak_vals  = y_true_ref[peak_mask]
    peak_ts    = ts_ref[peak_mask]

    # Group nearby peaks — take the maximum within each contiguous block
    df_peaks = pd.DataFrame({'val': peak_vals, 'ts': peak_ts}).sort_values('ts')
    df_peaks['gap'] = df_peaks['ts'].diff() > pd.Timedelta(hours=window_hours)
    df_peaks['group'] = df_peaks['gap'].cumsum()
    top_peaks = (df_peaks.groupby('group')
                          .apply(lambda g: g.loc[g['val'].idxmax()])
                          .nlargest(n_peaks, 'val'))

    print(f"\nTop-{n_peaks} peaks:")
    for _, row in top_peaks.iterrows():
        print(f"  {row['ts']}  H_bar={row['val']:.4f}")

    # ── Zoom plots around each peak ───────────────────────────────────────────
    delta = pd.Timedelta(hours=window_hours)
    n_methods = len(result_dict)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78']

    for peak_idx, (_, peak_row) in enumerate(top_peaks.iterrows()):
        center = peak_row['ts']
        t0, t1 = center - delta, center + delta

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.set_title(f"Peak #{peak_idx+1}  |  {center.date()}  |  H_bar={peak_row['val']:.2f}",
                     fontsize=13)

        # Plot ground truth once (from reference)
        ref_y_true, _, ref_ts = result_dict[ref_name]
        ref_ts_pd = pd.to_datetime(ref_ts)
        win = (ref_ts_pd >= t0) & (ref_ts_pd <= t1)
        ax.plot(ref_ts_pd[win], ref_y_true[win],
                color='black', linewidth=2.5, label='True', zorder=10)

        # Plot each method's prediction
        for i, (name, (y_true, y_pred, ts)) in enumerate(result_dict.items()):
            ts_pd = pd.to_datetime(ts)
            win   = (ts_pd >= t0) & (ts_pd <= t1)
            ax.plot(ts_pd[win], y_pred[win],
                    color=colors[i % len(colors)], linewidth=1.4,
                    linestyle='--', alpha=0.85, label=name)

        # Mark the peak
        ax.axvline(center, color='red', linestyle=':', linewidth=1.2, alpha=0.6)
        ax.axhline(threshold, color='grey', linestyle=':', linewidth=1, alpha=0.5,
                   label=f'threshold ({int(peak_quantile*100)}th pct)')

        ax.set_xlabel('Timestamp')
        ax.set_ylabel('H̅ (Water Level)')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if export_file_name is not None:
            path = f"{export_file_name}_peak{peak_idx+1}.png"
            plt.savefig(path, dpi=150)
            print(f"Saved: {path}")
        plt.show()


def threshold_sweep_analysis(result_dict, n_thresholds=50,
                             quantile_range=(0.50, 0.99),
                             key_quantiles=(0.50, 0.75, 0.90, 0.95, 0.99),
                             min_samples=10,
                             export_file_name=None):
    """
    Threshold sweep: how does prediction quality change as we focus on increasingly
    extreme flood events?

    For each of N thresholds spanning the quantile_range:
      - Regression metrics (Bias, MAE, RMSE, R², Rel.Bias%) on y_true >= threshold
      - Classification metrics (Hit-Rate, False-Alarm) for predicting "is this a peak?"

    Outputs:
      - 6-panel plot — each metric vs threshold, one line per method
      - CSV table — full sweep [method, quantile, h_bar_threshold, N, ...metrics]
      - Console summary at key_quantiles (default 50/75/90/95/99)

    Args:
        result_dict:    {name: (y_true, y_pred, timestamps)}
        n_thresholds:   number of sweep points (default 50)
        quantile_range: (low, high) quantile bounds (default 0.50–0.99)
        key_quantiles:  quantiles to print in console summary table
        min_samples:    skip metric computation if N below this (default 10)
        export_file_name: base path for PNG/CSV export
    """
    # Use first entry as reference for threshold values (consistent across methods)
    ref_name = next(iter(result_dict))
    y_true_ref, _, _ = result_dict[ref_name]
    y_true_ref = np.array(y_true_ref)

    # Generate threshold quantiles and corresponding H_bar values
    sweep_qs = np.linspace(quantile_range[0], quantile_range[1], n_thresholds)
    sweep_thresholds = np.quantile(y_true_ref, sweep_qs)

    # Build full sweep table
    rows = []
    for name, (y_true, y_pred, _) in result_dict.items():
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        for q, thr in zip(sweep_qs, sweep_thresholds):
            mask_true  = y_true >= thr
            mask_pred  = y_pred >= thr
            n_peaks    = int(mask_true.sum())

            # Classification (always computable)
            tp = int((mask_true & mask_pred).sum())
            fn = int((mask_true & ~mask_pred).sum())
            fp = int((~mask_true & mask_pred).sum())
            tn = int((~mask_true & ~mask_pred).sum())
            hit_rate    = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            false_alarm = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            precision   = tp / (tp + fp) if (tp + fp) > 0 else np.nan

            # Regression on peak subset
            if n_peaks >= min_samples:
                bias = float((y_pred[mask_true] - y_true[mask_true]).mean())
                mae  = mean_absolute_error(y_true[mask_true], y_pred[mask_true])
                rmse = root_mean_squared_error(y_true[mask_true], y_pred[mask_true])
                # R² only meaningful if there's variance in the subset
                if y_true[mask_true].std() > 1e-6 and n_peaks >= 30:
                    r2 = r2_score(y_true[mask_true], y_pred[mask_true])
                else:
                    r2 = np.nan
                rel_bias = bias / max(y_true[mask_true].mean(), 1e-6) * 100
            else:
                bias = mae = rmse = r2 = rel_bias = np.nan

            rows.append({
                'method':         name,
                'quantile':       q,
                'h_bar_threshold': thr,
                'N':              n_peaks,
                'Bias':           bias,
                'MAE':            mae,
                'RMSE':           rmse,
                'R2':             r2,
                'Rel_Bias_pct':   rel_bias,
                'Hit_Rate':       hit_rate,
                'False_Alarm':    false_alarm,
                'Precision':      precision,
            })

    df_sweep = pd.DataFrame(rows)

    # ── Console summary at key quantiles ────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"Threshold Sweep Summary — Key Quantiles")
    print(f"{'='*100}")
    for q in key_quantiles:
        thr_q = np.quantile(y_true_ref, q)
        n_peaks = int((y_true_ref >= thr_q).sum())
        print(f"\nQuantile {q:.2f}  |  H_bar ≥ {thr_q:.2f}  |  N={n_peaks} peak timesteps")
        print(f"{'Method':<28} {'Bias':>8} {'MAE':>8} {'RMSE':>8} {'R²':>6} "
              f"{'Hit-Rate':>9} {'FA-Rate':>8}")
        print("-" * 88)
        # Find the closest sweep point to this key quantile
        for name in result_dict:
            sub = df_sweep[(df_sweep['method'] == name) &
                           (np.isclose(df_sweep['quantile'],
                                       df_sweep['quantile'].iloc[(df_sweep['quantile']-q).abs().argmin()]))]
            if sub.empty:
                continue
            r = sub.iloc[0]
            r2_str = f"{r['R2']:.3f}" if not np.isnan(r['R2']) else "  n/a"
            print(f"{name:<28} {r['Bias']:>+8.2f} {r['MAE']:>8.2f} {r['RMSE']:>8.2f} "
                  f"{r2_str:>6} {r['Hit_Rate']:>9.2%} {r['False_Alarm']:>8.2%}")

    # ── CSV export ──────────────────────────────────────────────────────────────
    if export_file_name is not None:
        os.makedirs(os.path.dirname(export_file_name) or '.', exist_ok=True)
        csv_path = f"{export_file_name}_threshold_sweep.csv"
        df_sweep.to_csv(csv_path, index=False)
        print(f"\nSaved full sweep table: {csv_path}")

    # ── 6-panel plot ────────────────────────────────────────────────────────────
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Threshold Sweep — Performance vs Peak Severity', fontsize=14)

    panels = [
        ('RMSE',        'RMSE',         'lower is better'),
        ('MAE',         'MAE',          'lower is better'),
        ('Bias',        'Bias (pred − true)', 'closer to 0 is better'),
        ('R2',          'R²',           'higher is better'),
        ('Hit_Rate',    'Hit Rate (recall)', 'higher is better'),
        ('False_Alarm', 'False Alarm Rate', 'lower is better'),
    ]

    for ax, (col, ylabel, hint) in zip(axes.flat, panels):
        for i, name in enumerate(result_dict):
            sub = df_sweep[df_sweep['method'] == name]
            ax.plot(sub['h_bar_threshold'], sub[col],
                    color=colors[i % len(colors)],
                    linewidth=2, label=name, marker='.', markersize=4)
        if col == 'Bias':
            ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('H̅ threshold')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel}  —  {hint}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')

    plt.tight_layout()
    if export_file_name is not None:
        path = f"{export_file_name}_threshold_sweep.png"
        plt.savefig(path, dpi=150)
        print(f"Saved plot: {path}")
    plt.show()

    return df_sweep


def diagnostic_plot(result_dict, h_bar_quantiles=(0.5, 0.75, 0.90, 0.95),
                    export_file_name=None):
    """
    Bias analysis plots for each method in result_dict.

    Panel 1 — Scatter (y_true vs y_pred) with 1:1 reference line.
              Colour-codes points by H_bar level (calm / moderate / high / peak).
    Panel 2 — Residual (y_pred − y_true) vs y_true level.
              Horizontal red line at 0; running mean overlaid.
    Panel 3 — Bias statistics table by H_bar quantile band.

    No leakage — uses only the already-generated predictions.
    """
    n_methods = len(result_dict)
    quantile_labels = [f"<{int(q*100)}th" for q in h_bar_quantiles]

    for name, (y_true, y_pred, _) in result_dict.items():
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        residual = y_pred - y_true

        # Quantile bands for colouring
        thresholds = np.quantile(y_true, h_bar_quantiles)
        bands = np.zeros(len(y_true), dtype=int)
        for i, thr in enumerate(thresholds):
            bands[y_true >= thr] = i + 1
        band_labels = ['calm'] + [f'≥{int(q*100)}th' for q in h_bar_quantiles]
        colors_band = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Bias Analysis — {name}", fontsize=13)

        # --- Panel 1: Scatter y_true vs y_pred ---
        ax = axes[0]
        for b, (label, col) in enumerate(zip(band_labels, colors_band)):
            mask = bands == b
            ax.scatter(y_true[mask], y_pred[mask], s=4, alpha=0.4,
                       color=col, label=label)
        lo, hi = y_true.min(), y_true.max()
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, label='1:1')
        ax.set_xlabel('y_true (H̅)')
        ax.set_ylabel('y_pred (H̅)')
        ax.set_title('Scatter: pred vs true')
        ax.legend(fontsize=7, markerscale=3)
        ax.grid(True, alpha=0.3)

        # --- Panel 2: Residual vs y_true ---
        ax = axes[1]
        for b, (label, col) in enumerate(zip(band_labels, colors_band)):
            mask = bands == b
            ax.scatter(y_true[mask], residual[mask], s=4, alpha=0.4, color=col)
        ax.axhline(0, color='red', linewidth=1.5, linestyle='--')
        # Running mean sorted by y_true
        sort_idx = np.argsort(y_true)
        win = max(1, len(y_true) // 50)
        running_mean = np.convolve(residual[sort_idx], np.ones(win)/win, mode='same')
        ax.plot(y_true[sort_idx], running_mean, color='black', linewidth=2, label='running mean')
        ax.set_xlabel('y_true (H̅)')
        ax.set_ylabel('y_pred − y_true (residual)')
        ax.set_title('Residual vs level')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Panel 3: Bias stats table by band ---
        ax = axes[2]
        ax.axis('off')
        rows = []
        for b, label in enumerate(band_labels):
            mask = bands == b
            if mask.sum() == 0:
                continue
            n    = mask.sum()
            bias = residual[mask].mean()
            mae  = np.abs(residual[mask]).mean()
            rel  = bias / np.maximum(y_true[mask].mean(), 1e-6) * 100
            rows.append([label, f"{n}", f"{bias:+.2f}", f"{mae:.2f}", f"{rel:+.1f}%"])

        overall_bias = residual.mean()
        overall_mae  = np.abs(residual).mean()
        rows.append(['ALL', f"{len(y_true)}", f"{overall_bias:+.2f}",
                     f"{overall_mae:.2f}", f"{overall_bias/max(y_true.mean(),1e-6)*100:+.1f}%"])

        col_labels = ['Band', 'N', 'Bias\n(pred−true)', 'MAE', 'Rel.Bias']
        tbl = ax.table(cellText=rows, colLabels=col_labels,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.1, 1.6)
        ax.set_title('Bias by H̅ level', pad=12)

        plt.tight_layout()
        if export_file_name is not None:
            path = f"{export_file_name}_{name.replace(' ', '_')}_diagnostic.png"
            os.makedirs(os.path.dirname(export_file_name) or '.', exist_ok=True)
            plt.savefig(path, dpi=150)
            print(f"Saved: {path}")
        plt.show()


def export_results_to_csv(result_dict, export_file_name):
    ''' Export results (timestamp, y_true, y_predictions) to separate CSV files per method.
    
    Args:
        result_dict (dict): Dictionary where keys are method names and values are tuples of
                            (y_true, y_pred, time_steps).
        export_file_name (str): Base filename for the exported CSV files (with or without .csv extension).
                                Method name will be appended before the .csv extension.
    
    Behavior:
        Creates a separate CSV file for each method with columns:
        - Timestamp: index derived from time_steps
        - True: ground truth values
        - Prediction: predictions for the method
        
        Saves files as: <export_file_name>_<method_name>.csv
    '''
    # Remove .csv extension if present to avoid duplication
    if export_file_name.endswith('.csv'):
        base_name = export_file_name[:-4]
    else:
        base_name = export_file_name
    
    # Save a separate CSV for each method
    for method_name, (y_true, y_pred, time_steps) in result_dict.items():
        df = pd.DataFrame({
            'Timestamp': time_steps,
            'True': y_true,
            'Prediction': y_pred
        })
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp').sort_index()
        
        csv_filename = f"{base_name}_{method_name}.csv"
        
        try:
            df.to_csv(csv_filename)
            print(f"Exported results to CSV: {csv_filename}")
        except Exception as e:
            print(f"Warning: could not write results to CSV {csv_filename}: {e}")
