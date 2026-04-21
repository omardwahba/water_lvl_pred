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
    A PyTorch Dataset that applies windowed normalization for each sample.
    For each (X, y) sample, it normalizes both X and y using the
    min/max statistics calculated *only* from the X (48h lookback) window.
    AND MIN/MAX for y normalization is computed using both the lookback X window
    and the future y window, to avoid data leakage.
    Args:
        X_seq (numpy arr or list): Input sequences of shape (num_samples, lookback, num_features).
        y_seq (numpy arr or list): Output sequences of shape (num_samples, horizon).
        h_bar_index (int): Index of the H_bar feature in the data.
        epsilon (float): Small constant for numerical stability.
    Returns:
        tuple: (x_scaled, y_scaled, h_bar_min, h_bar_range) for each sample.
    1. x_scaled: Normalized input sequence.
    2. y_scaled: Normalized output sequence.
    3. h_bar_min: Minimum value used for y normalization.
    4. h_bar_range: Range (max-min) used for y normalization.
    5. These last two are needed to unscale predictions later.
    """
    def __init__(self, X_seq, y_seq, h_bar_index, epsilon=EPSILON):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y_seq = torch.tensor(y_seq, dtype=torch.float32)
        self.h_bar_index = h_bar_index
        self.epsilon = epsilon

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        x_raw = self.X_seq[idx]  # Shape: (48, 4)
        y_raw = self.y_seq[idx]  # Shape: (24,)
        
        # 1. Get stats from the 48h lookback window (x_raw)
        # min/max per feature, shape (4,)
        min_vals, _ = torch.min(x_raw, dim=0) 
        max_vals, _ = torch.max(x_raw, dim=0)
        # Add epsilon for stability (to avoid division by zero if max==min)
        range_vals = max_vals - min_vals + self.epsilon
        x_scaled = (x_raw - min_vals) / range_vals
        
        # 3. Compute H_bar stats using both the lookback X window and the future y window
        #    so min/max for H_bar considers values from both x_raw[:, h_bar_index] and y_raw.
        y_min = torch.min(y_raw)
        y_max = torch.max(y_raw)

        h_bar_min = torch.min(min_vals[self.h_bar_index], y_min)
        h_bar_max = torch.max(max_vals[self.h_bar_index], y_max)
        h_bar_range = h_bar_max - h_bar_min + self.epsilon
        y_scaled = (y_raw - h_bar_min) / h_bar_range

        # Return everything needed for training and unscaling
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
                       use_amnesia_strategy=False, amnesia_threshold=2.5, amnesia_warmup_batches=10,
                       amnesia_new_lr=None,   # <-- NEW: LR to set *during* a spike
                       amnesia_reset_lr=None): # <-- NEW: The original LR to return to

    # --- Setup from original function ---
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_batches = len(train_loader)
    y_ground_truth, y_predictions = np.array([0.0]).reshape(1,1), np.array([0.0]).reshape(1,1)
    all_indices, batch_losses = np.array(['']), []

    # --- NEW: Setup for Amnesia Strategy ---
    running_avg_loss, ema_alpha = 0.0, 0.1
    amnesia_active = False # Flag to track if we're in a "spike" state

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
                    amnesia_active = True # Set flag
                    print(f"\n  *** AMNESIA TRIGGERED at batch {i+1} ***")
                    print(f"      Loss {current_loss:.6f} > Threshold ({spike_threshold:.6f})")
                    print(f"      Resetting optimizer state...")
                    optimizer.state = defaultdict(dict) # Reset optimizer
                    
                    # --- NEW: Set new learning rate ---
                    if amnesia_new_lr is not None:
                        print(f"      Setting LR to {amnesia_new_lr}")
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = amnesia_new_lr
                    
                    running_avg_loss = current_loss 
                
                # --- RESET AFTER SPIKE ---
                elif current_loss < running_avg_loss and amnesia_active:
                    amnesia_active = False # Clear flag
                    print(f"  *** AMNESIA RESET at batch {i+1} ***")
                    print(f"      Loss {current_loss:.6f} is back below average.")
                    
                    # --- NEW: Reset to original learning rate ---
                    if amnesia_reset_lr is not None:
                        print(f"      Resetting LR to {amnesia_reset_lr}")
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = amnesia_reset_lr
        
        # 3. Backward pass and optimize (Original)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f"   Batch {i+1}/{num_batches}, Loss: {current_loss:.6f}")
            
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
