import os
import sys
import time
import logging
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim

# Set the backend to PyTorch before importing deepxde
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde

import random
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler

# Matplotlib settings for fonts and style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = '12'

def parse_arguments():
    """
    Parse command-line arguments for training configuration.
    """
    parser = argparse.ArgumentParser(description='Train DeepONet with deepxde on 3D point cloud data.')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('-r', '--RUN', type=int, default=0, help='Run number')
    parser.add_argument('-bic', '--branch_input_components', type=str, default='mlc', help='Branch net input components')
    parser.add_argument('-tic', '--trunk_input_components', type=str, default='xyzd', help='Trunk net input components')
    parser.add_argument('-oc', '--output_components', type=str, default='xyzs', help='Output components')
    parser.add_argument('-N_p', '--N_pt', type=int, default=5000, help='Number of points per sample')
    parser.add_argument('-i', '--N_iterations', type=int, default=150000, help='Number of training iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-dir', '--dir_base_load_data', type=str, default='../data/sampled', help='Data directory to load from')
    parser.add_argument('-N_d', '--N_samples', type=int, default=3000, help='Number of samples')
    parser.add_argument('-sm', '--split_method', type=str, choices=['mass', 'random'], default='random', help='Data split method')
    parser.add_argument('--base_dir', type=str, default='../experiments', help='Base directory to save experiment results')
    args = parser.parse_args()
    return args

def set_random_seed():
    """
    Set a fixed random seed for reproducibility.
    """
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging(args, name):
    """
    Create a directory for the experiment and configure logging.
    """
    experiment_dir = os.path.join(args.base_dir, name)
    os.makedirs(experiment_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', 
                        handlers=[logging.FileHandler(f"{experiment_dir}/training_log.txt"), logging.StreamHandler(sys.stdout)])
    return experiment_dir

def get_device(gpu):
    """
    Select the computing device (GPU or CPU).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_parameters(args, device, name):
    """
    Log the parameters of the current run.
    """
    logging.info('\n\nModel parameters:')
    logging.info(f'Device: {device}')
    logging.info(f'Run number: {args.RUN}')
    logging.info(f'Folder name: {name}')
    logging.info(f'Iterations: {args.N_iterations}')
    logging.info(f'Batch Size: {args.batch_size}')
    logging.info(f'Learning Rate: {args.learning_rate}')
    logging.info(f'Number of Samples: {args.N_samples}')
    logging.info(f'Data Split Method: {args.split_method}')
    logging.info(f'Branch input components: {args.branch_input_components}')
    logging.info(f'Trunk input components: {args.trunk_input_components}')
    logging.info(f'Output components: {args.output_components}\n\n')

def process_branch_input(input_components, tmp):
    """
    Extract branch net input features from the loaded data.
    The branch input is taken from a single point per sample.
    """
    input_mapping = {'m': 4, 'l': 5, 'c': [6, 7, 8]}
    selected_indices = [input_mapping[comp] for comp in input_components if comp in input_mapping]
    # Flatten list if there are multiple indices (e.g., for 'c')
    selected_indices = [i for idx in selected_indices for i in (idx if isinstance(idx, list) else [idx])]
    branch_data = tmp['a'][:, 0, selected_indices].astype(np.float32)
    return branch_data

def process_trunk_input(input_components, tmp):
    """
    Extract trunk net input features from the loaded data.
    Trunk input is taken from all points of each sample.
    """
    input_mapping = {'x': 0, 'y': 1, 'z': 2, 'd': 3, 'm': 4, 'l': 5, 'c': [6, 7, 8]}
    selected_indices = []
    for comp in input_components:
        if comp == 'c':
            selected_indices.extend(input_mapping[comp])
        elif comp in input_mapping:
            selected_indices.append(input_mapping[comp])
    trunk_input = tmp['a'][:, :, selected_indices].astype(np.float32)
    return trunk_input

def get_clipping_ranges_for_direction(direction):
    """
    Define direction-specific clipping ranges for each output component to ensure consistent data scaling.
    """
    if direction == 'ver':
        return {0: (-0.068, 0.473), 1: (-0.093, 0.073), 2: (-0.003, 0.824), 3: (0., 232.19)}
    elif direction == 'hor':
        return {0: (-0.421, 0.008), 1: (-0.024, 0.029), 2: (-0.388, 0.109), 3: (0., 227.78)}
    elif direction == 'dia':
        return {0: (-0.079, 0.016), 1: (-0.057, 0.056), 2: (-0.006, 0.214), 3: (0., 172.19)}
    else:
        raise ValueError(f"Unknown direction {direction}")

def calculate_r2(true_vals, pred_vals):
    """
    Compute the R² (coefficient of determination) score.
    """
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    return 1 - ss_res / ss_tot

def process_output(output_components, output_vals, identifiers):
    """
    Select output components and apply direction-specific clipping.
    """
    output_mapping = {'x': 0, 'y': 1, 'z': 2, 's': 3}
    selected_indices = [output_mapping[comp] for comp in output_components]
    combined_output = output_vals[:, :, selected_indices]

    unique_directions = set(id.split('_')[0] for id in identifiers)
    clipping_ranges_dict = {direction: get_clipping_ranges_for_direction(direction) for direction in unique_directions}
    for direction, clipping_ranges in clipping_ranges_dict.items():
        indices = [i for i, id in enumerate(identifiers) if id.split('_')[0] == direction]
        if indices:
            indices = np.array(indices)
            for j, idx in enumerate(selected_indices):
                min_val, max_val = clipping_ranges[idx]
                combined_output[indices, :, j] = np.clip(combined_output[indices, :, j], min_val, max_val)

    if combined_output.shape[-1] == 1:
        combined_output = combined_output.squeeze(-1)
    return combined_output

def map_keys_to_indices(keys, all_keys):
    """
    Map a list of keys to their corresponding indices in the all_keys array.
    """
    return [np.where(all_keys == key)[0][0] for key in keys]

def load_and_preprocess_data(args, dir_base_save_model):
    """
    Load data from file, extract branch/trunk inputs and outputs, apply scaling, and prepare train/test splits.
    """
    tmp = np.load(f'{args.dir_base_load_data}/Rpt{str(args.RUN)}_N{str(args.N_pt)}.npz')

    # Prepare inputs
    input_data_branch = process_branch_input(args.branch_input_components, tmp)
    input_data_trunk = process_trunk_input(args.trunk_input_components, tmp)

    # Prepare outputs with clipping
    combined_output = process_output(args.output_components, tmp['b'], tmp['c'])

    # Define scalers with a specific feature range
    scaler_fun = partial(MinMaxScaler, feature_range=(-1, 1))
    combined_scaler = scaler_fun()
    combined_scaler.fit(input_data_branch)
    input_data_branch = combined_scaler.transform(input_data_branch)

    trunk_scaler = scaler_fun()
    ss = input_data_trunk.shape
    tmp_input_trunk = input_data_trunk.reshape([ss[0]*ss[1], ss[2]])
    trunk_scaler.fit(tmp_input_trunk)
    input_data_trunk = trunk_scaler.transform(tmp_input_trunk).reshape(ss)

    output_scalers = scaler_fun()
    ss = combined_output.shape
    tmp_output = combined_output.reshape([ss[0]*ss[1], ss[2]])
    output_scalers.fit(tmp_output)
    combined_output = output_scalers.transform(tmp_output).reshape(ss)

    # Save scalers
    pickle.dump(combined_scaler, open(f'{dir_base_save_model}/combined_scaler', 'wb'))
    pickle.dump(trunk_scaler, open(f'{dir_base_save_model}/trunk_scaler', 'wb'))
    pickle.dump(output_scalers, open(f'{dir_base_save_model}/output_scaler', 'wb'))

    # Load split (train/valid) keys
    if args.split_method == 'random':
        split_file = f'../data/npy/combined_{args.N_samples}_split_random_train_valid.npz'
    else:
        split_file = f'../data/npy/combined_{args.N_samples}_split_mass_train_valid.npz'
    split_data = np.load(split_file)

    train_case = map_keys_to_indices(split_data['train'], np.array([key for key in tmp['c']]))
    test_case = map_keys_to_indices(split_data['valid'], np.array([key for key in tmp['c']]))

    train_case_file = tmp['c'][train_case]
    test_case_file = tmp['c'][test_case]

    branch_input_train = input_data_branch[train_case].astype(np.float32)
    branch_input_test = input_data_branch[test_case].astype(np.float32)
    trunk_input_train = input_data_trunk[train_case].astype(np.float32)
    trunk_input_test = input_data_trunk[test_case].astype(np.float32)
    s_train = combined_output[train_case].astype(np.float32)
    s_test = combined_output[test_case].astype(np.float32)

    logging.info(f'branch_input_train.shape = {branch_input_train.shape}')
    logging.info(f'branch_input_test.shape = {branch_input_test.shape}')
    logging.info(f'trunk_input_train.shape = {trunk_input_train.shape}')
    logging.info(f'trunk_input_test.shape = {trunk_input_test.shape}')
    logging.info(f's_train.shape = {s_train.shape}')
    logging.info(f's_test.shape = {s_test.shape}')

    x_train = (branch_input_train, trunk_input_train)
    y_train = s_train
    x_test = (branch_input_test, trunk_input_test)
    y_test = s_test

    return x_train, y_train, x_test, y_test, output_scalers, train_case_file, test_case_file

class TripleCartesianProd(Data):
    """
    Custom data class implementing the deepxde Data interface.
    Provides train/test batching for DeepONet training.
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test
        self.num_train_samples = self.train_x[0].shape[0]
        self.sampler = BatchSampler(self.num_train_samples, shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(outputs, targets)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.sampler.get_next(batch_size)
        return ((self.train_x[0][indices], self.train_x[1][indices]), self.train_y[indices],)

    def test(self):
        return self.test_x, self.test_y

class DeepONetCartesianProd(dde.maps.NN, nn.Module):
    def __init__(self, branch_net, trunk_net, kernel_initializer, regularization=None, num_outputs=4):
        nn.Module.__init__(self)
        dde.maps.NN.__init__(self)
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.hidden_dim = 128
        self.b = nn.Parameter(torch.zeros(1))
        self.num_outputs = num_outputs  # Number of output components (e.g., ux, uy, uz, vm)

        # Attributes required by deepxde
        self.initializer = kernel_initializer
        self.regularizer = regularization

        self.tanh = nn.Tanh()

    def forward(self, inputs):
        # inputs is a tuple: (branch_input, trunk_input)
        # branch_input: (batch_size, input_dim_branch)
        # trunk_input: (batch_size, Npt, input_dim_trunk)

        branch_input = inputs[0]   # Shape: [batch_size, input_dim_branch]
        trunk_input = inputs[1]    # Shape: [batch_size, Npt, input_dim_trunk]

        # The branch network processes a single global vector per sample
        # Output of branch_net: (batch_size, hidden_dim)
        branch_output = self.branch_net(branch_input)  # [batch_size, hidden_dim]

        # The trunk network processes all points for each sample
        # First, reshape trunk_input to a 2D tensor for batch processing through trunk_net
        # trunk_input reshaped: (batch_size * Npt, input_dim_trunk)
        bs, Npt, input_dim = trunk_input.shape
        trunk_input = trunk_input.reshape(bs * Npt, input_dim)

        # Pass through the trunk_net
        # Output of trunk_net: (batch_size * Npt, hidden_dim * num_outputs)
        trunk_output = self.trunk_net(trunk_input)

        # Reshape trunk_output back to a 4D tensor: (batch_size, Npt, hidden_dim, num_outputs)
        trunk_output = trunk_output.reshape(bs, Npt, self.hidden_dim, self.num_outputs)

        # Now we have:
        # branch_output: (batch_size, hidden_dim)
        # trunk_output: (batch_size, Npt, hidden_dim, num_outputs)

        # We want to combine them via an inner product along the hidden_dim dimension.
        # Using einsum:
        # "bh,bnhc->bnc"
        # b = batch_size, h = hidden_dim, n = Npt, c = num_outputs
        # Result: (batch_size, Npt, num_outputs)
        output = torch.einsum("bh,bnhc->bnc", branch_output, trunk_output)  # [batch_size, Npt, num_outputs]

        # Add the bias parameter (broadcasted over batch and Npt dimensions)
        # output: (batch_size, Npt, num_outputs)
        output = output + self.b

        # Apply tanh activation
        # Final output: (batch_size, Npt, num_outputs)
        return self.tanh(output)

    def parameters(self):
        return (list(self.branch_net.parameters()) +
                list(self.trunk_net.parameters()) +
                [self.b])


def define_model(args, device, data, output_scalers):
    """
    Define the DeepONet model, optimizer, scheduler, and loss/metrics.
    """
    sample_branch_input = data.train_x[0]
    input_dim_branch = sample_branch_input.shape[1]
    activation_fn = nn.SiLU()
    num_outputs = 4

    # Branch network
    branch_net = nn.Sequential(
        nn.Linear(input_dim_branch, 128),
        activation_fn,
        nn.Linear(128, 128),
        activation_fn,
        nn.Linear(128, 256),
        activation_fn,
        nn.Linear(256, 128),
        activation_fn,
    ).to(device)
    logging.info('\n\nbranch_net:')
    logging.info(branch_net)

    # Trunk network
    sample_trunk_input = data.train_x[1]
    input_dim_trunk = sample_trunk_input.shape[-1]
    trunk_net = nn.Sequential(
        nn.Linear(input_dim_trunk, 128),
        activation_fn,
        nn.Linear(128, 128),
        activation_fn,
        nn.Linear(128, 256),
        activation_fn,
        nn.Linear(256, 128 * num_outputs),
        activation_fn,
    ).to(device)

    logging.info('\n\ntrunk_net:')
    logging.info(trunk_net)

    def loss_func(outputs, targets):
        return torch.mean((outputs - targets) ** 2)

    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError('Data must be torch.Tensor or np.ndarray')

    def inv(data, scaler):
        data_shape = data.shape
        data_flat = data.reshape(-1, data_shape[-1])
        data_inv = scaler.inverse_transform(data_flat)
        return data_inv.reshape(data_shape)

    def err_MAE(true_vals, pred_vals):
        return np.mean(np.abs(true_vals - pred_vals), axis=1)

    def metric_factory(idx):
        # Produces functions to compute MAE and R² for a specific component index
        def comp_MAE(outputs, targets):
            outputs_np = to_numpy(outputs)
            targets_np = to_numpy(targets)
            true_vals = inv(targets_np, output_scalers)[:, :, idx]
            pred_vals = inv(outputs_np, output_scalers)[:, :, idx]
            return np.mean(err_MAE(true_vals, pred_vals))

        def comp_R2(outputs, targets):
            outputs_np = to_numpy(outputs)
            targets_np = to_numpy(targets)
            true_vals = inv(targets_np, output_scalers)[:, :, idx].flatten()
            pred_vals = inv(outputs_np, output_scalers)[:, :, idx].flatten()
            return calculate_r2(true_vals, pred_vals)

        return comp_MAE, comp_R2

    # If output is 'xyzs', define metrics for ux, uy, uz, vm
    if args.output_components == 'xyzs':
        ux_MAE, ux_R2 = metric_factory(0)
        uy_MAE, uy_R2 = metric_factory(1)
        uz_MAE, uz_R2 = metric_factory(2)
        vm_MAE, vm_R2 = metric_factory(3)
        metrics = [ux_MAE, uy_MAE, uz_MAE, vm_MAE, ux_R2, uy_R2, uz_R2, vm_R2]
    else:
        metrics = []

    model = DeepONetCartesianProd(branch_net, trunk_net, "Glorot normal", num_outputs=num_outputs).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, verbose=True)

    model = dde.Model(data, model)
    model.compile(
        optimizer=optimizer,
        loss=loss_func,
        decay=("inverse time", 1, args.learning_rate/10.),
        metrics=metrics,
    )
    return model, scheduler, optimizer

def count_parameters(model):
    """
    Count the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.net.parameters() if p.requires_grad)

def train_model(model, scheduler, optimizer, args, output_scalers, experiment_dir, device):
    """
    Train the model using deepxde's training loop and save checkpoints/results.
    """
    callbacks = []
    torch.autograd.set_detect_anomaly(True)
    start_time = time.time()

    losshistory, train_state = model.train(
        epochs=args.N_iterations,
        batch_size=args.batch_size,
        display_every=100,
        model_save_path=f"{experiment_dir}/model_checkpoint.pth",
        callbacks=callbacks
    )
    total_training_time = time.time() - start_time
    scheduler.step(losshistory.loss_test[-1])
    torch.save(model.net.state_dict(), f'{experiment_dir}/model_final.pth')

    total_params = count_parameters(model)
    with open(f'{experiment_dir}/total_params.txt', 'w') as f:
        f.write(f"{total_params}")

    with open(os.path.join(experiment_dir, 'training_time.txt'), 'w') as f:
        f.write(f"{total_training_time:.2f}")

    np.save(f'{experiment_dir}/train_losses.npy', np.array(losshistory.loss_train))
    np.save(f'{experiment_dir}/val_losses.npy', np.array(losshistory.loss_test))
    return losshistory, train_state

def plot_loss_curves(losshistory, experiment_dir):
    """
    Plot and save the training and validation loss curves.
    """
    plt.figure()
    plt.plot(losshistory.loss_train, label='Train Loss')
    plt.plot(losshistory.loss_test, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss over Iterations')
    plt.savefig(f'{experiment_dir}/loss_curve.jpg', dpi=300)
    plt.close()

def plot_r2_scatter(true_vals, pred_vals, comp, dataset_type, experiment_dir):
    """
    Plot a scatter plot of predictions vs true values for a given component and dataset subset.
    """
    r2 = calculate_r2(true_vals, pred_vals)
    plt.figure(figsize=(6,6))
    plt.scatter(true_vals, pred_vals, alpha=0.3, s=10, label='Data Points')
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit')
    plt.xlabel(f'True {comp}')
    plt.ylabel(f'Predicted {comp}')
    plt.title(f'{dataset_type.capitalize()} Dataset: {comp} Prediction vs True\nR² = {r2:.3f}')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, f'{dataset_type}_scatter_{comp}.jpg'), dpi=300)
    plt.close()

def main():
    """
    Main entry point for training the DeepONet model.
    """
    args = parse_arguments()
    set_random_seed()
    device = get_device(args.gpu)
    name = f'DeepONet_RUN{args.RUN}_D{args.N_samples}_N{args.N_pt}_branchinput_{args.branch_input_components}_trunkinput_{args.trunk_input_components}_output_{args.output_components}_split_{args.split_method}'
    experiment_dir = setup_logging(args, name)
    log_parameters(args, device, name)

    x_train, y_train, x_test, y_test, output_scalers, train_case_file, test_case_file = load_and_preprocess_data(args, experiment_dir)
    data = TripleCartesianProd(x_train, y_train, x_test, y_test)
    model, scheduler, optimizer = define_model(args, device, data, output_scalers)

    logging.info(f'\nModel Structure:\n{model}')
    total_params = count_parameters(model)
    logging.info(f'Total number of trainable parameters: {total_params}\n')

    losshistory, train_state = train_model(model, scheduler, optimizer, args, output_scalers, experiment_dir, device)
    plot_loss_curves(losshistory, experiment_dir)

    test_outputs = model.predict(model.data.test_x)
    test_targets = model.data.test_y

    def inv(data, scaler):
        data_shape = data.shape
        data_flat = data.reshape(-1, data_shape[-1])
        data_inv = scaler.inverse_transform(data_flat)
        return data_inv.reshape(data_shape)

    test_outputs_inv = inv(test_outputs, output_scalers)
    test_targets_inv = inv(test_targets, output_scalers)
    components = ['ux', 'uy', 'uz', 'vm']
    directions = [tcf.split('_')[0] for tcf in test_case_file]
    unique_directions = list(set(directions))

    subset_metrics_all_dir = os.path.join(experiment_dir, 'subset_metrics_all')
    os.makedirs(subset_metrics_all_dir, exist_ok=True)

    subset_plots_dir = os.path.join(experiment_dir, 'subset_plots')
    os.makedirs(subset_plots_dir, exist_ok=True)

    for direction in unique_directions:
        indices = [i for i, dir_label in enumerate(directions) if dir_label == direction]
        if not indices:
            logging.warning(f"No test samples found for direction: {direction}")
            continue
        true_vals_dir = test_targets_inv[indices, :, :]
        pred_vals_dir = test_outputs_inv[indices, :, :]
        for i, comp in enumerate(components):
            true_comp = true_vals_dir[:, :, i].flatten()
            pred_comp = pred_vals_dir[:, :, i].flatten()
            mae = np.mean(np.abs(pred_comp - true_comp))
            rmse = np.sqrt(np.mean((pred_comp - true_comp)**2))
            r2 = calculate_r2(true_comp, pred_comp)
            with open(os.path.join(subset_metrics_all_dir, f'{direction}_{comp}_MAE.txt'), 'w') as f_mae:
                f_mae.write(f"{mae}")
            with open(os.path.join(subset_metrics_all_dir, f'{direction}_{comp}_RMSE.txt'), 'w') as f_rmse:
                f_rmse.write(f"{rmse}")
            with open(os.path.join(subset_metrics_all_dir, f'{direction}_{comp}_R2.txt'), 'w') as f_r2:
                f_r2.write(f"{r2:.3f}")
            logging.info(f"subset Metrics for {direction} {comp}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
            plot_r2_scatter(true_comp, pred_comp, comp, f'test_{direction}_{comp}', subset_plots_dir)

if __name__ == "__main__":
    main()
