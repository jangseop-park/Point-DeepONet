import os
import sys
import time
import logging
import argparse
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Matplotlib settings for fonts and style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = '12'

def parse_arguments():
    """
    Parse command-line arguments for training configuration.
    """
    parser = argparse.ArgumentParser(description='Train PointNet on 3D point cloud data.')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('-r', '--RUN', type=int, default=0, help='Run number')
    parser.add_argument('-ic', '--input_components', type=str, default='xyzmlc', help='Input features (e.g., xyz, xyzd, xyzdmlc)')
    parser.add_argument('-oc', '--output_components', type=str, default='xyzs', help='Output features (e.g., x, y, z, s, xyz)')
    parser.add_argument('-N_p', '--N_pt', type=int, default=5000, help='Number of points per sample')
    parser.add_argument('-i', '--N_iterations', type=int, default=4000, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('-dir', '--dir_base_load_data', type=str, default='../data/sampled', help='Directory to load data from')
    parser.add_argument('-N_d', '--N_samples', type=int, default=3000, help='Number of samples')
    parser.add_argument('--split_method', type=str, choices=['mass', 'random'], default='random', help='Data split method')
    parser.add_argument('--scaling', type=float, default=0.53, help='Scaling factor for network channels')
    parser.add_argument('--base_dir', type=str, default='../experiments', help='Base directory for experiment results')
    args = parser.parse_args()
    return args

def setup_logging(args, name):
    """
    Set up logging to both console and file.
    """
    experiment_dir = os.path.join(args.base_dir, name)
    os.makedirs(experiment_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.FileHandler(f"{experiment_dir}/training_log.txt"),
        logging.StreamHandler(sys.stdout)
    ])
    return experiment_dir

def get_device(gpu):
    """
    Select the appropriate device (GPU or CPU) for training.
    """
    return torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

def log_parameters(args, device, name):
    """
    Log the parameters of the current run.
    """
    logging.info('\n\nModel parameters:')
    logging.info(f'Using device: {device}')
    logging.info(f'Run number: {args.RUN}')
    logging.info(f'Experiment name: {name}')
    logging.info(f'Iterations (epochs): {args.N_iterations}')
    logging.info(f'Batch Size: {args.batch_size}')
    logging.info(f'Learning Rate: {args.learning_rate}')
    logging.info(f'Number of Samples: {args.N_samples}')
    logging.info(f'Data Split Method: {args.split_method}')
    logging.info(f'Input components: {args.input_components}')
    logging.info(f'Output components: {args.output_components}')
    logging.info(f'Scaling factor: {args.scaling}\n\n')

def get_clipping_ranges_for_direction(direction):
    """
    Return predefined min/max clipping values for each output component based on direction.
    """
    if direction == 'ver':
        return {
            0: (-0.068, 0.473),
            1: (-0.093, 0.073),
            2: (-0.003, 0.824),
            3: (0., 232.19),
        }
    elif direction == 'hor':
        return {
            0: (-0.421, 0.008),
            1: (-0.024, 0.029),
            2: (-0.388, 0.109),
            3: (0., 227.78),
        }
    elif direction == 'dia':
        return {
            0: (-0.079, 0.016),
            1: (-0.057, 0.056),
            2: (-0.006, 0.214),
            3: (0., 172.19),
        }
    else:
        raise ValueError(f"Unknown direction: {direction}")

def process_input(input_components, tmp):
    """
    Select specified input components from the loaded data.
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
        if not indices:
            continue
        indices = np.array(indices)
        for j, idx in enumerate(selected_indices):
            min_val, max_val = clipping_ranges[idx]
            combined_output[indices, :, j] = np.clip(combined_output[indices, :, j], min_val, max_val)

    if combined_output.shape[-1] == 1:
        combined_output = combined_output.squeeze(-1)
    return combined_output

def map_keys_to_indices(keys, all_keys):
    """
    Map a list of keys to their corresponding indices in all_keys.
    """
    return [np.where(all_keys == key)[0][0] for key in keys]

def load_and_preprocess_data(args, dir_base_save_model):
    """
    Load data, select components, apply scaling, and prepare train/test splits.
    """
    tmp = np.load(f'{args.dir_base_load_data}/Rpt{str(args.RUN)}_N{str(args.N_pt)}.npz')

    combined_input = process_input(args.input_components, tmp)
    combined_output = process_output(args.output_components, tmp['b'], tmp['c'])

    # Scale inputs
    combined_scalers = MinMaxScaler()
    combined_input = combined_scalers.fit_transform(combined_input.reshape(-1, combined_input.shape[-1])).reshape(combined_input.shape)
    
    # Scale outputs
    output_scalers = MinMaxScaler()
    combined_output = output_scalers.fit_transform(combined_output.reshape(-1, combined_output.shape[-1])).reshape(combined_output.shape)
    
    # Save scalers
    pickle.dump(combined_scalers, open(f'{dir_base_save_model}/combined_scaler.pkl', 'wb'))
    pickle.dump(output_scalers, open(f'{dir_base_save_model}/output_scaler.pkl', 'wb'))
    
    # Determine split file
    if args.split_method == 'random':
        split_file = f'../data/npy/combined_{args.N_samples}_split_random_train_valid.npz'
    else:
        split_file = f'../data/npy/combined_{args.N_samples}_split_mass_train_valid.npz'

    # Load train/valid splits
    split_data = np.load(split_file)
    train_case = map_keys_to_indices(split_data['train'], np.array([key for key in tmp['c']]))
    test_case = map_keys_to_indices(split_data['valid'], np.array([key for key in tmp['c']]))

    train_case_file = tmp['c'][train_case]
    test_case_file = tmp['c'][test_case]

    input_training = combined_input[train_case].astype(np.float32)
    input_test = combined_input[test_case].astype(np.float32)
    output_training = combined_output[train_case].astype(np.float32)
    output_test = combined_output[test_case].astype(np.float32)

    logging.info(f'\n\ninput_training.shape: {input_training.shape}')
    logging.info(f'input_test.shape: {input_test.shape}')
    logging.info(f'output_training.shape: {output_training.shape}')
    logging.info(f'output_test.shape: {output_test.shape}')
    
    return input_training, output_training, input_test, output_test, output_scalers, train_case_file, test_case_file

class PointCloudDataset(Dataset):
    """
    Custom PyTorch Dataset for point cloud data.
    """
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float32), torch.tensor(self.outputs[idx], dtype=torch.float32)

def create_dataloaders(input_training, output_training, input_test, output_test, batch_size):
    """
    Create DataLoaders for training and testing.
    """
    num_workers = 8
    pin_memory = True
    prefetch_factor = 4
    persistent_workers = True

    train_loader = DataLoader(
        PointCloudDataset(input_training, output_training),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    test_loader = DataLoader(
        PointCloudDataset(input_test, output_test),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    return train_loader, test_loader

class PointNet(nn.Module):
    """
    PointNet model for point cloud regression tasks.
    """
    def __init__(self, scaling, input_numbers, point_numbers, targets_numbers):
        super(PointNet, self).__init__()
        self.scaling = scaling
        self.point_numbers = point_numbers
        self.targets_numbers = targets_numbers
        
        # Shared MLP layers
        self.conv1 = nn.Conv1d(input_numbers, int(64 * scaling), 1)
        self.bn1 = nn.BatchNorm1d(int(64 * scaling))
        self.conv2 = nn.Conv1d(int(64 * scaling), int(64 * scaling), 1)
        self.bn2 = nn.BatchNorm1d(int(64 * scaling))
        self.activation = nn.ReLU()
        
        self.conv3 = nn.Conv1d(int(64 * scaling), int(64 * scaling), 1)
        self.bn3 = nn.BatchNorm1d(int(64 * scaling))
        self.conv4 = nn.Conv1d(int(64 * scaling), int(128 * scaling), 1)
        self.bn4 = nn.BatchNorm1d(int(128 * scaling))
        self.conv5 = nn.Conv1d(int(128 * scaling), int(1024 * scaling), 1)
        self.bn5 = nn.BatchNorm1d(int(1024 * scaling))
        
        self.maxpool = nn.MaxPool1d(self.point_numbers)
        
        # Layers after combining local and global features
        self.conv6 = nn.Conv1d(int(64 * scaling) + int(1024 * scaling), int(512 * scaling), 1)
        self.bn6 = nn.BatchNorm1d(int(512 * scaling))
        self.conv7 = nn.Conv1d(int(512 * scaling), int(256 * scaling), 1)
        self.bn7 = nn.BatchNorm1d(int(256 * scaling))
        self.conv8 = nn.Conv1d(int(256 * scaling), int(128 * scaling), 1)
        self.bn8 = nn.BatchNorm1d(int(128 * scaling))
        
        self.conv9 = nn.Conv1d(int(128 * scaling), int(128 * scaling), 1)
        self.bn9 = nn.BatchNorm1d(int(128 * scaling))
        self.conv10 = nn.Conv1d(int(128 * scaling), self.targets_numbers, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the PointNet model.
        """
        batch_size = x.size(0)
        x = x.transpose(2, 1)
        
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        
        local_feature = x.clone()
        
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.activation(self.bn5(self.conv5(x)))
        
        global_feature = self.maxpool(x).view(batch_size, -1)
        global_feature = global_feature.unsqueeze(2).repeat(1, 1, self.point_numbers)
        
        x = torch.cat([local_feature, global_feature], dim=1)
        
        x = self.activation(self.bn6(self.conv6(x)))
        x = self.activation(self.bn7(self.conv7(x)))
        x = self.activation(self.bn8(self.conv8(x)))
        
        x = self.activation(self.bn9(self.conv9(x)))
        x = self.sigmoid(self.conv10(x))
        x = x.transpose(2, 1)
        
        return x

def define_model(args, N_inputs, device):
    """
    Initialize and return a PointNet model.
    """
    model = PointNet(
        scaling=args.scaling,
        input_numbers=N_inputs,
        point_numbers=args.N_pt,
        targets_numbers=4,
    ).to(device)
    return model

def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_mae(true_vals, pred_vals):
    """
    Calculate Mean Absolute Error (MAE) for each component.
    """
    return np.mean(np.abs(pred_vals - true_vals), axis=(0, 1))

def calculate_rmse(true_vals, pred_vals):
    """
    Calculate Root Mean Square Error (RMSE) for each component.
    """
    return np.sqrt(np.mean((pred_vals - true_vals) ** 2, axis=(0, 1)))

def calculate_r2(true_vals, pred_vals):
    """
    Calculate R² score for each component.
    """
    true_flat = true_vals.reshape(-1, true_vals.shape[-1])
    pred_flat = pred_vals.reshape(-1, pred_vals.shape[-1])
    return r2_score(true_flat, pred_flat, multioutput='raw_values')

def train_model(model, train_loader, test_loader, args, output_scalers, experiment_dir, device):
    """
    Train the PointNet model and save intermediate checkpoints and metrics.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    train_losses, val_losses = [], []
    total_training_time = 0

    logging.info("Epoch    Train Loss Val Loss   MAE_ux MAE_uy MAE_uz MAE_vm R2_ux R2_uy R2_uz R2_vm Elapsed Time (s)")

    for epoch in range(args.N_iterations):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss_epoch = running_loss / len(train_loader)
        train_losses.append(train_loss_epoch)

        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        val_loss_epoch = val_loss / len(test_loader)
        val_losses.append(val_loss_epoch)

        elapsed_time = time.time() - epoch_start_time
        total_training_time += elapsed_time

        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        outputs_original = output_scalers.inverse_transform(all_outputs.reshape(-1, all_outputs.shape[-1])).reshape(all_outputs.shape)
        targets_original = output_scalers.inverse_transform(all_targets.reshape(-1, all_targets.shape[-1])).reshape(all_targets.shape)

        mae = calculate_mae(targets_original, outputs_original)
        r2 = calculate_r2(targets_original, outputs_original)

        logging.info(f"[{epoch+1}/{args.N_iterations}] [{train_loss_epoch:.2e}] [{val_loss_epoch:.2e}] "
                     f"[{mae[0]:.2e}, {mae[1]:.2e}, {mae[2]:.2e}, {mae[3]:.2e}, "
                     f"{r2[0]:.2e}, {r2[1]:.2e}, {r2[2]:.2e}, {r2[3]:.2e}] {total_training_time:.2f}s")

        scheduler.step()

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f'{experiment_dir}/model_checkpoint_epoch_{epoch+1}.pth')

    torch.save(model.state_dict(), f'{experiment_dir}/model_final.pth')

    total_params = count_parameters(model)
    with open(f'{experiment_dir}/total_params.txt', 'w') as f:
        f.write(f"{total_params}")

    with open(os.path.join(experiment_dir, 'training_time.txt'), 'w') as f:
        f.write(f"{total_training_time:.2f}")

    np.save(f'{experiment_dir}/train_losses.npy', np.array(train_losses))
    np.save(f'{experiment_dir}/val_losses.npy', np.array(val_losses))

    return train_losses, val_losses

def plot_loss_curves(train_losses, val_losses, experiment_dir):
    """
    Plot and save training and validation loss curves.
    """
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss over Epochs')
    plt.savefig(f'{experiment_dir}/loss_curve.jpg', dpi=300)
    plt.close()

def evaluate_model(model, test_loader, output_scalers, test_case_file, args, experiment_dir, device):
    """
    Evaluate the trained model on the test set and save metrics and plots.
    """
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    outputs_original = output_scalers.inverse_transform(all_outputs.reshape(-1, all_outputs.shape[-1])).reshape(all_outputs.shape)
    targets_original = output_scalers.inverse_transform(all_targets.reshape(-1, all_targets.shape[-1])).reshape(all_targets.shape)

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

        true_vals_dir = targets_original[indices, :, :]
        pred_vals_dir = outputs_original[indices, :, :]

        for i, comp in enumerate(components):
            true_comp = true_vals_dir[:, :, i].flatten()
            pred_comp = pred_vals_dir[:, :, i].flatten()

            mae = np.mean(np.abs(pred_comp - true_comp))
            rmse = np.sqrt(np.mean((pred_comp - true_comp) ** 2))
            r2 = r2_score(true_comp, pred_comp)

            with open(os.path.join(subset_metrics_all_dir, f'{direction}_{comp}_MAE.txt'), 'w') as f_mae:
                f_mae.write(f"{mae}")
            with open(os.path.join(subset_metrics_all_dir, f'{direction}_{comp}_RMSE.txt'), 'w') as f_rmse:
                f_rmse.write(f"{rmse}")
            with open(os.path.join(subset_metrics_all_dir, f'{direction}_{comp}_R2.txt'), 'w') as f_r2:
                f_r2.write(f"{r2:.3f}")

            logging.info(f"Subset Metrics for {direction} {comp}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
            plot_r2_scatter(true_comp, pred_comp, comp, direction, subset_plots_dir)

def plot_r2_scatter(true_vals, pred_vals, comp, direction, experiment_dir):
    """
    Plot a scatter plot to visualize the R² score for a specific component and direction.
    """
    r2 = r2_score(true_vals, pred_vals)
    plt.figure(figsize=(6,6))
    plt.scatter(true_vals, pred_vals, alpha=0.3, s=10, label='Data Points')
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit')
    plt.xlabel(f'True {comp}')
    plt.ylabel(f'Predicted {comp}')
    plt.title(f'{direction.capitalize()} Direction: {comp} Prediction vs True\nR² = {r2:.3f}')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, f'{direction}_scatter_{comp}.jpg'), dpi=300)
    plt.close()

def main():
    """
    Main entry point for training and evaluating the PointNet model.
    """
    args = parse_arguments()
    device = get_device(args.gpu)
    name = f'PointNet_RUN{args.RUN}_D{args.N_samples}_N{args.N_pt}_input_{args.input_components}_output_{args.output_components}_split_{args.split_method}_scaling_{args.scaling}'
    experiment_dir = setup_logging(args, name)
    log_parameters(args, device, name)

    input_training, output_training, input_test, output_test, output_scalers, train_case_file, test_case_file = load_and_preprocess_data(args, experiment_dir)
    train_loader, test_loader = create_dataloaders(input_training, output_training, input_test, output_test, args.batch_size)

    N_inputs = input_training.shape[-1]
    model = define_model(args, N_inputs, device)

    logging.info(f'\nModel Structure:\n{model}')
    total_params = count_parameters(model)
    logging.info(f'Total number of trainable parameters: {total_params}\n')

    train_losses, val_losses = train_model(model, train_loader, test_loader, args, output_scalers, experiment_dir, device)

    plot_loss_curves(train_losses, val_losses, experiment_dir)
    
    model_final_path = os.path.join(experiment_dir, 'model_final.pth')
    model.load_state_dict(torch.load(model_final_path, map_location=device))
    model.to(device)

    input_training, output_training, input_test, output_test, output_scalers, train_case_file, test_case_file = load_and_preprocess_data(args, experiment_dir)
    _, test_loader = create_dataloaders(input_training, output_training, input_test, output_test, args.batch_size)

    evaluate_model(model, test_loader, output_scalers, test_case_file, args, experiment_dir, device)

if __name__ == "__main__":
    main()
