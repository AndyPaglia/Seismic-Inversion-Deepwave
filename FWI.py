import argparse
import os
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np
import torch
import deepwave
from deepwave import scalar
from scipy.ndimage import gaussian_filter
import time
import warnings


def create_initial_model(true_vp, smoothing_sigma=5):
    """
    Crea il modello iniziale smoothato
    """
    initial_vp = gaussian_filter(true_vp, sigma=smoothing_sigma)
    return initial_vp


def fwi_step(v, dx, dt, source_amplitudes, source_locations, 
             receiver_locations, observed_data, accuracy=8, 
             pml_width=20, f0=10, batch_size=5):
    """
    Singolo step di FWI con mini-batching per gestire memoria GPU
    
    v deve avere requires_grad=True
    """
    n_shots = source_amplitudes.shape[0]
    total_loss = 0.0
    
    # Processa shots in mini-batch
    for batch_start in range(0, n_shots, batch_size):
        batch_end = min(batch_start + batch_size, n_shots)
        
        # Estrai batch
        batch_src_amp = source_amplitudes[batch_start:batch_end]
        batch_src_loc = source_locations[batch_start:batch_end]
        batch_rec_loc = receiver_locations[batch_start:batch_end]
        batch_obs = observed_data[batch_start:batch_end]
        
        # Forward modeling
        predicted_data = scalar(
            v, dx, dt,
            source_amplitudes=batch_src_amp,
            source_locations=batch_src_loc,
            receiver_locations=batch_rec_loc,
            accuracy=accuracy,
            pml_width=pml_width,
            pml_freq=f0
        )[-1]
        
        # Loss per questo batch
        batch_loss = torch.nn.functional.mse_loss(predicted_data, batch_obs)
        
        # Backward (accumula gradiente)
        batch_loss.backward()
        
        # Accumula loss
        total_loss += batch_loss.item() * (batch_end - batch_start)
        
        # Libera memoria
        del predicted_data, batch_loss
    
    # Media della loss
    avg_loss = total_loss / n_shots
    
    return avg_loss


def fwi_optimization(v_init, v_true, dx, dt, source_amplitudes, 
                    source_locations, receiver_locations, observed_data,
                    n_iterations=50, learning_rate=100.0, 
                    v_min=1500, v_max=4500,
                    accuracy=8, pml_width=20, f0=10, 
                    device='cuda', save_every=10, output_dir='fwi_results',
                    batch_size=5):
    """
    FWI optimization loop completo con mini-batching
    """
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    
    # Il modello deve essere una variabile che ottimizziamo
    # Usa torch.nn.Parameter per assicurarsi che sia ottimizzabile
    v = torch.nn.Parameter(v_init.clone().to(device))
    
    # Optimizer PyTorch (molto più robusto del gradient descent manuale)
    optimizer = torch.optim.Adam([v], lr=learning_rate)
    
    # Converti v_true in numpy per confronti
    v_true_np = v_true.cpu().numpy() if torch.is_tensor(v_true) else v_true
    v_init_np = v_init.cpu().numpy() if torch.is_tensor(v_init) else v_init
    
    # Calcola errore iniziale
    initial_error = np.sqrt(np.mean((v_init_np - v_true_np)**2))
    
    loss_history = []
    error_history = []
    
    print(f"\n{'='*70}")
    print(f"FWI OPTIMIZATION START")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model shape: {v.shape}")
    print(f"Shots: {source_amplitudes.shape[0]}")
    print(f"Receivers per shot: {receiver_locations.shape[1]}")
    print(f"Batch size: {batch_size} shots per mini-batch")
    print(f"Iterations: {n_iterations}")
    print(f"Learning rate: {learning_rate}")
    print(f"Optimizer: Adam")
    print(f"Initial model error: {initial_error:.2f} m/s")
    print(f"{'='*70}\n")
    
    for iteration in range(n_iterations):
        iter_start = time.time()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward + loss + backward (con batching)
        loss = fwi_step(
            v, dx, dt, 
            source_amplitudes, source_locations, receiver_locations,
            observed_data, accuracy, pml_width, f0, batch_size
        )
        
        # Salva il gradiente PRIMA dell'update
        grad_norm = v.grad.norm().item()
        
        # Optimizer step (aggiorna v in-place)
        optimizer.step()
        
        # Applica constraints DOPO l'update
        with torch.no_grad():
            v.data.clamp_(v_min, v_max)
        
        # Calcola metriche
        with torch.no_grad():
            v_current = v.detach().cpu().numpy()
            current_error = np.sqrt(np.mean((v_current - v_true_np)**2))
            improvement = ((initial_error - current_error) / initial_error) * 100
        
        loss_history.append(loss)
        error_history.append(current_error)
        
        iter_time = time.time() - iter_start
        
        # Print progress
        print(f"[{iteration+1:4d}/{n_iterations}] "
              f"Loss: {loss:.6e} | "
              f"Error: {current_error:7.2f} m/s | "
              f"Improv: {improvement:+6.2f}% | "
              f"GradNorm: {grad_norm:.4e} | "
              f"Time: {iter_time:6.2f}s")
        
        # Save and plot
        if (iteration + 1) % save_every == 0 or iteration == 0 or iteration == n_iterations - 1:
            save_iteration_results(
                v_current, v_true_np, v_init_np,
                loss_history, error_history,
                iteration + 1, output_dir
            )
    
    print(f"\n{'='*70}")
    print(f"FWI COMPLETED")
    print(f"{'='*70}")
    print(f"Final loss: {loss_history[-1]:.6e}")
    print(f"Final error: {error_history[-1]:.2f} m/s")
    print(f"Total improvement: {improvement:.2f}%")
    print(f"Error reduction: {initial_error - error_history[-1]:.2f} m/s")
    print(f"{'='*70}\n")
    
    return v.detach(), loss_history, error_history


def save_iteration_results(v_current, v_true, v_init, 
                          loss_history, error_history,
                          iteration, output_dir):
    """
    Salva modello e crea plot
    """
    # Salva modello
    np.save(os.path.join(output_dir, 'models', f'v_iter_{iteration:05d}.npy'), 
            v_current)
    
    # Crea figura
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    vmin, vmax = v_true.min(), v_true.max()
    
    # Modello vero
    im0 = axes[0, 0].imshow(v_true.T, cmap='jet', aspect='auto', 
                            vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('True Model', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Z')
    plt.colorbar(im0, ax=axes[0, 0], label='Velocity (m/s)')
    
    # Modello corrente
    im1 = axes[0, 1].imshow(v_current.T, cmap='jet', aspect='auto',
                            vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Current Model (Iter {iteration})', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    plt.colorbar(im1, ax=axes[0, 1], label='Velocity (m/s)')
    
    # Differenza
    diff = v_current - v_true
    diff_max = np.abs(diff).max()
    im2 = axes[0, 2].imshow(diff.T, cmap='seismic', aspect='auto',
                           vmin=-diff_max, vmax=diff_max)
    axes[0, 2].set_title('Difference (Current - True)', 
                        fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Z')
    plt.colorbar(im2, ax=axes[0, 2], label='Velocity Error (m/s)')
    
    # Loss history
    axes[1, 0].plot(loss_history, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Loss History', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error history
    axes[1, 1].plot(error_history, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('RMSE (m/s)', fontsize=12)
    axes[1, 1].set_title('Model Error History', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Profili verticali
    mid_x = v_true.shape[0] // 2
    axes[1, 2].plot(v_true[mid_x, :], 'b-', label='True', linewidth=2)
    axes[1, 2].plot(v_init[mid_x, :], 'g--', label='Initial', linewidth=2, alpha=0.7)
    axes[1, 2].plot(v_current[mid_x, :], 'r-', label='Current', linewidth=2)
    axes[1, 2].set_xlabel('Depth Index', fontsize=12)
    axes[1, 2].set_ylabel('Velocity (m/s)', fontsize=12)
    axes[1, 2].set_title('Vertical Profile (Center)', fontsize=14, fontweight='bold')
    axes[1, 2].legend(fontsize=11)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fwi_iter_{iteration:05d}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def main(args):
    """
    Main function
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"\n🎮 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load observed data
    print(f"\n Loading data from: {args.observed_data_path}")
    data = np.load(args.observed_data_path)
    observed_data_np = data['receiver_data']
    src_coords = data['src_coordinates']
    rec_coords = data['rec_coordinates']
    spacing = data['spacing']
    f0 = float(data['f0'])
    dt_ms = float(data['dt'])
    tn_ms = float(data['tn'])
    
    print(f"   Observed data shape: {observed_data_np.shape}")
    print(f"   Frequency: {f0} Hz")
    print(f"   dt: {dt_ms} ms")
    
    # Load true model
    print(f"\n Loading true model from: {args.true_model_path}")
    true_model = np.load(args.true_model_path)
    v_true = true_model['vp']
    
    # Convert to m/s if needed
    if v_true.max() < 10:
        print("  Converting velocities from km/s to m/s")
        v_true = v_true * 1000.0
    
    print(f"   Model shape: {v_true.shape}")
    print(f"   Velocity range: [{v_true.min():.0f}, {v_true.max():.0f}] m/s")
    
    # Create initial model
    print(f"\n Creating initial model (smoothing sigma={args.smoothing_sigma})")
    v_init_np = create_initial_model(v_true, args.smoothing_sigma)
    print(f"   Initial velocity range: [{v_init_np.min():.0f}, {v_init_np.max():.0f}] m/s")
    
    # Convert to torch
    v_init = torch.from_numpy(v_init_np).float()
    v_true_tensor = torch.from_numpy(v_true).float()
    observed_data = torch.from_numpy(observed_data_np).float().to(device)
    
    # Time parameters
    dt_sec = dt_ms / 1000.0
    tn_sec = tn_ms / 1000.0
    nt = int(tn_sec / dt_sec) + 1
    
    # Grid parameters
    dx = float(spacing[0])
    pml_width = args.pml_width
    
    # Setup sources and receivers
    max_x_idx = v_true.shape[0] - pml_width - 1
    max_z_idx = v_true.shape[1] - pml_width - 1
    
    # Filter valid sources
    src_x = torch.tensor(src_coords[:, 0] / dx + pml_width, dtype=torch.long)
    src_z = torch.tensor(src_coords[:, 1] / dx + pml_width, dtype=torch.long)
    valid_src = (src_x >= pml_width) & (src_x <= max_x_idx) & \
                (src_z >= pml_width) & (src_z <= max_z_idx)
    src_x = src_x[valid_src]
    src_z = src_z[valid_src]
    n_shots = len(src_x)
    
    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_locations[:, 0, 0] = src_x
    source_locations[:, 0, 1] = src_z
    
    # Filter valid receivers
    rec_x = torch.tensor(rec_coords[:, 0] / dx + pml_width, dtype=torch.long)
    rec_z = torch.tensor(rec_coords[:, 1] / dx + pml_width, dtype=torch.long)
    valid_rec = (rec_x >= pml_width) & (rec_x <= max_x_idx) & \
                (rec_z >= pml_width) & (rec_z <= max_z_idx)
    rec_x = rec_x[valid_rec]
    rec_z = rec_z[valid_rec]
    n_receivers = len(rec_x)
    
    receiver_locations = torch.zeros(n_shots, n_receivers, 2, dtype=torch.long, device=device)
    for i in range(n_shots):
        receiver_locations[i, :, 0] = rec_x
        receiver_locations[i, :, 1] = rec_z
    
    # Generate source wavelet
    peak_time = 1.5 / f0
    source_amplitudes = (
        deepwave.wavelets.ricker(f0, nt, dt_sec, peak_time)
        .repeat(n_shots, 1, 1)
        .to(device)
    )
    
    print(f"\n  FWI Configuration:")
    print(f"   Valid shots: {n_shots}")
    print(f"   Valid receivers: {n_receivers}")
    print(f"   Iterations: {args.n_iterations}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Optimizer: Adam")
    print(f"   Accuracy: {args.accuracy}")
    print(f"   PML width: {args.pml_width}")
    
    # Run FWI
    v_final, loss_history, error_history = fwi_optimization(
        v_init=v_init,
        v_true=v_true_tensor,
        dx=dx,
        dt=dt_sec,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        observed_data=observed_data,
        n_iterations=args.n_iterations,
        learning_rate=args.learning_rate,
        v_min=args.v_min,
        v_max=args.v_max,
        accuracy=args.accuracy,
        pml_width=args.pml_width,
        f0=f0,
        device=device,
        save_every=args.save_every,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    # Save final results
    v_final_np = v_final.cpu().numpy()
    np.savez(
        os.path.join(args.output_dir, 'fwi_final_results.npz'),
        v_final=v_final_np,
        v_init=v_init_np,
        v_true=v_true,
        loss_history=loss_history,
        error_history=error_history
    )
    
    print(f"\n Results saved to: {args.output_dir}")
    print(" Done!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FWI with Deepwave - GPU Optimized')
    
    # Data paths
    parser.add_argument('--observed_data_path', type=str, required=True,
                       help='Path to observed data (.npz)')
    parser.add_argument('--true_model_path', type=str, required=True,
                       help='Path to true velocity model (.npz)')
    
    # FWI parameters
    parser.add_argument('--n_iterations', type=int, default=100,
                       help='Number of FWI iterations')
    parser.add_argument('--learning_rate', type=float, default=1.0,
                       help='Learning rate (Adam optimizer) - start small!')
    parser.add_argument('--smoothing_sigma', type=float, default=5.0,
                       help='Smoothing sigma for initial model')
    
    # Model constraints
    parser.add_argument('--v_min', type=float, default=1500,
                       help='Minimum velocity (m/s)')
    parser.add_argument('--v_max', type=float, default=4500,
                       help='Maximum velocity (m/s)')
    
    # Numerical parameters
    parser.add_argument('--accuracy', type=int, default=8,
                       help='Spatial accuracy order (2, 4, 6, 8)')
    parser.add_argument('--pml_width', type=int, default=20,
                       help='PML width (cells)')
    
    # Output
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save model every N iterations')
    parser.add_argument('--output_dir', type=str, default='fwi_results',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Mini-batch size (shots per batch) for memory management')
    
    args = parser.parse_args()
    main(args)
