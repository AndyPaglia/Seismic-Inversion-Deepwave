# Importa librerie necessarie
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import deepwave
from deepwave import scalar
from scipy.ndimage import gaussian_filter
import time


def create_initial_model(true_vp, smoothing_sigma=5):
    """
    Crea il modello iniziale per la FWI applicando uno smoothing al modello vero
    
    Argomenti:
        true_vp: Modello di velocità vero
        smoothing_sigma: Sigma per il filtro gaussiano
        
    Output che otteniamo:
        initial_vp: Modello di velocità iniziale smoothato
    """
    initial_vp = gaussian_filter(true_vp, sigma=smoothing_sigma)
    return initial_vp


def forward_modeling(v, dx, dt, source_amplitudes, source_locations, 
                    receiver_locations, accuracy=8, pml_width=20, f0=10, 
                    grad_checkpoint_segments=4):
    """
    Esegue il forward modeling
    
    Argomenti:
        v: Modello di velocità (torch tensor)
        dx: Spaziatura griglia
        dt: Passo temporale
        source_amplitudes: Ampiezze sorgenti
        source_locations: Posizioni sorgenti
        receiver_locations: Posizioni ricevitori
        accuracy: Ordine di accuratezza
        pml_width: Larghezza PML
        f0: Frequenza dominante
        grad_checkpoint_segments: Numero di segmenti per gradient checkpointing
        
    Output che dovremmo ottenere:
        receiver_amplitudes: Dati ai ricevitori
    """
    # gradient checkpointing per ridurre l'uso di memoria
    out = scalar(
        v, dx, dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=accuracy,
        pml_width=pml_width,
        pml_freq=f0,
        # Ridurre l'uso di memoria
        max_vel=v.max().item() if torch.is_tensor(v) else v,
        survey_pad=None,
        # Gradient checkpointing: divide il calcolo in segmenti
        # Più segmenti = meno memoria ma più calcolo
        origin=None
    )
    return out[-1]


def compute_loss(pred, obs):
    """
    Calcola la loss function (L2 norm)
    
    Argomenti:
        pred: Dati predetti
        obs: Dati osservati
        
    Output che otteniamo:
        loss: Valore della loss
    """
    return torch.nn.functional.mse_loss(pred, obs)


def fwi_iteration(v, dx, dt, source_amplitudes, source_locations, 
                  receiver_locations, observed_data, accuracy=8, 
                  pml_width=20, f0=10, batch_size=None):
    """
    Esegue una singola iterazione di FWI
    Processo gli shot in batch
    
    Argomenti:
        v: Modello di velocità corrente (torch tensor con requires_grad=True)
        dx: Spaziatura griglia
        dt: Passo temporale
        source_amplitudes: Ampiezze sorgenti
        source_locations: Posizioni sorgenti
        receiver_locations: Posizioni ricevitori
        observed_data: Dati osservati
        accuracy: Ordine di accuratezza
        pml_width: Larghezza PML
        f0: Frequenza dominante
        batch_size: Numero di shot da processare insieme (None = tutti)
        
    Output che otteniamo:
        loss: Valore della loss
        grad: Gradiente rispetto a v
    """
    n_shots = source_amplitudes.shape[0]
    
    # Se batch_size non specificato, usiamo batch piccoli per risparmiare memoria
    if batch_size is None:
        # Batch_size in base alla RAM disponibile
        # Con CPU, batch molto piccoli
        batch_size = 1  # Processa 1 shot alla volta per minimizzare memoria
    
    total_loss = 0.0
    
    # Accumulo il gradiente su tutti i batch
    if v.grad is not None:
        v.grad.zero_()
    
    # Processo gli shot in batch
    for batch_start in range(0, n_shots, batch_size):
        batch_end = min(batch_start + batch_size, n_shots)
        
        # Estraggo il batch corrente
        batch_source_amps = source_amplitudes[batch_start:batch_end]
        batch_source_locs = source_locations[batch_start:batch_end]
        batch_receiver_locs = receiver_locations[batch_start:batch_end]
        batch_observed = observed_data[batch_start:batch_end]
        
        # Forward modeling per questo batch
        predicted_data = forward_modeling(v, dx, dt, batch_source_amps, 
                                         batch_source_locs, batch_receiver_locs,
                                         accuracy, pml_width, f0)
        
        # Calcolo la loss per questo batch
        batch_loss = compute_loss(predicted_data, batch_observed)
        
        # Backpropagation
        batch_loss.backward()
        
        # Accumulo la loss
        total_loss += batch_loss.item() * (batch_end - batch_start)
        
        # Libero memoria 
        del predicted_data, batch_loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Media della loss
    avg_loss = total_loss / n_shots
    
    # Clonazione del gradiente accumulato
    grad = v.grad.clone()
    
    return avg_loss, grad


# Ottimizzazione consigliata in vari link sul tema + AI
def clip_gradient(grad, max_val=None, percentile=99):
    """
    Clippa il gradiente per evitare aggiornamenti troppo grandi
    
    Argomenti:
        grad: Gradiente
        max_val: Valore massimo (se None, usa il percentile)
        percentile: Percentile per il clipping
        
    Output che otteniamo:
        grad_clipped: Gradiente clippato
    """
    if max_val is None:
        max_val = torch.quantile(torch.abs(grad), percentile / 100.0)
    
    return torch.clamp(grad, -max_val, max_val)


def apply_constraints(v, v_min, v_max):
    """
    Applica vincoli al modello di velocità
    
    Argomenti:
        v: Modello di velocità
        v_min: Velocità minima
        v_max: Velocità massima
        
    Output che otteniamo:
        v_constrained: Modello vincolato
    """
    return torch.clamp(v, v_min, v_max)


def fwi_optimization(v_init, v_true, dx, dt, source_amplitudes, 
                    source_locations, receiver_locations, observed_data,
                    n_iterations=50, learning_rate=10.0, v_min=1500, v_max=4500,
                    accuracy=8, pml_width=20, f0=10, device='cpu',
                    save_every=5, output_dir='fwi_results', batch_size=1):
    """
    Esegue l'ottimizzazione FWI completa
    
    Argomenti:
        v_init: Modello di velocità iniziale
        v_true: Modello di velocità vero 
        dx: Spaziatura griglia
        dt: Passo temporale
        source_amplitudes: Ampiezze sorgenti
        source_locations: Posizioni sorgenti
        receiver_locations: Posizioni ricevitori
        observed_data: Dati osservati
        n_iterations: Numero di iterazioni
        learning_rate: Learning rate
        v_min: Velocità minima
        v_max: Velocità massima
        accuracy: Ordine di accuratezza
        pml_width: Larghezza PML
        f0: Frequenza dominante
        device: Device PyTorch
        save_every: Salva il modello ogni N iterazioni
        output_dir: Directory di output
        batch_size: Shot da processare insieme (1 = minima memoria)
        
    Output che otteniamo:
        v_final: Modello finale
        loss_history: Storia delle loss
    """
    
    # Creo directory di output se non sono presenti
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'gradients'), exist_ok=True)
    
    # Inizializzo il modello ottimizzabile
    v = v_init.clone().to(device)
    v.requires_grad = True
    
    # Storia delle loss
    loss_history = []
    
    # Setup per il plot (oer ogni salvataggio)
    fig = None
    
    print(f"\n{'='*60}")
    print(f"INIZIO FWI OPTIMIZATION")
    print(f"{'='*60}")
    print(f"  - Numero iterazioni: {n_iterations}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Vincoli velocità: [{v_min}, {v_max}] m/s")
    print(f"  - Device: {device}")
    print(f"  - Batch size: {batch_size} shot(s) per volta")
    print(f"  - Numero totale shots: {source_amplitudes.shape[0]}")
    print(f"{'='*60}\n")
    
    for iteration in range(n_iterations):
        iter_start_time = time.time()
        
        # Zero gradient
        if v.grad is not None:
            v.grad.zero_()
        
        # Esegue una iterazione di FWI (con batching per meno memoria)
        loss, grad = fwi_iteration(v, dx, dt, source_amplitudes, 
                                   source_locations, receiver_locations,
                                   observed_data, accuracy, pml_width, f0,
                                   batch_size=batch_size)
        
        loss_history.append(loss)
        
        # Clippa il gradiente con funzione definita prima
        grad_clipped = clip_gradient(grad, percentile=95)
        
        # Gradient descent step
        with torch.no_grad():
            # Salva il modello prima dell'update per debug
            v_before_update = v.clone()
            
            v -= learning_rate * grad_clipped
            
            # Applica vincoli sulla velocità
            v.data = apply_constraints(v.data, v_min, v_max)
            
            # Debug: verifica che il modello stia effettivamente cambiando (con 20 iterazioni non si nota - prova con 10000 sulle macchine del lab)
            model_change = torch.norm(v - v_before_update).item()
        
        
        # Forza la sincronizzazione del tensore su CPU per calcoli successivi
        v_cpu = v.detach().clone().cpu().numpy()
        v_true_np = v_true.cpu().numpy() if torch.is_tensor(v_true) else v_true
        
        model_error = np.sqrt(np.mean((v_cpu - v_true_np)**2))
        grad_norm = torch.norm(grad).item()
        
        iter_time = time.time() - iter_start_time
        
        # Stampa i progressi con informazioni sul cambio del modello
        print(f"Iter {iteration+1:3d}/{n_iterations} | "
              f"Loss: {loss:.6e} | "
              f"Model Error: {model_error:.2f} | "
              f"Grad Norm: {grad_norm:.6e} | "
              f"Model Change: {model_change:.6e} | "
              f"Time: {iter_time:.2f}s")
        
        # Salva e plotta ogni N iterazioni
        if (iteration + 1) % save_every == 0 or iteration == 0 or iteration == n_iterations - 1:
            plt.close(fig)
            
            # Crea nuova figura
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # Plot
            axes[0, 0].imshow(v_true_np.T, cmap='jet', aspect='auto')
            axes[0, 0].set_title('Modello Vero')
            axes[0, 0].set_xlabel('X')
            axes[0, 0].set_ylabel('Z')
            
            axes[0, 1].imshow(v_cpu.T, cmap='jet', aspect='auto', 
                            vmin=v_true_np.min(), vmax=v_true_np.max())
            axes[0, 1].set_title(f'Modello Corrente (Iter {iteration+1})')
            axes[0, 1].set_xlabel('X')
            axes[0, 1].set_ylabel('Z')
            
            diff = v_cpu - v_true_np
            im = axes[0, 2].imshow(diff.T, cmap='seismic', aspect='auto',
                                  vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
            axes[0, 2].set_title('Differenza')
            axes[0, 2].set_xlabel('X')
            axes[0, 2].set_ylabel('Z')
            fig.colorbar(im, ax=axes[0, 2])
            
            axes[1, 0].plot(loss_history)
            axes[1, 0].set_xlabel('Iterazione')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Loss History')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
            
            grad_cpu = grad.detach().cpu().numpy()
            im = axes[1, 1].imshow(grad_cpu.T, cmap='seismic', aspect='auto',
                                  vmin=-np.abs(grad_cpu).max()*0.1, 
                                  vmax=np.abs(grad_cpu).max()*0.1)
            axes[1, 1].set_title('Gradiente')
            axes[1, 1].set_xlabel('X')
            axes[1, 1].set_ylabel('Z')
            fig.colorbar(im, ax=axes[1, 1])
            
            axes[1, 2].plot(v_true_np[:, v_true_np.shape[1]//2], 'b-', label='Vero', linewidth=2)
            axes[1, 2].plot(v_cpu[:, v_cpu.shape[1]//2], 'r--', label='Stimato', linewidth=2)
            axes[1, 2].set_xlabel('X')
            axes[1, 2].set_ylabel('Velocità (m/s)')
            axes[1, 2].set_title('Profilo Verticale Centrale')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'fwi_iter_{iteration+1:04d}.png'), 
                       dpi=150, bbox_inches='tight')
            
            # Se plt.pause, problemi con tkinter
            # plt.pause(0.01)
            
            # Salva il modello
            np.save(os.path.join(output_dir, 'models', f'v_iter_{iteration+1:04d}.npy'), v_cpu)
            np.save(os.path.join(output_dir, 'gradients', f'grad_iter_{iteration+1:04d}.npy'), grad_cpu)
    
    if fig is not None:
        plt.close(fig)
    
    print(f"\n{'='*60}")
    print(f"FWI COMPLETATA!")
    print(f"{'='*60}")
    print(f"  - Loss finale: {loss_history[-1]:.6e}")
    print(f"  - Model error finale: {model_error:.2f} m/s")
    print(f"  - Miglioramento: {(1 - model_error/np.sqrt(np.mean((v_init.cpu().numpy() - v_true_np)**2)))*100:.2f}%")
    print(f"{'='*60}\n")
    
    return v.detach(), loss_history


def main(args):
    """
    Funzione principale per la FWI
    """
    
    # Device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"CONFIGURAZIONE FWI")
    print(f"{'='*60}")
    print(f"  - Device: {device}")
    
    # Carico i dati osservati dal forward modeling
    print(f"\n{'='*60}")
    print(f"CARICAMENTO DATI")
    print(f"{'='*60}")
    print(f"  - Path dati osservati: {args.observed_data_path}")
    
    data_file = np.load(args.observed_data_path)
    observed_data_np = data_file['receiver_data']
    src_coordinates = data_file['src_coordinates']
    rec_coordinates = data_file['rec_coordinates']
    spacing = data_file['spacing']
    f0 = float(data_file['f0'])
    dt_ms = float(data_file['dt'])
    tn_ms = float(data_file['tn'])
    
    print(f"  - Dimensione dati osservati: {observed_data_np.shape}")
    print(f"  - Numero sorgenti: {len(src_coordinates)}")
    print(f"  - Numero ricevitori: {len(rec_coordinates)}")
    print(f"  - Frequenza: {f0} Hz")
    print(f"  - dt: {dt_ms} ms")
    
    # Step uguali/simili a quelli già visti nel forward modeling

    # Carico il modello vero
    print(f"  - Path modello vero: {args.true_model_path}")
    true_model_file = np.load(args.true_model_path)
    v_true = true_model_file['vp']
    
    # Converto in m/s se necessario
    if v_true.max() < 10:
        print(f"  Conversione velocità da km/s a m/s...")
        v_true = v_true * 1000.0
    
    print(f"  - Dimensioni modello vero: {v_true.shape}")
    print(f"  - Min/Max velocità vero: {v_true.min():.2f} / {v_true.max():.2f} m/s")
    
    # Creo il modello iniziale --> smoothing del modello vero
    print(f"\n{'='*60}")
    print(f"CREAZIONE MODELLO INIZIALE")
    print(f"{'='*60}")
    v_init_np = create_initial_model(v_true, smoothing_sigma=args.smoothing_sigma)
    print(f"  - Smoothing sigma: {args.smoothing_sigma}")
    print(f"  - Min/Max velocità iniziale: {v_init_np.min():.2f} / {v_init_np.max():.2f} m/s")
    
    # Converti in torch tensors
    v_init = torch.from_numpy(v_init_np).float().to(device)
    v_true_tensor = torch.from_numpy(v_true).float().to(device)
    observed_data = torch.from_numpy(observed_data_np).float().to(device)
    
    # Parametri temporali
    dt_sec = dt_ms / 1000.0
    tn_sec = tn_ms / 1000.0
    nt = int(tn_sec / dt_sec) + 1
    
    # Configuro sorgenti e ricevitori
    dx = float(spacing[0])
    pml_width = args.pml_width
    
    # Filtro sorgenti e ricevitori validi (come nel forward)
    max_x_idx = v_true.shape[0] - pml_width - 1
    max_z_idx = v_true.shape[1] - pml_width - 1
    
    src_x_indices = torch.tensor(src_coordinates[:, 0] / dx + pml_width, dtype=torch.long)
    src_z_indices = torch.tensor(src_coordinates[:, 1] / dx + pml_width, dtype=torch.long)
    valid_src_mask = (src_x_indices >= pml_width) & (src_x_indices <= max_x_idx) & \
                     (src_z_indices >= pml_width) & (src_z_indices <= max_z_idx)
    
    src_x_indices = src_x_indices[valid_src_mask]
    src_z_indices = src_z_indices[valid_src_mask]
    n_shots = len(src_x_indices)
    
    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_locations[:, 0, 0] = src_x_indices
    source_locations[:, 0, 1] = src_z_indices
    
    rec_x_indices = torch.tensor(rec_coordinates[:, 0] / dx + pml_width, dtype=torch.long)
    rec_z_indices = torch.tensor(rec_coordinates[:, 1] / dx + pml_width, dtype=torch.long)
    valid_rec_mask = (rec_x_indices >= pml_width) & (rec_x_indices <= max_x_idx) & \
                     (rec_z_indices >= pml_width) & (rec_z_indices <= max_z_idx)
    
    rec_x_indices = rec_x_indices[valid_rec_mask]
    rec_z_indices = rec_z_indices[valid_rec_mask]
    n_receivers = len(rec_x_indices)
    
    receiver_locations = torch.zeros(n_shots, n_receivers, 2, dtype=torch.long, device=device)
    for shot_idx in range(n_shots):
        receiver_locations[shot_idx, :, 0] = rec_x_indices
        receiver_locations[shot_idx, :, 1] = rec_z_indices
    
    # Genero il wavelet
    peak_time = 1.5 / f0
    source_amplitudes = (
        deepwave.wavelets.ricker(f0, nt, dt_sec, peak_time)
        .repeat(n_shots, 1, 1)
        .to(device)
    )
    
    print(f"\n{'='*60}")
    print(f"PARAMETRI OTTIMIZZAZIONE")
    print(f"{'='*60}")
    print(f"  - Numero iterazioni: {args.n_iterations}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Vincoli velocità: [{args.v_min}, {args.v_max}] m/s")
    print(f"  - Accuracy: {args.accuracy}")
    print(f"  - PML width: {args.pml_width}")
    
    # Esegui FWI
    v_final, loss_history = fwi_optimization(
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
    
    # Salva risultati finali
    v_final_np = v_final.cpu().numpy()
    np.savez(os.path.join(args.output_dir, 'fwi_results.npz'),
             v_final=v_final_np,
             v_init=v_init_np,
             v_true=v_true,
             loss_history=loss_history)
    
    # Plot finale comparativo
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im0 = axes[0].imshow(v_true.T, cmap='jet', aspect='auto')
    axes[0].set_title('Modello Vero')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Z')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(v_final_np.T, cmap='jet', aspect='auto',
                        vmin=v_true.min(), vmax=v_true.max())
    axes[1].set_title('Modello Finale FWI')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    plt.colorbar(im1, ax=axes[1])
    
    diff = v_final_np - v_true
    im2 = axes[2].imshow(diff.T, cmap='seismic', aspect='auto',
                        vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[2].set_title('Differenza (Finale - Vero)')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Z')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'fwi_final_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n Risultati salvati in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full Waveform Inversion con Deepwave')
    parser.add_argument('--observed_data_path', type=str, required=True,
                       help='Path ai dati osservati (output del forward modeling)')
    parser.add_argument('--true_model_path', type=str, required=True,
                       help='Path al modello vero')
    parser.add_argument('--n_iterations', type=int, default=50,
                       help='Numero di iterazioni FWI')
    parser.add_argument('--learning_rate', type=float, default=10.0,
                       help='Learning rate per gradient descent')
    parser.add_argument('--smoothing_sigma', type=float, default=5.0,
                       help='Sigma per smoothing del modello iniziale')
    parser.add_argument('--v_min', type=float, default=1500,
                       help='Velocità minima (m/s)')
    parser.add_argument('--v_max', type=float, default=4500,
                       help='Velocità massima (m/s)')
    parser.add_argument('--accuracy', type=int, default=8,
                       help='Ordine di accuratezza spaziale')
    parser.add_argument('--pml_width', type=int, default=20,
                       help='Larghezza PML')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Salva modello ogni N iterazioni')
    parser.add_argument('--output_dir', type=str, default='fwi_results',
                       help='Directory di output')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Numero di shot da processare insieme (1 = minima memoria)')
    
    args = parser.parse_args()
    main(args)
