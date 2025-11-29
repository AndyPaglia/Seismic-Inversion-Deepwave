# Import delle librerie necessarie
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import deepwave
from deepwave import scalar
from tqdm import tqdm


def forward_modeling_deepwave(vp, spacing, src_coordinates, rec_coordinates, 
                              f0, tn, dt, device, accuracy=8, pml_width=20):
    """
    Forward modeling usando Deepwave
    
    Argomenti della funzione:
        vp: Modello di velocità
        spacing: Spaziatura della griglia [dx, dz] in metri
        src_coordinates: Coordinate delle sorgenti (n_src, 2) in metri
        rec_coordinates: Coordinate dei ricevitori (n_rec, 2) in metri
        f0: Frequenza dominante della sorgente in Hz
        tn: Tempo finale di registrazione in millisecondi
        dt: Passo temporale in millisecondi
        device: Device PyTorch 
        accuracy: Ordine di accuratezza delle differenze finite spaziali
        pml_width: Larghezza del PML in numero di celle
        
    In output otteniamo:
        receiver_amplitudes: Dati sintetici rilevati dai ricevitori
        source_amplitudes: Ampiezze delle sorgenti usate
    """
    
    # Converto il modello di velocità in tensore PyTorch (necessario per Deepwave)
    v = torch.from_numpy(vp).float().to(device)
    
    # Debug informazioni modello
    print(f"\n{'='*60}")
    print(f"DEBUG - INFORMAZIONI MODELLO")
    print(f"{'='*60}")
    print(f"  - Dimensione originale vp (numpy): {vp.shape}")
    print(f"  - Dimensione tensore v (torch): {v.shape}")
    print(f"  - Min velocità: {v.min():.2f} m/s")
    print(f"  - Max velocità: {v.max():.2f} m/s")
    print(f"  - Media velocità: {v.mean():.2f} m/s")
    print(f"  - Dtype: {v.dtype}")
    
    # Parametri temporali
    dt_sec = dt / 1000.0
    tn_sec = tn / 1000.0
    nt = int(tn_sec / dt_sec) + 1
    
    # Debug parametri temporali
    print(f"\n{'='*60}")
    print(f"DEBUG - PARAMETRI TEMPORALI")
    print(f"{'='*60}")
    print(f"  - dt (input): {dt} ms → {dt_sec} s")
    print(f"  - tn (input): {tn} ms → {tn_sec} s")
    print(f"  - Numero campioni temporali (nt): {nt}")
    print(f"  - Durata totale: {nt * dt_sec:.3f} s")
    
    # Numero di sorgenti e ricevitori
    n_shots = len(src_coordinates)
    n_receivers_per_shot = len(rec_coordinates)
    
    # Calcolo il tempo di picco per il wavelet di Ricker
    peak_time = 1.5 / f0
    
    # Spacing (Deepwave vuole dx scalare)
    dx = float(spacing[0]) # spaziatura orizzontale
    dz = float(spacing[1]) if len(spacing) > 1 else dx # spaziatura verticale
    
    # Debug parametri griglia
    print(f"\n{'='*60}")
    print(f"DEBUG - PARAMETRI GRIGLIA")
    print(f"{'='*60}")
    print(f"  - dx (spaziatura orizzontale): {dx} m")
    print(f"  - dz (spaziatura verticale): {dz} m")
    print(f"  - Celle griglia (nx x nz): {v.shape[0]} x {v.shape[1]}")
    print(f"  - Dimensione fisica X: {v.shape[0] * dx} m")
    print(f"  - Dimensione fisica Z: {v.shape[1] * dz} m")
    print(f"  - PML width: {pml_width} celle = {pml_width * dx} m")
    
    # Calcolo i limiti validi per le posizioni (escludendo il PML)
    max_x_idx = v.shape[0] - pml_width - 1
    max_z_idx = v.shape[1] - pml_width - 1
    
    # Converto coordinate in indici di griglia con offset PML (da sommare)
    src_x_indices = torch.tensor(src_coordinates[:, 0] / dx + pml_width, dtype=torch.long)
    src_z_indices = torch.tensor(src_coordinates[:, 1] / dz + pml_width, dtype=torch.long)
    
    # Debug informazioni sorgenti
    print(f"\n{'='*60}")
    print(f"DEBUG - SORGENTI (PRIMA DEL FILTRAGGIO)")
    print(f"{'='*60}")
    print(f"  - Numero totale sorgenti: {n_shots}")
    print(f"  - Coordinate X sorgenti (metri): [{src_coordinates[:, 0].min():.1f}, {src_coordinates[:, 0].max():.1f}]")
    print(f"  - Coordinate Z sorgenti (metri): {src_coordinates[0, 1]:.1f}")
    print(f"  - Indici X sorgenti (con PML): [{src_x_indices.min()}, {src_x_indices.max()}]")
    print(f"  - Indici Z sorgenti (con PML): {src_z_indices[0]}")
    
    # Filtro le sorgenti che sono fuori dalla geometria del modello considerato
    valid_src_mask = (src_x_indices >= pml_width) & (src_x_indices <= max_x_idx) & \
                     (src_z_indices >= pml_width) & (src_z_indices <= max_z_idx)
    
    src_x_indices = src_x_indices[valid_src_mask]
    src_z_indices = src_z_indices[valid_src_mask]
    n_shots_valid = len(src_x_indices)
    
    # Debug informazioni sorgenti dopo aver fatto il filtraggio
    print(f"\n{'='*60}")
    print(f"DEBUG - SORGENTI (DOPO IL FILTRAGGIO)")
    print(f"{'='*60}")
    print(f"  - Range valido X: [{pml_width}, {max_x_idx}] indici")
    print(f"  - Range valido Z: [{pml_width}, {max_z_idx}] indici")
    print(f"  - Sorgenti valide: {n_shots_valid}/{n_shots}")
    print(f"  - Indici X sorgenti valide: [{src_x_indices.min()}, {src_x_indices.max()}]")
    print(f"  - Indici Z sorgenti valide: {src_z_indices[0]}")
    
    # Preparo le posizioni delle sorgenti (solo quelle valide dalla geometria)
    source_locations = torch.zeros(n_shots_valid, 1, 2, dtype=torch.long, device=device)
    source_locations[:, 0, 0] = src_x_indices
    source_locations[:, 0, 1] = src_z_indices
    
    print(f"  - Dimensione source_locations: {source_locations.shape}")
    print(f"  - Prima sorgente [x, z]: [{source_locations[0, 0, 0]}, {source_locations[0, 0, 1]}]")
    print(f"  - Ultima sorgente [x, z]: [{source_locations[-1, 0, 0]}, {source_locations[-1, 0, 1]}]")
    
    # Converto coordinate ricevitori in indici
    rec_x_indices = torch.tensor(rec_coordinates[:, 0] / dx + pml_width, dtype=torch.long)
    rec_z_indices = torch.tensor(rec_coordinates[:, 1] / dz + pml_width, dtype=torch.long)
    
    # Debug informazioni ricevitori
    print(f"\n{'='*60}")
    print(f"DEBUG - RICEVITORI (PRIMA DEL FILTRAGGIO)")
    print(f"{'='*60}")
    print(f"  - Numero totale ricevitori: {n_receivers_per_shot}")
    print(f"  - Coordinate X ricevitori (metri): [{rec_coordinates[:, 0].min():.1f}, {rec_coordinates[:, 0].max():.1f}]")
    print(f"  - Coordinate Z ricevitori (metri): {rec_coordinates[0, 1]:.1f}")
    print(f"  - Indici X ricevitori (con PML): [{rec_x_indices.min()}, {rec_x_indices.max()}]")
    print(f"  - Indici Z ricevitori (con PML): {rec_z_indices[0]}")
    
    # Filtro i ricevitori che sono fuori dal modello
    valid_rec_mask = (rec_x_indices >= pml_width) & (rec_x_indices <= max_x_idx) & \
                     (rec_z_indices >= pml_width) & (rec_z_indices <= max_z_idx)
    
    rec_x_indices = rec_x_indices[valid_rec_mask]
    rec_z_indices = rec_z_indices[valid_rec_mask]
    n_receivers_valid = len(rec_x_indices)
    
    # Debug informazioni ricevitori dopo il filtraggio
    print(f"\n{'='*60}")
    print(f"DEBUG - RICEVITORI (DOPO IL FILTRAGGIO)")
    print(f"{'='*60}")
    print(f"  - Ricevitori validi: {n_receivers_valid}/{n_receivers_per_shot}")
    print(f"  - Indici X ricevitori validi: [{rec_x_indices.min()}, {rec_x_indices.max()}]")
    print(f"  - Indici Z ricevitori validi: {rec_z_indices[0]}")
    
    # Preparo le posizioni dei ricevitori
    receiver_locations = torch.zeros(n_shots_valid, n_receivers_valid, 2, dtype=torch.long, device=device)
    for shot_idx in range(n_shots_valid):
        receiver_locations[shot_idx, :, 0] = rec_x_indices
        receiver_locations[shot_idx, :, 1] = rec_z_indices
    
    print(f"  - Dimensioni receiver_locations: {receiver_locations.shape}")
    print(f"  - Primo ricevitore dello shot 0 [x, z]: [{receiver_locations[0, 0, 0]}, {receiver_locations[0, 0, 1]}]")
    print(f"  - Ultimo ricevitore dello shot 0 [x, z]: [{receiver_locations[0, -1, 0]}, {receiver_locations[0, -1, 1]}]")
    
    # Aggiorno il numero di shots e ricevitori
    n_shots = n_shots_valid
    n_receivers_per_shot = n_receivers_valid
    
    # Genero il wavelet di Ricker per tutte le sorgenti
    source_amplitudes = (
        deepwave.wavelets.ricker(f0, nt, dt_sec, peak_time)
        .repeat(n_shots, 1, 1)
        .to(device)
    )
    
    # Debug informazioni wavelet
    print(f"\n{'='*60}")
    print(f"DEBUG - WAVELET SORGENTE")
    print(f"{'='*60}")
    print(f"  - Frequenza dominante: {f0} Hz")
    print(f"  - Peak time: {peak_time:.4f} s")
    print(f"  - Shape source_amplitudes: {source_amplitudes.shape}")
    print(f"  - Min ampiezza wavelet: {source_amplitudes.min():.10f}")
    print(f"  - Max ampiezza wavelet: {source_amplitudes.max():.10f}")
    print(f"  - Media ampiezza wavelet: {source_amplitudes.mean():.10f}")
    print(f"  - Std ampiezza wavelet: {source_amplitudes.std():.10f}")
    print(f"  - Indice del picco: {source_amplitudes[0, 0].argmax()}")
    print(f"  - Valore al picco: {source_amplitudes[0, 0].max():.10f}")
    
    # Configurazione del forward modeling con tutti i parametri da considerare
    print(f"\n{'='*60}")
    print(f"CONFIGURAZIONE FORWARD MODELING")
    print(f"{'='*60}")
    print(f"  - Dimensioni modello: {v.shape}")
    print(f"  - Numero shots: {n_shots}")
    print(f"  - Ricevitori per shot: {n_receivers_per_shot}")
    print(f"  - Campioni temporali: {nt}")
    print(f"  - dt: {dt_sec} s")
    print(f"  - Frequenza dominante: {f0} Hz")
    print(f"  - Accuracy: {accuracy}")
    print(f"  - PML width: {pml_width}")
    print(f"  - Device: {device}")
    
    # Eseguo il forward modeling con Deepwave
    print(f"\n{'='*60}")
    print(f"ESECUZIONE FORWARD MODELING...")
    print(f"{'='*60}")
    
    # Funzione principale di Deepwave per il forward modeling che ci permette di implementare la propagazione di onde acustiche nel mezzo
    out = scalar(
        v, dx, dt_sec,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=accuracy,
        pml_width=pml_width,
        pml_freq=f0
    )
    
    receiver_amplitudes = out[-1]
    
    # Debug informazioni output ottenuto dopo aver eseguito il forward modeling
    print(f"\n{'='*60}")
    print(f"OUTPUT FORWARD MODELING")
    print(f"{'='*60}")
    print(f"  - Shape receiver_amplitudes: {receiver_amplitudes.shape}")
    print(f"  - Min ampiezza ricevuta: {receiver_amplitudes.min():.10f}")
    print(f"  - Max ampiezza ricevuta: {receiver_amplitudes.max():.10f}")
    print(f"  - Media ampiezza ricevuta: {receiver_amplitudes.mean():.10f}")
    print(f"  - Std ampiezza ricevuta: {receiver_amplitudes.std():.10f}")
    print(f"  - Mediana ampiezza ricevuta: {receiver_amplitudes.median():.10f}")
    
    # Controllo finale per vedere se ci sono valori anomali o fuori dalla geometria
    n_zeros = (receiver_amplitudes == 0).sum().item()
    n_total = receiver_amplitudes.numel()
    pct_zeros = (n_zeros / n_total) * 100
    
    # Analisi complessiva dei dati ricevuti per verificare la qualità del forward modeling e la presenza di eventuali anomalie
    print(f"\n{'='*60}")
    print(f"ANALISI DATI RICEVUTI")
    print(f"{'='*60}")
    print(f"  - Elementi totali: {n_total}")
    print(f"  - Elementi zero: {n_zeros} ({pct_zeros:.2f}%)")
    print(f"  - Tutti zeri? {(receiver_amplitudes == 0).all().item()}")
    print(f"  - NaN? {torch.isnan(receiver_amplitudes).any().item()}")
    print(f"  - Inf? {torch.isinf(receiver_amplitudes).any().item()}")
    
    # Analisi per shot
    print(f"\n  Analisi shot centrale (shot {n_shots//2}):")
    mid_shot = receiver_amplitudes[n_shots//2]
    print(f"     - Min: {mid_shot.min():.10f}")
    print(f"     - Max: {mid_shot.max():.10f}")
    print(f"     - Std: {mid_shot.std():.10f}")
    
    # Analisi per ricevitore
    print(f"\n  Analisi ricevitore centrale (ricevitore {n_receivers_per_shot//2}):")
    mid_rec = receiver_amplitudes[:, n_receivers_per_shot//2, :]
    print(f"     - Min: {mid_rec.min():.10f}")
    print(f"     - Max: {mid_rec.max():.10f}")
    print(f"     - Std: {mid_rec.std():.10f}")
    
    # Prendo un subset per calcolare i percentili se il tensor è troppo grande (Consigliato da Claude.ai - Da riguardare assolutamente).
    if receiver_amplitudes.numel() > 10000000:
        sample_data = receiver_amplitudes.flatten()[::100]  # Prendi 1 ogni 100 valori
        p5, p95 = torch.quantile(sample_data, torch.tensor([0.05, 0.95]).to(device))
    else:
        p5, p95 = torch.quantile(receiver_amplitudes, torch.tensor([0.05, 0.95]).to(device))
    
    print(f"\n  Percentili per visualizzazione:")
    print(f"     - 5°:  {p5:.10f}")
    print(f"     - 95°: {p95:.10f}")
    
    print(f"{'='*60}\n")
    
    return receiver_amplitudes, source_amplitudes


def main(args):
    """
    Funzione principale per il forward modeling con Deepwave
    """
    
    # Configura il device (GPU se disponibile)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"CONFIGURAZIONE SISTEMA")
    print(f"{'='*60}")
    print(f"  - Device utilizzato: {device}")
    if device.type == 'cuda':
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - Memoria GPU disponibile: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Carica il modello di velocità
    print(f"\n{'='*60}")
    print(f"CARICAMENTO MODELLO")
    print(f"{'='*60}")
    print(f"  - Path: {args.vp_model_path}")
    
    npzfile = np.load(args.vp_model_path)
    vp = npzfile["vp"]
    spacing = npzfile["spacing"]
    
    if vp.max() < 10:  # stare attenti alle conversioni km/s a m/s
        print(f"  ATTENZIONE: conversione in m/s se è in km/s - Importante")
        vp = vp * 1000.0
    
    # Stampo informazioni modello di velocità
    print(f"  - Dimensioni modello: {vp.shape}")
    print(f"  - Spaziatura griglia: {spacing}")
    print(f"  - Min/Max velocità: {vp.min():.2f} / {vp.max():.2f} m/s")
    print(f"  - Dimensione fisica: {vp.shape[0]*spacing[0]} x {vp.shape[1]*spacing[1]} m")
    
    # Visualizzo il modello di velocità
    plt.figure(figsize=(12, 6))
    plt.imshow(vp.T, cmap='jet', aspect='auto', extent=[0, vp.shape[0]*spacing[0], 
                                                          vp.shape[1]*spacing[1], 0])
    plt.colorbar(label='Velocità (m/s)')
    plt.xlabel('Distanza (m)')
    plt.ylabel('Profondità (m)')
    plt.title('Modello di velocità')
    
    # Creo la directory di output se non esiste in cui salvare i risultati/le figure
    os.makedirs(os.path.join("data", "v_models"), exist_ok=True)
    plt.savefig(os.path.join("data", "v_models", 
                             os.path.splitext(os.path.basename(args.vp_model_path))[0] + "_deepwave.png"),
                bbox_inches='tight', dpi=150)
    plt.show()
    
    # Calcolo le coordinate delle sorgenti
    domain_size_x = vp.shape[0] * spacing[0]
    n_src = int(np.ceil(domain_size_x / args.src_spacing))
    src_coordinates = np.empty((n_src, 2))
    src_coordinates[:, 0] = np.arange(0, args.src_spacing * n_src, args.src_spacing)
    src_coordinates[:, 1] = args.src_depth
    
    # Calcolo le coordinate dei ricevitori
    n_rec = int(np.ceil(domain_size_x / args.rec_spacing))
    rec_coordinates = np.empty((n_rec, 2))
    rec_coordinates[:, 0] = np.arange(0, n_rec * args.rec_spacing, args.rec_spacing)
    rec_coordinates[:, 1] = args.rec_depth
    
    # Stampo informazioni acquisizione
    print(f"\n{'='*60}")
    print(f"CONFIGURAZIONE ACQUISIZIONE")
    print(f"{'='*60}")
    print(f"  SORGENTI:")
    print(f"    - Numero: {n_src}")
    print(f"    - Spaziatura: {args.src_spacing} m")
    print(f"    - Profondità: {args.src_depth} m")
    print(f"    - Range X: {src_coordinates[:, 0].min():.1f} - {src_coordinates[:, 0].max():.1f} m")
    print(f"  RICEVITORI:")
    print(f"    - Numero: {n_rec}")
    print(f"    - Spaziatura: {args.rec_spacing} m")
    print(f"    - Profondità: {args.rec_depth} m")
    print(f"    - Range X: {rec_coordinates[:, 0].min():.1f} - {rec_coordinates[:, 0].max():.1f} m")
    
    # Eseguo il forward modeling con la funzione definita sopra
    receiver_amplitudes, source_amplitudes = forward_modeling_deepwave(
        vp=vp,
        spacing=spacing,
        src_coordinates=src_coordinates,
        rec_coordinates=rec_coordinates,
        f0=args.f0,
        tn=args.tn,
        dt=args.dt,
        device=device,
        accuracy=args.accuracy,
        pml_width=args.pml_width
    )
    
    # Stampo informazioni finali sui dati ricevuti
    print(f"\n{'='*60}")
    print(f"FORWARD MODELING COMPLETATO!")
    print(f"{'='*60}")
    print(f"  - Dimensioni dati ricevuti: {receiver_amplitudes.shape}")
    
    # Converto i risultati in numpy per il salvataggio (contrario di quanto fatto all'inizio)
    receiver_data = receiver_amplitudes.cpu().numpy()
    
    # Visualizzo uno shot centrale
    middle_shot = len(receiver_data) // 2
    vmin, vmax = np.percentile(receiver_data[middle_shot], [5, 95])
    
    print(f"\n{'='*60}")
    print(f"VISUALIZZAZIONE SHOT CENTRALE (shot {middle_shot})")
    print(f"{'='*60}")
    print(f"  - Clip values: vmin={vmin:.6f}, vmax={vmax:.6f}")
    
    plt.figure(figsize=(12, 8))
    plt.imshow(receiver_data[middle_shot].T, cmap='gray', aspect='auto', 
               vmin=vmin, vmax=vmax)
    plt.colorbar(label='Ampiezza')
    plt.xlabel('Canale ricevitore')
    plt.ylabel('Campione temporale')
    plt.title(f'Shot {middle_shot} (centrale)')
    
    # Salvo la figura nel path di output
    os.makedirs(os.path.join("data", "shots"), exist_ok=True)
    plt.savefig(os.path.join("data", "shots", 
                             os.path.splitext(os.path.basename(args.vp_model_path))[0] + "_deepwave.png"),
                bbox_inches='tight', dpi=150)
    plt.show()
    
    # Visualizzo un common receiver gather (centrale)
    middle_receiver = receiver_data.shape[1] // 2
    vmin, vmax = np.percentile(receiver_data[:, middle_receiver, :], [5, 95])
    
    print(f"\n{'='*60}")
    print(f"VISUALIZZAZIONE COMMON RECEIVER GATHER (ricevitore {middle_receiver})")
    print(f"{'='*60}")
    print(f"  - Clip values: vmin={vmin:.6f}, vmax={vmax:.6f}")
    
    plt.figure(figsize=(12, 8))
    plt.imshow(receiver_data[:, middle_receiver, :].T, cmap='gray', aspect='auto',
               vmin=vmin, vmax=vmax)
    plt.colorbar(label='Ampiezza')
    plt.xlabel('Numero shot')
    plt.ylabel('Campione temporale')
    plt.title(f'Common Receiver Gather - Ricevitore {middle_receiver}')
    plt.savefig(os.path.join("data", "shots", 
                             os.path.splitext(os.path.basename(args.vp_model_path))[0] + "_crg_deepwave.png"),
                bbox_inches='tight', dpi=150)
    plt.show()
    
    # Salvo i dati in un file .npz
    out_path = os.path.join("data", "shots", 
                           os.path.splitext(os.path.basename(args.vp_model_path))[0] + "_deepwave.npz")
    
    print(f"\n{'='*60}")
    print(f"SALVATAGGIO DATI")
    print(f"{'='*60}")
    print(f"  - Path: {out_path}")
    
    np.savez(
        out_path,
        receiver_data=receiver_data,
        src_coordinates=src_coordinates,
        rec_coordinates=rec_coordinates,
        spacing=spacing,
        f0=args.f0,
        dt=args.dt,
        tn=args.tn,
        vp_shape=vp.shape
    )
    
    # Messaggio di completamento finale
    print(f"\n{'='*60}")
    print(f"ELABORAZIONE COMPLETATA!")
    print(f"{'='*60}\n")
    
    return None


if __name__ == "__main__":
    """
    Esegue la funzione principale
    """
    
    # Parse degli argomenti
    parser = argparse.ArgumentParser(description='Forward Modeling con Deepwave')
    parser.add_argument('--vp_model_path', type=str, 
                       default="./data/v_models/marmousi_sp25.npz",
                       help='Path al modello di velocità')
    parser.add_argument('--accuracy', type=int, default=8,
                       help='Ordine di accuratezza spaziale (2, 4, 6, 8)')
    parser.add_argument('--pml_width', type=int, default=20,
                       help='Larghezza del PML in numero di celle')
    parser.add_argument('--src_spacing', type=float, default=100,
                       help='Spaziatura tra le sorgenti (m)')
    parser.add_argument('--rec_spacing', type=float, default=25,
                       help='Spaziatura tra i ricevitori (m)')
    parser.add_argument('--rec_depth', type=float, default=20,
                       help='Profondità dei ricevitori (m)')
    parser.add_argument('--src_depth', type=float, default=20,
                       help='Profondità delle sorgenti (m)')
    parser.add_argument('--f0', type=float, default=25,
                       help='Frequenza dominante della sorgente (Hz)')
    parser.add_argument('--tn', type=float, default=3000,
                       help='Tempo finale di registrazione (ms)')
    parser.add_argument('--dt', type=float, default=2,
                       help='Passo temporale (ms)')
    
    args = parser.parse_args()
    
    # Eseguiamo la funzione principale
    main(args)