# Seismic-Inversion-Deepwave
Forward Modeling + Full-Waveform inversion (FWI) implementation using Deepwave library

FORWARD MODELING WITH DEEPWAVE:
The Deepwave library used in the context of seismic inversion, in particular, in the implementation of forward modeling, simulates the propagations of seismic waves with variable velocities and records the signal received by the receivers.

1) Propagation in the velocity model.
2) Configuration of temporal parameters.
3) Spatial grid management + PML (perfectly matched layer --> artificial absorbing boundary to prevent reflections from the edge of the computational domain) + coordinate conversions.
4) Filtering of the non-useful sources/receivers that are out of the geometry model considered.
5) Creations of position tensors both for sources and receivers. Output type [numberOfShots, 1, 2] where 1 means 1 source per shot and 2 means 2 coordinates (x, z). For the receivers: [numberOfShots, numberOfReceivers, 2].
6) Ricker Wavelet generation --> seismic impulse with f = f0. Same wavelet for each shot.
7) Execution of forward modeling using scalar (scalar resolves the acoustic wave equation in 2D. It uses finite differences with accuracy to the 8th order. It considers boundary PML). Output: receiver amplitudes --> [numberOfShots, numberOfReceivers, nt].

MAIN FUNCTION:
1) Load the velocity model.
2) Check unit of measures.
3) Geometry set-up.
4) Forward Modeling execution.
5) Visualization and saving in .npz format.

FULL-WAVEFORM INVERSION WITH DEEPWAVE:
Try to find the true velocity model starting from the observed data, using an iterative optimization process.

1) Creation of the initial model --> smoothed version of the true model. It is very similar but it does not contains details since it could converge to local minima if it is too different from the true one.
2) Forward modeling for FWI --> it builts a computational graph to calculate the gradient.
3) Calculation of the loss and the gradient.
4) FWI iterations with Batching --> all the shots together = too much memory. Gradients are accumulated through the batches. Backpropagation.
5) Gradient clipping so we avoid instable updates of the gradient.
6) Optimization loop --> it calculates loss and gradient, it clips the gradient and calculates the gradient descent step considering the constraints regarding the maximum and minimum velocities.
7) Monitoring and saving. It studies parameters such as: loss, model error (RMSE between true model and the current one), GradNorm (gradient magnitude) and Model Change (measure that defines how does the model changes at each iteration).
