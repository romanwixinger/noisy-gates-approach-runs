""" Execution script for the simulation with Qiskit.

Note:
    File as migrated from https://github.com/mvischi/Noisy-Quantum-Gates/blob/parallel/main/main_IBM.py.

Todo:
    Check if the multiprocessing now works.
"""


from numpy import savetxt
import os
import multiprocessing
import time

from qiskit.providers.aer import AerSimulator

from quantum_gates.utilities import fix_counts, load_config, setup_backend
from quantum_gates.utilities import DeviceParameters
from quantum_gates.utilities import mock_parallel_simulation as perform_parallel_simulation
from quantum_gates.quantum_algorithms import hadamard_reverse_qft_circ

from configuration.token import IBM_TOKEN
from src.utilities import construct_ibm_noise_model, create_qc_list


def main(backend,
         do_simulation,
         n_experiment: int,
         shots: int,
         min_nqubits: int,
         max_nqubits: int,
         qubits_layout: list,
         folder_results: str,
         location_device_parameters: str,
         circuit_generator: callable):

    nqubits_list = list(range(min_nqubits, max_nqubits + 1))
    qc_list = create_qc_list(
        circuit_generator=circuit_generator,
        nqubits_list=nqubits_list, qubits_layout=qubits_layout,
        backend=backend
    )

    # Main calculations
    args = [{
        'qc': qc,
        'backend': backend,
        'qubits_layout': qubits_layout,
        'shots': shots,
        'nqubits': nqubits,
        'folder_results': folder_results,
        'location_device_parameters': location_device_parameters,
        'process': process
    } for (nqubits, qc) in zip(nqubits_list, qc_list) for process in range(n_experiment)]

    max_workers = int(0.5 * multiprocessing.cpu_count())

    # Run
    perform_parallel_simulation(
        args=args,
        simulation=do_simulation,
        max_workers=max_workers
    )
    return


def do_simulation(arg_dict: dict):
    """ This is the inside function that will take specific parameters multiple processes will start executing this
        function.
    """

    # Extract args
    nqubits = arg_dict['nqubits']
    qubits_layout = arg_dict['qubits_layout']
    location = arg_dict["location_device_parameters"]
    backend = arg_dict['backend']
    qc = arg_dict['qc']
    shots = arg_dict['shots']
    folder_results = arg_dict['folder_results']
    process = arg_dict['process']

    print(f"Start simulating {nqubits} qubits.")
    start = time.time()

    # Load device parameters (noise)
    device_param = DeviceParameters(qubits_layout=qubits_layout)
    device_param.load_from_texts(location=location)
    device_param.check_T1_and_T2_times(do_raise_exception=True)

    # Create noise model
    noise_model = construct_ibm_noise_model(
        backend=backend,
        qubits_layout=qubits_layout,
        device_param=device_param
    )

    # Create simulator
    sim_noise = AerSimulator(noise_model=noise_model, method='statevector')

    # Run simulator
    job = sim_noise.run(qc, shots=shots)

    # Postprocess result
    result = job.result()
    counts_0 = result.get_counts()
    counts = fix_counts(nqubits, counts_0)
    p_ibm = [counts[j][1] / shots for j in range(0, 2 ** nqubits)]

    print(f"It took {time.time() - start} s to simulate {nqubits} qubits.")

    # Save result
    if not os.path.exists(folder_results):
        os.makedirs(folder_results)
    savetxt(f"{folder_results}/QFT_{nqubits}qubits_IBM_{process}.txt", p_ibm)

    return


if __name__ == '__main__':

    # Load configuration
    config = load_config("qiskit.json")
    run_config = config["run"]
    backend_config = config["backend"]

    # Setup backend
    backend = setup_backend(IBM_TOKEN, **backend_config)

    # Run main
    main(backend, do_simulation, circuit_generator=hadamard_reverse_qft_circ, **run_config)
