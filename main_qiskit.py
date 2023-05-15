""" Execution script for the simulation with Qiskit.

Note:
    File as migrated from https://github.com/mvischi/Noisy-Quantum-Gates/blob/parallel/main/main_IBM.py.
"""


from numpy import savetxt
import os
import multiprocessing
import copy
import concurrent.futures
import time

from qiskit.providers.aer import AerSimulator

from configuration.token import IBM_TOKEN
from src.utility.simulations_utility import create_qc_list, fix_counts, load_config, setup_backend
from main.circuits import hadamard_reverse_QFT_circ
from src.utility.ibm_noise_model import construct_ibm_noise_model
from src.utility.device_parameters import DeviceParameters


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
    qc_list = create_qc_list(circuit_generator, nqubits_list, qubits_layout, backend)

    # Main calculations
    args = {
        'qc': qc_list,
        'backend': backend,
        'qubits_layout': qubits_layout,
        'shots': shots,
        'nqubits_list': nqubits_list,
        'folder_results': folder_results,
        'location_device_parameters': location_device_parameters
    }

    # Implement optimization of shots vs ncores
    ncores = int(0.8 * multiprocessing.cpu_count())
    N_process = [ncores for i in range(int(n_experiment / ncores))]
    N_process.append(n_experiment % ncores)

    # Run
    for n in N_process:
        mock_perform_parallel_simulation(
            args=args,
            simulation=do_simulation,
            max_workers=n
        )
    return


def do_simulation(arg_dict):
    """ This is the inside function that will take specific parameters multiple processes will start executing this
        function.
    """

    for nqubits in arg_dict['nqubits_list']:

        print(f"Start simulating {nqubits} qubits.")
        start = time.time()

        # Load device parameters (noise)
        device_param = DeviceParameters(qubits_layout=arg_dict['qubits_layout'][0:nqubits])
        device_param.load_from_texts(location=arg_dict["location_device_parameters"])

        # Create noise model
        noise_model = construct_ibm_noise_model(
            backend=arg_dict['backend'],
            qubits_layout=arg_dict['qubits_layout'],
            device_param=device_param
        )

        # Create simulator
        sim_noise = AerSimulator(noise_model=noise_model, method='statevector')

        # Run simulator
        job = sim_noise.run(arg_dict['qc'], shots=arg_dict['shots'])

        # Postprocess result
        result = job.result()
        counts_0 = result.get_counts()
        counts = fix_counts(nqubits, counts_0)
        p_ibm = [counts[j][1] / arg_dict['shots'] for j in range(0, 2 ** nqubits)]
        k = arg_dict['process']

        print(f"It took {time.time() - start} s to simulate {nqubits} qubits.")

        # Save result
        if not os.path.exists(arg_dict['folder_results']):
            os.makedirs(arg_dict['folder_results'])
        savetxt(arg_dict['folder_results'] + "/QFT_%dqubits_IBM_%d.txt" % (nqubits, k), p_ibm)
    return


def perform_parallel_simulation(args: dict, simulation: callable, max_workers: int):
    """ The .map method allows to execute the function simulation N_process times simultaneously by preserving the order
        of the given comprehension list. We create a deepcopy of the argument, the simulation currently modifies it
        during execution.
    """
    # Build function arguments
    arguments = [copy.deepcopy({**args, 'process': k + 1}) for k in range(max_workers)]

    # Execute parallel simulations
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(simulation, arguments)


def mock_perform_parallel_simulation(args: dict, simulation: callable, max_workers: int):
    """ This function mocks the parallel simulation. It is useful for debugging, because the error messages are
        displayed. In the real parallel simulation, they are muted.
     """
    # Build function arguments
    arguments = [copy.deepcopy({**args, 'process': k + 1}) for k in range(max_workers)]

    for arg in arguments:
        simulation(arg)


if __name__ == '__main__':

    # Load configuration
    config = load_config("IBM_configuration.json")
    run_config = config["run"]
    backend_config = config["backend"]

    # Setup backend
    backend = setup_backend(IBM_TOKEN, **backend_config)

    # Run main
    main(backend, do_simulation, circuit_generator=hadamard_reverse_QFT_circ, **run_config)