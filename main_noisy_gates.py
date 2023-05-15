""" Execution script for the simulation with the Noisy gates approach.

Note:
    File as migrated from https://github.com/mvischi/Noisy-Quantum-Gates/blob/parallel/main/main_NG.py.
"""


import os
import numpy as np
import time
import copy

from quantum_gates.utilities import load_config, setup_backend
from quantum_gates.utilities import DeviceParameters
from quantum_gates.quantum_algorithms import hadamard_reverse_qft_circ
from quantum_gates.circuits import EfficientCircuit
from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.gates import legacy_gates
from quantum_gates.utilities import multiprocessing_parallel_simulation as perform_parallel_simulation

from configuration.token import IBM_TOKEN
from src.utilities import create_qc_list


def main(backend,
         do_simulation,
         n_experiment,
         shots: int,
         splits: int,
         qubits_list: list,
         qubits_layout: list,
         folder_results: str,
         location_device_parameters: str,
         circuit_generator: callable):

    # Create the list of qubit number and compile the circuit for each
    qc_list = create_qc_list(circuit_generator, qubits_list, qubits_layout, backend)

    # Prepare the arguments
    device_param = DeviceParameters(qubits_layout=qubits_layout)
    device_param.load_from_texts(location=location_device_parameters)
    device_param = device_param.__dict__()
    time_as_str = time.strftime("%Y%m%d-%H%M%S")

    # Perform split of shots -> Otherwise, arg_list would consume to much memory and computing the shots for high number
    # of qubits would take too much time.
    assert shots % splits == 0, \
        f"main() assumes the number of shots ({shots}) to be divisible by the splits ({splits}), but found otherwise."

    for i in range(splits):
        print(f"Start split {i+1}/{splits}.", flush=True)
        # Create the arguments for each simulator run in this split
        arg_list = [
            {
                't_qiskit_circ': copy.deepcopy(qc),
                'qubits_layout': copy.deepcopy(qubits_layout[:nqubits]),
                'shots': shots // splits,
                'nqubits': nqubits,
                'folder_results': folder_results,
                'device_param': copy.deepcopy(device_param),
                'process': k,
                'split': i,
                'time_as_str': time_as_str,
                'nqubit': nqubits,
            } for k in range(n_experiment) for (nqubits, qc) in zip(qubits_list, qc_list)
        ]

        # Execute simulations
        cpu_count = os.cpu_count()
        print(f"We have {cpu_count} CPUs. We use the same number as max_workers.", flush=True)
        perform_parallel_simulation(args=arg_list, simulation=do_simulation, max_workers=cpu_count)
        print(f"Finished split {i+1}/{splits}.", flush=True)
    return


def do_simulation(args: dict) -> tuple:
    """ This is the inside function that will be executed in parallel.
        Returns a tuple (time, nqubit) that tells how long it took to do the simulation.
    """
    # Set random seed, otherwise each experiment gets the same result
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    # Setup initial state
    psi0 = np.zeros(2**args['nqubits'])
    psi0[0] = 1

    # Create simulator
    sim = MrAndersonSimulator(gates=legacy_gates, CircuitClass=EfficientCircuit, parallel=False)

    # Timeit
    start = time.time()

    # Run simulator
    print(f"Run simulation with {args['nqubits']} nqubits.", flush=True)
    p_ng = sim.run(
        t_qiskit_circ=args['t_qiskit_circ'],
        qubits_layout=args['qubits_layout'],
        psi0=psi0,
        shots=args['shots'],
        device_param=args['device_param'],
        nqubit=args['nqubit']
    )
    total_time = time.time() - start
    print(f"Took {total_time} s for {args['shots']} shots with {args['nqubits']} qubits.", flush=True)

    # Save results
    time_as_str = args['time_as_str']
    if not os.path.exists(args["folder_results"] + "/" + time_as_str):
        os.makedirs(args["folder_results"] + "/" + time_as_str)
    np.savetxt(
        args["folder_results"] + "/" + time_as_str + '/bis_QFT_%dqubits_NOISEGATES_%d_%d.txt' % (args['nqubits'], args['process'], args['split']),
        p_ng
    )
    return total_time, args['nqubits']


if __name__ == '__main__':

    # Load configuration
    config = load_config("test_noisy_gates.json")
    run_config = config["run"]
    backend_config = config["backend"]

    # Setup backend
    backend = setup_backend(IBM_TOKEN, **backend_config)

    # Run main
    main(backend, do_simulation, circuit_generator=hadamard_reverse_qft_circ, **run_config)
