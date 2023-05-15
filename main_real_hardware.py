""" Execution script for the real run.

Note:
    File as migrated from https://github.com/mvischi/Noisy-Quantum-Gates/blob/parallel/main/main_REAL.py.
"""

import numpy as np

from quantum_gates.utilities import fix_counts, load_config, setup_backend
from quantum_gates.utilities import DeviceParameters
from quantum_gates.quantum_algorithms import hadamard_reverse_qft_circ

from configuration.token import IBM_TOKEN
from src.utilities import create_qc_list


def main(backend,
         shots: int,
         min_nqubits: int,
         max_nqubits: int,
         qubits_layout: list,
         folder_results: str,
         location_device_parameters: str,
         circuit_generator: callable):
    """ Runs the experiment on real quantum hardware.
    """

    # Setup qubit number list
    nqubits_list = list(range(min_nqubits, max_nqubits + 1))

    # Save the backend parameters
    device_param = DeviceParameters(qubits_layout=qubits_layout)
    device_param.load_from_backend(backend)
    device_param.save_to_texts(location=location_device_parameters)

    # Create the circuits list and do the transpile
    qc_list = create_qc_list(circuit_generator, nqubits_list, qubits_layout, backend)

    # Run the circuits
    job = backend.run(qc_list, shots=shots)
    result = job.result()

    # Postprocess and save the result
    for nqubit in nqubits_list:
        counts_0 = result.get_counts(qc_list[nqubit-min_nqubits])
        counts = fix_counts(nqubit, counts_0)
        p_real = [counts[j][1]/shots for j in range(0, 2**nqubit)]
        np.savetxt(f'{folder_results}/QFT_%dqubits_DEVICE.txt' % nqubit, p_real)

    print('REAL DEVICE FINISHED')
    return


if __name__ == '__main__':

    # Load configuration
    config = load_config(filename="real_hardware.json")
    run_config = config["run"]
    backend_config = config["backend"]

    # Setup backend
    backend = setup_backend(IBM_TOKEN, **backend_config)

    # Run main
    main(backend, circuit_generator=hadamard_reverse_qft_circ, **run_config)
