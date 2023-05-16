""" Execution script for the real run.

Note:
    File as migrated from https://github.com/mvischi/Noisy-Quantum-Gates/blob/parallel/main/main_REAL.py.
"""

from quantum_gates.utilities import load_config, setup_backend, DeviceParameters

from configuration.token import IBM_TOKEN
from qiskit_ibm_provider import IBMProvider


def check_backends(Token: str, hub: str, group: str, project: str):
    """Takes the backend configuration and prints the available backends.

    Args:
        Token (str): Token generated with the IBM Quantum Experience account.
        hub (str): Hub name of the account where the project is located.
        group (str): Group name of the account.
        project (str): Project under which the user has access to the device.
    """
    IBMProvider.delete_account()
    IBMProvider.save_account(token=Token)
    provider = IBMProvider(instance=hub+'/'+group+'/'+project)
    print([backend.name for backend in provider.backends()])
    return


def check_device_parameters(
        backend,
        qubits_layout: list,
        location_device_parameters: str):
    """ Runs the experiment on real quantum hardware.
    """

    # Check
    device_param = DeviceParameters(qubits_layout=qubits_layout)
    device_param.load_from_backend(backend)
    device_param.check_T1_and_T2_times(do_raise_exception=True)
    return


if __name__ == '__main__':

    # Load configuration
    config = load_config(filename="real_hardware.json")
    run_config = config["run"]
    backend_config = config["backend"]

    backend = setup_backend(IBM_TOKEN, **backend_config)
    check_device_parameters(
        backend=backend,
        qubits_layout=config["run"]["qubits_layout"],
        location_device_parameters=config["run"]["location_device_parameters"]
    )


