""" Execution script for the real run.

Note:
    File as migrated from https://github.com/mvischi/Noisy-Quantum-Gates/blob/parallel/main/main_REAL.py.
"""

from quantum_gates.utilities import load_config

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


if __name__ == '__main__':

    # Load configuration
    config = load_config(filename="real_hardware.json")
    run_config = config["run"]
    backend_config = config["backend"]

    # Check which backends are available
    check_backends(
        Token=IBM_TOKEN,
        hub=backend_config["hub"],
        group=backend_config["group"],
        project=backend_config["project"]
    )
