"""Allow ``python -m multi_scenario.cli ...`` to invoke the Typer app.

The OVH container job uses this entry form (see
:class:`OvhJobConfig.command_template`); keeping it stable preserves the
remote-runner contract.
"""

from multi_scenario.cli import main

if __name__ == "__main__":
    main()
