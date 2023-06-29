from logging import getLogger

logger = getLogger(__name__)


class SessionManager(object):
    """Manager class for keeping experimental session

    This server keeps description of experiment and provide
    ServerWrapper object of data vault and registry pointing at appropriate directory.
    """

    def __init__(self,
                 cooling_down_id: str, experiment_username: str, sample_name: str,
                 ) -> None:
        """Initializer of session manager

        Initialize connection to labrad servers and users

        Args:
            cooling_down_id (str): Identifier of current cooling down
            experiment_username (str): User name used for registry and data vault
        Raises:
            ValueError: experiment_username is empty
        """

        if len(experiment_username) == 0:
            raise ValueError("experiment_username must not be empty.")

        self.cooling_down_id = cooling_down_id
        self.experiment_username = experiment_username
        self.package_name = sample_name


    def __repr__(self) -> str:
        """String representation of session

        Returns:
            str: string representation
        """
        s = "Session\n"
        s += "* coolingdown {}\n".format(self.cooling_down_id)
        s += "* username    {}\n".format(self.experiment_username)
        s += "* package     {}\n".format(self.package_name)
        return s
