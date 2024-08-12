from logging import getLogger

logger = getLogger(__name__)


class SessionManager(object):
    """Manager class for keeping experimental session

    This server keeps description of experiment 
    """

    def __init__(self,
                 cooling_down_id: str, experiment_username: str, sample_name: str, save_path: str
                 ) -> None:
        """Initializer of session manager

        Initialize connection to labrad servers and users

        Args:
            cooling_down_id (str): Identifier of current cooling down
            experiment_username (str): User name
        Raises:
            ValueError: experiment_username is empty
        """

        if len(experiment_username) == 0:
            raise ValueError("experiment_username must not be empty.")

        self.cooling_down_id = cooling_down_id
        self.experiment_username = experiment_username
        self.sample_name = sample_name
        if save_path[-1] != "/" and save_path[-2:] != "\\":
            save_path += "/"
        self.save_path = save_path + f"{self.cooling_down_id}/{self.sample_name}/"


    def __repr__(self) -> str:
        """String representation of session

        Returns:
            str: string representation
        """
        s = "Session\n"
        s += "* coolingdown {}\n".format(self.cooling_down_id)
        s += "* user        {}\n".format(self.experiment_username)
        s += "* sample      {}\n".format(self.sample_name)
        return s
