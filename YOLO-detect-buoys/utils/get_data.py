# Path: YOLO-detect-buoys/utils/get_data.py

"""Get data from Roboflow."""

from os import getenv

from dotenv import load_dotenv
from roboflow import Roboflow
from roboflow.core.dataset import Dataset


def get_data_roboflow() -> Dataset:
    """Get data from Roboflow.

    Do not expose the API key in the code.
    """
    # import environment variables

    load_dotenv()

    rf = Roboflow(api_key=getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace(the_workspace=getenv("WORKSPACE")).project(
        project_id=getenv("ROBOFLOW_PROJECT_ID")
    )
    version = project.version(version_number=int(getenv("ROBOFLOW_PROJECT_VERSION")))
    dataset = version.download(
        model_format=getenv("DATASET_FORMAT"), overwrite=True, location="./data"
    )

    return dataset


def get_data(*args, **kwargs):
    """Get data from Roboflow.

    Interface function for any method of getting data.
    """
    return get_data_roboflow(*args, **kwargs)
