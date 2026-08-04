import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--tensorboard-log-dir",
        action="store",
        default="logs/Gr00t_TRL_loco/.tensorboard",
        help="Directory containing tensorboard logs",
    )


@pytest.fixture
def tensorboard_log_dir(request):
    return request.config.getoption("--tensorboard-log-dir")
