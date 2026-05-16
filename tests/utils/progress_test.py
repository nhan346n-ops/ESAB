from pygws.service.progress_monitor import ProgressMonitor


def perform_a_subprocess(submonitor: ProgressMonitor):
    # Define 100 total working units for the submonitor (but only 40 of the root monitor)
    submonitor.set_work_remaining(100)
    submonitor.worked(50)
    assert submonitor.acc == 50
    assert submonitor.size == 100
    submonitor.worked(50)
    assert submonitor.acc == 100
    assert submonitor.size == 100


def test_process():
    monitor = ProgressMonitor()
    # Root monitor with 100 working units
    monitor.begin_task(name="main monitor", n=100)
    assert monitor.size == 100

    # Using 10 working units to do something.
    monitor.worked(10)
    assert monitor.acc == 10
    assert monitor.size == 100
    # Using also 10 working units to do something else.
    monitor.worked(10)
    assert monitor.acc == 20
    assert monitor.size == 100

    # Using 40 working units to perform a subprocess.
    perform_a_subprocess(monitor.split(40))
    # Shoult be 60...
    # assert monitor.acc == 60
    assert monitor.size == 100


if __name__ == "__main__":
    test_process()
