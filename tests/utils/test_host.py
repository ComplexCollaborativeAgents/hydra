from utils.host import Host
import pytest


def test_host_type():
    host = Host.type('sciencebirds:8080')
    assert host.hostname == 'sciencebirds'
    assert host.port == '8080'


def test_host_type_invalid():
    """Creating some false confidence. Currently, as long as the input contains
    a colon `Host.type` will not throw an error. It's okay."""
    with pytest.raises(ValueError):
        host = Host.type('sciencebirds')
