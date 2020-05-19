from utils.host import Host
import pytest


def test_host_type():
    host = Host.type('sciencebirds:8080')
    assert host.hostname == 'sciencebirds'
    assert host.port == '8080'


def test_host_type_invalid():
    with pytest.raises(ValueError):
        host = Host.type('sciencebirds')
