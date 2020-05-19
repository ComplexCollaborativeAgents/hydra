from dataclasses import dataclass


@dataclass
class Host:
    hostname: str
    port: str

    @classmethod
    def type(cls, host: str):
        try:
            hostname, port = host.split(':')
            return cls(hostname, port)
        except ValueError:
            raise ValueError('Host must have the format: hostname:port')
