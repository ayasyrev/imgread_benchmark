import os
import time


def record_decode(data):
    with open(os.environ["IMGREAD_TEST_DECODE_LOG"], "ab", buffering=0) as stream:
        stream.write(bytes(data))
    return data[0]


def crash_decode(data):
    if data[0] == 2:
        os._exit(19)
    return data[0]


def hang_decode(data):
    if data[0] == 2:
        time.sleep(60)
    return data[0]


def invalid_decode(data):
    raise ValueError("initializer decoder failed")


def hold_lease(descriptor, connection):
    from imgread_benchmark.encoded_cache import open_cache

    with open_cache(descriptor) as lease:
        connection.send(bytes(lease.views[0]))
        connection.recv()
    connection.close()
