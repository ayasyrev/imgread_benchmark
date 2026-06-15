import cffi

ffi = cffi.FFI()
ffi.cdef("void *tjInitDecompress();")
try:
    lib = ffi.dlopen("libjpeg.so.8")
    print("libjpeg.so.8 loaded")
    try:
        lib.tjInitDecompress()
        print("tjInitDecompress found")
    except Exception as e:
        print(f"tjInitDecompress not found: {e}")
except Exception as e:
    print(f"Failed to load libjpeg.so.8: {e}")
