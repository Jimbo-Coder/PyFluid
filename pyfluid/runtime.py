DEFAULT_NUMBA_THREADS = 4


def configure_numba_threads(threads=None):
   import numba

   selected = DEFAULT_NUMBA_THREADS if threads is None else int(threads)
   if selected <= 0:
      raise ValueError("numba threads must be positive")
   numba.set_num_threads(selected)
   return numba.get_num_threads()
