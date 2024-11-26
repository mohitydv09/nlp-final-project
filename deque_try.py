from collections import deque
from itertools import islice

class DequeSlice(deque):
    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(islice(self, index.start or 0, index.stop or len(self), index.step or 1))
        return super().__getitem__(index)

if __name__ == "__main__":
    dq = DequeSlice(range(10), 10)
    print(dq[2:5])