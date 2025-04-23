from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Sequence, Any

import numpy as np
import numpy.typing as npt
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")
tilenum:int = 20


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)

class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")
        blocksize:int = int(len(out)//tilenum)
        if((len(out_strides) == len(in_strides)) and (out_strides == in_strides).all() and (out_shape == in_shape).all()):
            for i in prange(tilenum):
                curaddr = i*blocksize
                for j in range(blocksize):
                    out[curaddr+j] = fn(in_storage[curaddr+j])
                    # print(len(out),curaddr+j)
            for i in range(len(out)%tilenum):
                out[tilenum*blocksize+i] = fn(in_storage[tilenum*blocksize+i])
                # print(len(out),tilenum*blocksize+i)
            # print(len(out),tilenum,blocksize)
        else:
            for j in prange(tilenum):
                for i in range(blocksize):
                    curaddr = j*blocksize
                    out_idx:npt.NDArray[np.int32] = np.array([0] * len(out_shape))
                    to_index(curaddr+i, out_shape, out_idx)
                    in_idx:npt.NDArray[np.int32] = np.array([0] * len(in_shape))
                    broadcast_index(out_idx, out_shape, in_shape, in_idx)
                    in_pos = index_to_position(in_idx, in_strides)
                    out[index_to_position(out_idx,out_strides)] = fn(in_storage[in_pos])
            for j in range(len(out)%tilenum):
                out_idx:npt.NDArray[np.int32] = np.array([0] * len(out_shape))
                to_index(tilenum*blocksize+j, out_shape, out_idx)
                in_idx:npt.NDArray[np.int32] = np.array([0] * len(in_shape))
                broadcast_index(out_idx, out_shape, in_shape, in_idx)
                in_pos = index_to_position(in_idx, in_strides)
                out[index_to_position(out_idx,out_strides)] = fn(in_storage[in_pos])

        # print(in_storage,in_strides,out,out_strides)

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")
        blocksize:int = int(len(out)//tilenum)
        if((len(out_strides) == len(a_strides)) and (len(out_strides) == len(b_strides)) and (out_strides == a_strides).all() and (out_strides == b_strides).all() and (out_shape == a_shape).all() and (out_shape == b_shape).all()):
            for i in prange(tilenum):
                curaddr = i*blocksize
                for j in range(blocksize):
                    out[curaddr+j] = fn(a_storage[curaddr+j],b_storage[curaddr+j])
            for i in range(len(out)%tilenum):
                out[tilenum*blocksize+i] = fn(a_storage[tilenum*blocksize+i],b_storage[tilenum*blocksize+i])
        else:
            for j in prange(tilenum):
                for i in range(blocksize):
                    out_idx:npt.NDArray[np.int32] = np.array([0] * len(out_shape))
                    to_index(blocksize*j+i, out_shape, out_idx)
                    a_idx:npt.NDArray[np.int32] = np.array([0] * len(a_shape))
                    broadcast_index(out_idx, out_shape, a_shape, a_idx)
                    a_pos = index_to_position(a_idx, a_strides)

                    b_idx:npt.NDArray[np.int32] = np.array([0] * len(b_shape))
                    broadcast_index(out_idx, out_shape, b_shape, b_idx)
                    b_pos = index_to_position(b_idx, b_strides)

                    out[index_to_position(out_idx,out_strides)] = fn(a_storage[a_pos], b_storage[b_pos])
            
            for j in range(len(out)%tilenum):
                out_idx:npt.NDArray[np.int32] = np.array([0] * len(out_shape))
                to_index(tilenum*blocksize+j, out_shape, out_idx)
                a_idx:npt.NDArray[np.int32] = np.array([0] * len(a_shape))
                broadcast_index(out_idx, out_shape, a_shape, a_idx)
                a_pos = index_to_position(a_idx, a_strides)

                b_idx:npt.NDArray[np.int32] = np.array([0] * len(b_shape))
                broadcast_index(out_idx, out_shape, b_shape, b_idx)
                b_pos = index_to_position(b_idx, b_strides)

                out[index_to_position(out_idx,out_strides)] = fn(a_storage[a_pos], b_storage[b_pos])

        # print(a_storage,a_strides,b_storage,b_strides,out,out_strides)

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")
        blocksize:int = int(len(out)//tilenum)
        for j in prange(tilenum):
            for i in range(blocksize):
                out_idx:npt.NDArray[np.int32] = np.array([0] * len(out_shape))
                to_index(j*blocksize+i, out_shape, out_idx)

                a_idx:npt.NDArray[np.int32] = np.array([0] * len(a_shape))
                broadcast_index(out_idx, out_shape, a_shape, a_idx)

                for k in range(a_shape[reduce_dim]):
                    a_idx[reduce_dim] = k
                    a_pos = index_to_position(a_idx, a_strides)
                    out[index_to_position(out_idx,out_strides)] += a_storage[a_pos]

        for j in range(len(out)%tilenum):
            out_idx:npt.NDArray[np.int32] = np.array([0] * len(out_shape))
            to_index(tilenum*blocksize+j, out_shape, out_idx)

            a_idx:npt.NDArray[np.int32] = np.array([0] * len(a_shape))
            broadcast_index(out_idx, out_shape, a_shape, a_idx)

            for k in range(a_shape[reduce_dim]):
                a_idx[reduce_dim] = k
                a_pos = index_to_position(a_idx, a_strides)
                out[index_to_position(out_idx,out_strides)] += a_storage[a_pos]

        # print(a_storage,a_strides,out,out_strides)


    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # TODO: Implement for Task 3.2.
    # raise NotImplementedError("Need to implement for Task 3.2")
    assert a_shape[-1] == b_shape[-2]
    # tmpshapea = a_shape.tolist()
    # tmpshapeb = b_shape.tolist()
    # tmpshapea[-1]=1
    # tmpshapeb[-2]=1
    # tmp_out_shape = list(shape_broadcast(tuple(tmpshapea),tuple(tmpshapeb)))
    # out_shape = np.array(tmp_out_shape)
    for i in prange(len(out)):
        tmpi = i + 0
        out_idx:npt.NDArray[np.int32] = np.array([0] * len(out_shape))
        to_index(tmpi, out_shape, out_idx)
        out_pos = index_to_position(out_idx, out_strides)
        for j in range(a_shape[-1]):
            tmpj = j + 0
            
            tmp_a_idx:npt.NDArray[np.int32] = out_idx.copy()
            tmp_a_idx[-1]=tmpj
            a_idx:npt.NDArray[np.int32] = np.array([0] * len(a_shape))
            broadcast_index(tmp_a_idx, out_shape, a_shape, a_idx)
            # a_idx[-2] = out_idx[-2]
            tmp_b_idx:npt.NDArray[np.int32] = out_idx.copy()
            tmp_b_idx[-2]=tmpj
            b_idx:npt.NDArray[np.int32] = np.array([0] * len(b_shape))
            broadcast_index(tmp_b_idx, out_shape, b_shape, b_idx)
            # b_idx[-1] = out_idx[-1]


            # a_idx[-1]=tmpj
            # b_idx[-2]=tmpj
            a_pos = index_to_position(a_idx, a_strides)
            b_pos = index_to_position(b_idx, b_strides)
            out[out_pos] += a_storage[a_pos]*b_storage[b_pos]
            # print(a_idx,b_idx)
    # print(a_storage,a_shape,b_storage,b_shape,out,out_shape)


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
