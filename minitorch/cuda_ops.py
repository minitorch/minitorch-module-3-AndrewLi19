# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from numba import config
# config.CUDA_ENABLE_PYNVJITLINK = 1

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

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
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        out_dim = len(out_shape)
        in_strides_shared = cuda.shared.array(MAX_DIMS, numba.int32)
        out_strides_shared = cuda.shared.array(MAX_DIMS, numba.int32)
        in_shape_shared = cuda.shared.array(MAX_DIMS, numba.int32)
        out_shape_shared = cuda.shared.array(MAX_DIMS, numba.int32)
        pos = cuda.threadIdx.x
        if(pos < out_dim):
          in_strides_shared[pos] = in_strides[pos]
          out_strides_shared[pos] = out_strides[pos]
          in_shape_shared[pos] = in_shape[pos]
          out_shape_shared[pos] = out_shape[pos]
        cuda.syncthreads()
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        if(i<out_size):
            to_index(i,out_shape_shared[:out_dim],out_index)
            broadcast_index(out_index,out_shape_shared[:out_dim],in_shape_shared[:out_dim],in_index)
            out_pos=index_to_position(out_index,out_strides_shared[:out_dim])
            out[out_pos]=fn(in_storage[index_to_position(in_index,in_strides_shared[:out_dim])])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        if(i<out_size):
            to_index(i, out_shape, out_index)

            broadcast_index(out_index, out_shape, a_shape, a_index)
            a_pos = index_to_position(a_index, a_strides)

            broadcast_index(out_index, out_shape, b_shape, b_index)
            b_pos = index_to_position(b_index, b_strides)

            out[index_to_position(out_index,out_strides)] = fn(a_storage[a_pos], b_storage[b_pos])


    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")
    if(i<size):
        cache[pos] = a[i]
    else:
        cache[pos] = 0
    cuda.syncthreads()

    stride = cuda.blockDim.x//2
    while(stride>0):
        if(pos<stride):
            cache[pos]+=cache[pos+stride]
        cuda.syncthreads()
        stride = stride//2
    if(pos==0):
        out[cuda.blockIdx.x]=cache[0]

jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        # a_strides = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        out_dim = len(out_shape)
        a_strides_shared = cuda.shared.array(MAX_DIMS, numba.int32)
        out_strides_shared = cuda.shared.array(MAX_DIMS, numba.int32)
        a_shape_shared = cuda.shared.array(MAX_DIMS, numba.int32)
        out_shape_shared = cuda.shared.array(MAX_DIMS, numba.int32)
        if(pos < out_dim):
          a_strides_shared[pos] = a_strides[pos]
          out_strides_shared[pos] = out_strides[pos]
          a_shape_shared[pos] = a_shape[pos]
          out_shape_shared[pos] = out_shape[pos]
        cuda.syncthreads()

        
        if(out_pos<out_size):
            to_index(out_pos,out_shape_shared[:out_dim],out_index)
            new_out_pos = index_to_position(out_index,out_strides_shared[:out_dim])
            out[new_out_pos] = reduce_value
            reducedim = a_shape_shared[reduce_dim]
            if(pos<reducedim):
                out_index[reduce_dim] = pos
                a_pos = index_to_position(out_index,a_strides_shared[:out_dim])
                cache[pos] = a_storage[a_pos]
                
                stride = reducedim//2
                res = reducedim%2
                while(stride>0):
                    cuda.syncthreads()
                    if(pos < stride):
                        cache[pos] = fn(cache[pos],cache[pos+stride])
                    if(pos==0 and res!=0):
                        cache[pos] = fn(cache[pos],cache[pos+stride*2])
                    res = stride%2
                    stride = stride//2
                if(pos==0):
                    out[new_out_pos]=fn(out[new_out_pos],cache[pos])

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")
    # out_len = len(out)
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x = cuda.threadIdx.x
    y = cuda.threadIdx.y
    if(x<size and y<size):
        outitem = 0
        for i in range(size):
            outitem = outitem + a[x*size+i] * b[i*size+y]
        out[x*size+y] = outitem
jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    # raise NotImplementedError("Need to implement for Task 3.4")
    lenth = out_shape[-2]
    width = out_shape[-1]
    iterlenth = a_shape[-1]

    out_idx = cuda.local.array(3,numba.int32)
    out_idx[0] = batch
    out_idx[1] = i
    out_idx[2] = j
    out_pos = 0
    if(i<lenth and j<width):
        out_pos = index_to_position(out_idx,out_strides)
    a_idx = cuda.local.array(3,numba.int32)
    broadcast_index(out_idx,out_shape,a_shape,a_idx)
    b_idx = cuda.local.array(3,numba.int32)
    broadcast_index(out_idx,out_shape,b_shape,b_idx)
    tmp = 0
    total_tiles = (iterlenth + BLOCK_DIM - 1) // BLOCK_DIM
    for tile in range(total_tiles):
        a_idx[-2] = i
        a_idx[-1] = tile*BLOCK_DIM + pj
        if(a_idx[-2]<lenth and a_idx[-1]<iterlenth):
            a_shared[pi][pj] = a_storage[index_to_position(a_idx,a_strides)]
        else:
            a_shared[pi][pj] = 0
        b_idx[-2] = tile*BLOCK_DIM + pi
        b_idx[-1] = j
        if(b_idx[-2]<iterlenth and b_idx[-1]<width):
            b_shared[pi][pj] = b_storage[index_to_position(b_idx,b_strides)]
        else:
            b_shared[pi][pj] = 0
        cuda.syncthreads()
        for k in range(BLOCK_DIM):
            tmp=tmp+a_shared[pi][k]*b_shared[k][pj]
        cuda.syncthreads()

    if(i<lenth and j<width):
        out[out_pos]=tmp


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
