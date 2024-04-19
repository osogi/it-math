# Task-3: Сингулярное разложение и сжатие изображений

## Структура сжатого файла

| Name   | Offset | Type   | Description   |
| ------ | ------ | ------ | ------------- |
| header | 0x0    | header | File metadata |
| body   | 0x10   | body   | File payload  |

### Header
First 16 bytes of the file are allocated for the header.

| Name | Offset | Type    | Description                       |
| ---- | ------ | ------- | --------------------------------- |
| n    | 0x0    | uint32  | Horizontal size of original image |
| m    | 0x4    | uint32  | Vertical size of original image   |
| k    | 0x8    | uint32  | One of sizes of stored matrices   |
| mode | 0xC    | char[4] | Order of saved channels           |

### Body
After file header comes main body, consisting of several consecutive channels (in accordance with channels in the original image).

| Name     | Type   | Description                           |
| -------- | ------ | ------------------------------------- |
| channels | chan[] | Compressed channels of original image |

#### Channel
Each channel is described by `n*k+k+k*m` 32 bit floating point numbers:
- first `n*k` floats define the matrix U
- next `k` define the diagonal of matrix S
- next `k*m` define the matrix Vh
Where matrices U, S and Vh correspond to matrices $`U`$, $`S`$, and $`V^*`$, from [here](https://en.wikipedia.org/wiki/Singular_value_decomposition#Compact_SVD)

| Name | Type         | Description              |
| ---- | ------------ | ------------------------ |
| U    | float32[n*k] | First matrix             |
| S    | float32[k]   | Second matrix (diagonal) |
| Vh   | float32[k*m] | Third matrix             |
