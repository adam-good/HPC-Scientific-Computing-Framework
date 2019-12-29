#include <iostream>
#include <vector>
#include <array>
#include <sstream>
#include <string>
#include <functional>

#define THREADS_PER_BLOCK 1024

__global__ void add_vectors(double* x, double* y, double* z, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        z[index] = x[index] + y[index];
}

__global__ void subtract_vectors(double* x, double* y, double* z, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        z[index] = x[index] - y[index];
}

__global__ void scalar_multiply(double *x, double *y, double s, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        y[index] = s*x[index];
}

// TODO: Implement with reduction
// http://cuda-programming.blogspot.com/2013/01/vector-dot-product-in-cuda-c.html
__global__ void element_product(double *x, double *y, double *z, int n)
{
    int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIndex < n)
        z[threadIndex] = x[threadIndex] * y[threadIndex];
}

__global__ void matrix_product(double *x, double *yt, double *z, int m, int n, int k)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ double temp[]; // size set when called. should be m*n*k*sizeof(double)

    if (index < m*k) {
        for (int i = index%m; i < m*k; i += m)
        {
            temp[m*k*(index/m) + i] = x[index] * yt[i];
        }
    }

    __syncthreads();

    if (index < m*k)
    {
        z[index] = 0;
        for (int i = 0; i < m; i++)
        {
            z[index] += temp[index*m+i];
        }
    }

    __syncthreads();
}

/*
    An N-Dimensional Matrix Implementation
*/
template<unsigned int N>
class HyperMatrix_CUDA
{
private:
    int dims = N;
    std::array<int, N> shape;
    std::array<int, N> strides;
    std::vector<double> values;

    void calculateStride();
    int convertIndex(std::array<int, N> indices);
    std::string toString() const;

public:
    /// <summary> Matrix </summary>
    /// <param name=shape>Array containing the sizes of each dimension of the Matrix</param>
    HyperMatrix_CUDA(std::array<int, N> shape);

    /// <summary>Create a Matrix with shape `shape` with values in `values`</summary>
    /// <param name=shape>Array containing the sizes of each dimension of the Matrix</param>
    /// <param name=values>Vector of values for the matrix. Vector should be single dimensional</param>
    HyperMatrix_CUDA(std::array<int, N> shape, std::vector<double> values);

    ~HyperMatrix_CUDA();

    /// <summary> Creates an N dimensional Matrix of 0s with shape `shape` </summary>
    /// <param name=shape> Array containing the sizes of each dimension of the Martix</param>
    static HyperMatrix_CUDA<N> Zeros(std::array<int, N> shape);

    /// <summary> Creates an N dimensional Matrix of 0s with shape `shape` </summary>
    /// <param name=shape> Array containing the sizes of each dimension of the Matrix </param>
    static HyperMatrix_CUDA<N> Ones(std::array<int, N> shape);

    /// <summary> Creates an N dimensional Identity Matrix with shape `shape` </summary>
    /// <param name=shape> Array containing the sizes of each dimension of the Matrix </param>
    static HyperMatrix_CUDA<N> Identity(std::array<int, N> shape);

    /// <summary> Tests A and B for equality
    /// <param name=A> HyperMatrix for equality test
    /// <param name=B> HyperMatrix for equality test
    static bool Equals(HyperMatrix_CUDA<N> A, HyperMatrix_CUDA<N> B);

    /// <summary> Add two N dimensional Hyper Matrices element-wise
    /// <param name=A> HyperMatrix for addition
    /// <param name=B> HyperMatrix for addition 
    static HyperMatrix_CUDA<N> Add(HyperMatrix_CUDA<N> A, HyperMatrix_CUDA<N> B);

    /// <summary> Add two N dimensional Hyper Matrices element-wise
    /// <param name=A> HyperMatrix for subtraction
    /// <param name=B> HyperMatrix for subtraction 
    static HyperMatrix_CUDA<N> Subtract(HyperMatrix_CUDA<N> A, HyperMatrix_CUDA<N> B);

    /// <summary> Multiply Matrix A by Scalar scalar s
    /// <param name=A> HyperMatrix to be multiplied by s
    /// <param name=s> Scalar to be multiplied by A
    static HyperMatrix_CUDA<N> ScalarMultiply(HyperMatrix_CUDA<N> A, double s);

    /// <summary> Standard Dot Product
    /// <param name=A> Vector for multiplication
    /// <param name=B> Vector for multiplication 
    static HyperMatrix_CUDA<N> VectorProduct(HyperMatrix_CUDA<N> A, HyperMatrix_CUDA<N> B);

    /// <summary> Multiply two N dimensional Hyper Matrices
    /// <param name=A> Right HyperMatrix for multiplication
    /// <param name=B> Left HyperMatrix for multiplication 
    static HyperMatrix_CUDA MatrixProduct(HyperMatrix_CUDA<N> A, HyperMatrix_CUDA<N> B);

    /// <summary> Add all elements of the matrix for a total sum
    /// <param name=A> HyperMatrix to total
    static double Sum(HyperMatrix_CUDA<N> A);

    /// <summary> Compare two N dimensional Hyper Matrices and return the larger one
    /// <param name=A> HyperMatrix to compare
    /// <param name=B> HyperMatrix to compare
    static double LargerSum(HyperMatrix_CUDA<N> A, HyperMatrix_CUDA<N> B);

    /// <summary> Compare two N dimensional Hyper Matrices and return the smaller one
    /// <param name=A> HyperMatrix to compare
    /// <param name=B> HyperMatrix to compare
    static double SmallerSum(HyperMatrix_CUDA<N> A, HyperMatrix_CUDA<N> B);

    /// <summary> Apply the function func to every value in the matrix. Returns matrix of new values. </summary>
    /// <param name=func> Function to be applied to values in Matrix </param>
    HyperMatrix_CUDA<N> Apply(std::function<double(double)> func);

    /// <summary> Return the number of dimensions N this matrix has </summary>
    int GetDims();

    /// <summary> returns value found at specified indices </summary>
    /// <param name=indices> Array containing values of indices for the requested value </param>
    double At(std::array<int, N> indices);

    /// <summary> Calculate stride based on given shape </summary>
    /// <param name=shape> Shape to be used to calculate the stride </param>
    static std::array<int, N> CalculateStride(std::array<int, N> shape);

    /// <summary> Convert a list of indices to their flat array equivalent given strides </summary>
    /// <param name=indices> Array containing values </param>
    /// <param name=strides> Array containing the strides to be used to translate the indices </param>
    static int ConvertIndex(std::array<int, N> indices, std::array<int,N> strides);

    // Getters
    std::array<int,N> getShape();
    std::array<int,N> getStrides();
    std::vector<double> getValues();

    operator std::string() const { return this->toString(); }
};

template<unsigned int N>
HyperMatrix_CUDA<N>::HyperMatrix_CUDA(std::array<int, N> shape)
{
    this->shape = shape;
    this->calculateStride();
}

template<unsigned int N>
HyperMatrix_CUDA<N>::HyperMatrix_CUDA(std::array<int, N> shape, std::vector<double> values)
{
    this->shape = shape;
    this->values = values;
    this->calculateStride();
}

template<unsigned int N>
HyperMatrix_CUDA<N>::~HyperMatrix_CUDA()
{}

template<unsigned int N>
HyperMatrix_CUDA<N> HyperMatrix_CUDA<N>::Zeros(std::array<int, N> shape)
{
    int length = 1;
    for (int i = 0; i < N; i++)
        length *= shape[i];
    std::vector<double> values(length, 0.0);
    return HyperMatrix_CUDA<N>(shape, values);
}

template<unsigned int N>
HyperMatrix_CUDA<N> HyperMatrix_CUDA<N>::Ones(std::array<int, N> shape)
{
    int length = 1;
    for (int i = 0; i < N; i++)
        length *= shape[i];
    std::vector<double> values(length, 1.0);
    return HyperMatrix_CUDA<N>(shape, values);
}

template<unsigned int N>
HyperMatrix_CUDA<N> HyperMatrix_CUDA<N>::Identity(std::array<int, N> shape)
{
    // Identity must be "square" so all dimensions must be equal
    for (int i = 0; i < N-1; i++)
        if (shape[i] != shape[i+1])
            throw;

    HyperMatrix_CUDA<N> identity = HyperMatrix_CUDA<N>::Zeros(shape);
    int dim = shape[0];

    for (int i = 0; i < dim; i++) 
    {
        int idx = 0;
        for (int j = 0; j < N; j++)
            idx += i * identity.getStrides()[j];
        identity.values[idx] = 1;
    }

    return identity;
}

template<unsigned int N>
bool HyperMatrix_CUDA<N>::Equals(HyperMatrix_CUDA<N> A, HyperMatrix_CUDA<N> B)
{
    for (int n = 0; n < N; n++)
        if (A.shape[n] != B.shape[n])
            return false;

    int size = A.values.size();

    for (int i = 0; i < size; i++)
        if (A.values[i] != B.values[i])
            return false;

    return true;
}

template<unsigned int N>
HyperMatrix_CUDA<N> HyperMatrix_CUDA<N>::Add(HyperMatrix_CUDA<N> A, HyperMatrix_CUDA<N> B)
{
    for (int i = 0; i < N; i++)
    {
        if (A.shape[i] != B.shape[i])
        {
            std::cout << "Cannot Add Matrices of Different Shapes!" << std::endl;
            throw;
        }
    }

    HyperMatrix_CUDA<N> result = HyperMatrix_CUDA::Zeros(A.getShape());
    // for (int i = 0; i < A.values.size(); i++)
    //     result.values[i] = A.values[i] + B.values[i];
    int size = A.values.size() * sizeof(double);
    double* x;
    double* y;
    double* z;
    cudaMalloc(&x, size);
    cudaMalloc(&y, size);
    cudaMalloc(&z, size);
    cudaMemcpy(x, A.values.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(y, B.values.data(), size, cudaMemcpyHostToDevice);

    int num_blocks = (A.values.size() + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
    add_vectors<<<num_blocks, THREADS_PER_BLOCK>>>(x,y,z, A.values.size());

    cudaDeviceSynchronize();
    cudaMemcpy(result.values.data(), z, size, cudaMemcpyDeviceToHost);

    return result;
}

template<unsigned int N>
HyperMatrix_CUDA<N> HyperMatrix_CUDA<N>::Subtract(HyperMatrix_CUDA<N> A, HyperMatrix_CUDA<N> B)
{
    for (int i = 0; i < N; i++)
    {
        if (A.shape[i] != B.shape[i])
        {
            std::cout << "Cannot Subtract Matrices of Different Shapes!" << std::endl;
            throw;
        }
    }

    HyperMatrix_CUDA<N> result = HyperMatrix_CUDA::Zeros(A.getShape());
    int size = A.values.size() * sizeof(double);
    double* x;
    double* y;
    double* z;
    cudaMalloc(&x, size);
    cudaMalloc(&y, size);
    cudaMalloc(&z, size);
    cudaMemcpy(x, A.values.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(y, B.values.data(), size, cudaMemcpyHostToDevice);

    int num_blocks = (A.values.size() + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
    subtract_vectors<<<num_blocks, THREADS_PER_BLOCK>>>(x,y,z, A.values.size());

    cudaDeviceSynchronize();
    cudaMemcpy(result.values.data(), z, size, cudaMemcpyDeviceToHost);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return result;
}

template<unsigned int N>
HyperMatrix_CUDA<N> HyperMatrix_CUDA<N>::ScalarMultiply(HyperMatrix_CUDA<N> A, double s)
{
    HyperMatrix_CUDA<N> result = HyperMatrix_CUDA<N>::Zeros(A.shape);
    // for (int i = 0; i < result.values.size(); i++)
    //     result.values[i] = s*A.values[i];
    int size = A.values.size() * sizeof(double);
    double* x;
    double* y;
    cudaMalloc(&x, size);
    cudaMalloc(&y, size);
    cudaMemcpy(x, A.values.data(), size, cudaMemcpyHostToDevice);

    int num_blocks = (A.values.size() + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
    scalar_multiply<<<num_blocks, THREADS_PER_BLOCK>>>(x,y,s,A.values.size());

    cudaDeviceSynchronize();
    cudaMemcpy(result.values.data(), y, size, cudaMemcpyDeviceToHost);
    cudaFree(x);
    cudaFree(y);

    return result;
}

template<unsigned int N>
HyperMatrix_CUDA<N> HyperMatrix_CUDA<N>::VectorProduct(HyperMatrix_CUDA<N> A, HyperMatrix_CUDA<N> B)
{
    if (N != 1)
        throw;
    else if (A.shape[0] != B.shape[0])
    {
        std::cout << "Vectors must be same length for dot product!" << std::endl;
        throw;
    }

    int size = A.shape[0] * sizeof(double);
    double* dx;
    double* dy;
    double* dz;
    cudaMalloc(&dx, size);
    cudaMalloc(&dy, size);
    cudaMalloc(&dz, size);
    cudaMemcpy(dx, A.values.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, B.values.data(), size, cudaMemcpyHostToDevice);

    int num_blocks = (A.values.size() + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
    element_product<<<num_blocks, THREADS_PER_BLOCK>>>(dx,dy,dz,A.shape[0]);

    double* hz = new double[A.shape[0]];
    cudaMemcpy(hz, dz, size, cudaMemcpyDeviceToHost);

    double result = 0;
    for (int i = 0; i < A.shape[0]; i++)
        result += hz[i];


    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    delete [] hz;

    return HyperMatrix_CUDA<N>({1}, {result});
}

template<unsigned int N>
HyperMatrix_CUDA<N> HyperMatrix_CUDA<N>::MatrixProduct(HyperMatrix_CUDA<N> A, HyperMatrix_CUDA<N> B)
{
    if (N == 1)
    {
        // TODO: Why won't this work?
        return VectorProduct(A,B);
    }
    else
    {
        // Compare dimensions to iterate over
        if (A.shape[N-1] != B.shape[N-2])
        {
            std::cout << "Improper Shape for N-Dimensional Matrix Multiplication" << std::endl;
            throw;
        }
        // A submatrix shape: mxn
        // B submatrix shape: nxk
        int m = A.shape[N-2];
        int n = A.shape[N-1]; // shared dimension
        int k = B.shape[N-1];
        
        // Determine New Shape
        // (same in all dimensions as A and B except last two: mxk)
        std::vector<int> shape_vec(A.shape.begin(), A.shape.end());
        shape_vec[shape_vec.size()-1] = k; // replace n with k in new shape
        std::array<int, N> new_shape;
        std::copy(shape_vec.begin(), shape_vec.end(), new_shape.begin());

        // Calculate New Size
        int new_size = 1;
        for (int i = 0; i < N; i++)
            new_size *= new_shape[i];

        // Determine the number of rank-2 hypermatrices are in the hypermatrix
        int num_matrices = 1;
        for (int i = 0; i < N-2; i++)
            num_matrices *= new_shape[i];
        
        // Calculate the size of the rank-2 hypermatrices
        int a_mat_size = m * n;
        int b_mat_size = n * k;
        int c_mat_size = m * k;

        // Initialize new values
        std::vector<double> new_values(new_size);
        int newval_idx = 0;
        int shared_dim = n;

        // Set up cuda stuff
        int a_cuda_size = a_mat_size * sizeof(double);
        int b_cuda_size = b_mat_size * sizeof(double);
        int c_cuda_size = c_mat_size * sizeof(double);
        double *dx, *dy, *dz, *hz;
        cudaMalloc(&dx, a_cuda_size); cudaMalloc(&dy, b_cuda_size); cudaMalloc(&dz, c_cuda_size);
        hz = new double[c_cuda_size];

        for (int M = 0; M < num_matrices; M++)
        {
            // set up values. transpose b for easy indexing
            std::vector<double> a_vals(A.values.begin() + M*a_mat_size, A.values.begin() + M*a_mat_size + a_mat_size);
            std::vector<double> b_vals(B.values.begin() + M*b_mat_size, B.values.begin() + M*b_mat_size + b_mat_size);
            std::vector<double> b_vals_transpose(b_mat_size);
            for (int i = 0; i < B.shape[N-2]; i++)
            {
                for (int j = 0; j < B.shape[N-1]; j++)
                {
                    int idx = i * B.shape[N-1] + j;
                    int idxT = j * B.shape[N-2] + i;
                    b_vals_transpose[idxT] = b_vals[idx];
                }
            }
            b_vals = b_vals_transpose;

            cudaMemcpy(dx, a_vals.data(), a_cuda_size, cudaMemcpyHostToDevice);
            cudaMemcpy(dy, b_vals.data(), b_cuda_size, cudaMemcpyHostToDevice);

            int num_blocks = (A.values.size() + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
            matrix_product<<<num_blocks, THREADS_PER_BLOCK, m*n*k*sizeof(double)>>>(dx,dy,dz,m,n,k);

            cudaMemcpy(hz, dz, c_cuda_size, cudaMemcpyDeviceToHost);

            for (int i = 0; i < c_mat_size; i++)
            {
                new_values[newval_idx] = hz[i];
                newval_idx += 1;
            }
        }

        cudaFree(dx); cudaFree(dy); cudaFree(dz);
        delete [] hz;

        cudaDeviceSynchronize();
        return HyperMatrix_CUDA(new_shape, new_values);
    }
}

template<unsigned int N>
double HyperMatrix_CUDA<N>::Sum(HyperMatrix_CUDA<N> A)
{
    int result = 0;

    for (int i = 0; i < A.values.size(); i++)
        result += A.values[i];

    return result;
}

template<unsigned int N>
double HyperMatrix_CUDA<N>::LargerSum(HyperMatrix_CUDA<N> A, HyperMatrix_CUDA<N> B)
{
    int Aresult = 0;
    int Bresult = 0;

    for (int i = 0; i < A.values.size(); i++)
        Aresult += A.values[i];
    for (int i = 0; i < B.values.size(); i++)
        Bresult += B.values[i];

    if (Aresult > Bresult)
    {
        return Aresult;
    }
    else
    {
        return Bresult;
    }
}

template<unsigned int N>
double HyperMatrix_CUDA<N>::SmallerSum(HyperMatrix_CUDA<N> A, HyperMatrix_CUDA<N> B)
{
    int Aresult = 0;
    int Bresult = 0;

    for (int i = 0; i < A.values.size(); i++)
        Aresult += A.values[i];
    for (int i = 0; i < B.values.size(); i++)
        Bresult += B.values[i];

    if (Aresult < Bresult)
    {
        return Aresult;
    }
    else
    {
        return Bresult;
    }
}

template<unsigned int N>
HyperMatrix_CUDA<N> HyperMatrix_CUDA<N>::Apply(std::function<double(double)> func)
{
    int size = this->values.size();
    std::vector<double> newValues(size);
    for (int i = 0; i < size; i++)
        newValues[i] = func(this->values[i]);

    HyperMatrix_CUDA<N> newMatrix(this->shape, newValues);
    return newMatrix;
}

template<unsigned int N>
int HyperMatrix_CUDA<N>::GetDims()
{
    return this->dims;
}

template<unsigned int N>
double HyperMatrix_CUDA<N>::At(std::array<int, N> indices)
{
    int idx = this->convertIndex(indices);
    return this->values[idx];
}

template<unsigned int N>
void HyperMatrix_CUDA<N>::calculateStride()
{
    this->strides = HyperMatrix_CUDA<N>::CalculateStride(this->shape);
}

template<unsigned int N>
std::array<int, N> HyperMatrix_CUDA<N>::CalculateStride(std::array<int, N> shape)
{
    // Row Major Strides (like numpy)
    // https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
    std::array<int, N> strides;
    for (int k = 0; k < N; k++)
    {
        strides[k] = 1;
        for (int j = k+1; j < N; j++)
            strides[k] *= shape[j];
    }

    return strides;
}

template<unsigned int N>
int HyperMatrix_CUDA<N>::ConvertIndex(std::array<int, N> indices, std::array<int, N> strides)
{
    int idx = 0;
    for (int i = 0; i < N; i++)
        idx += indices[i] * strides[i];
    return idx;
}

template<unsigned int N>
int HyperMatrix_CUDA<N>::convertIndex(std::array<int, N> indices)
{
    return HyperMatrix_CUDA<N>::ConvertIndex(indices, this->strides);
}

// TODO: Try to print values similar to numpy style...or anything really?
template<unsigned int N>
std::string HyperMatrix_CUDA<N>::toString() const
{
    std::stringstream ss;

    if (N == 1) // Vector Printing
    {
        ss << "[ ";
        for (int i = 0; i < values.size(); i++)
            ss << values[i] << " ";
        ss << "]";
    }
    else if (N == 2) // Matrix Printing
    {
        ss << "[ ";
        for (int i = 0; i < values.size(); i++)
        {
            ss << values[i] << " ";
            if ( (i+1) % shape[1] == 0 && (i+1) != values.size())
                ss << "\n  ";
        }
        ss << "]";
    }
    else
    {
        ss << "<HyperMatrix ";

        ss << "shape=[";
        for (int i = 0; i < N-1; i++)
            ss << this->shape[i] << ',';
        ss << this->shape[N-1] << ']';

        ss << '>';    
    }
    
    return ss.str();
}

template<unsigned int N>
std::array<int,N> HyperMatrix_CUDA<N>::getShape()
{
    return this->shape;
}

template<unsigned int N>
std::array<int, N> HyperMatrix_CUDA<N>::getStrides()
{
    return this->strides;
}

template<unsigned int N>
std::vector<double> HyperMatrix_CUDA<N>::getValues()
{
    return this->values;
}

template<unsigned int N>
inline std::ostream &operator<<(std::ostream &os, HyperMatrix_CUDA<N> const &M)
{
    return os << std::string(M);
}

template<unsigned int N>
inline bool operator==(const HyperMatrix_CUDA<N> A, const HyperMatrix_CUDA<N> B)
{
    return HyperMatrix_CUDA<N>::Equals(A,B);
}

template<unsigned int N>
inline HyperMatrix_CUDA<N> operator-(const HyperMatrix_CUDA<N>A)
{
    return HyperMatrix_CUDA<N>::ScalarMultiply(A,-1);
}

template<unsigned int N>
inline HyperMatrix_CUDA<N> operator+(const HyperMatrix_CUDA<N> A, const HyperMatrix_CUDA<N> B)
{
    return HyperMatrix_CUDA<N>::Add(A,B);
}

template<unsigned int N>
inline HyperMatrix_CUDA<N> operator-(const HyperMatrix_CUDA<N> A, const HyperMatrix_CUDA<N> B)
{
    return HyperMatrix_CUDA<N>::Subtract(A,B);
}

template<unsigned int N>
inline HyperMatrix_CUDA<N> operator*(const HyperMatrix_CUDA<N> A, const HyperMatrix_CUDA<N> B)
{
    return HyperMatrix_CUDA<N>::MatrixProduct(A,B);
}
