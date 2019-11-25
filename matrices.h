#include <iostream>
#include <vector>
#include <array>
#include <sstream>
#include <string>

/*
    An N-Dimensional Matrix Implementation
*/
template<unsigned int N>
class HyperMatrix
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
    HyperMatrix(std::array<int, N> shape);

    /// <summary>Create a Matrix with shape `shape` with values in `values`</summary>
    /// <param name=shape>Array containing the sizes of each dimension of the Matrix</param>
    /// <param name=values>Vector of values for the matrix. Vector should be single dimensional</param>
    HyperMatrix(std::array<int, N> shape, std::vector<double> values);

    ~HyperMatrix();

    /// <summary> Creates an N dimensional Matrix of 0s with shape `shape` </summary>
    /// <param name=shape> Array containing the sizes of each dimension of the Martix</param>
    static HyperMatrix<N> Zeros(std::array<int, N> shape);

    /// <summary> Creates an N dimensional Matrix of 0s with shape `shape` </summary>
    /// <param name=shape> Array containing the sizes of each dimension of the Matrix </param>
    static HyperMatrix<N> Ones(std::array<int, N> shape);

    /// <summary> Creates an N dimensional Identity Matrix with shape `shape` </summary>
    /// <param name=shape> Array containing the sizes of each dimension of the Matrix </param>
    static HyperMatrix<N> Identity(std::array<int, N> shape);

    /// <summary> Add two N dimensional Hyper Matrices element-wise
    /// <param name=A> HyperMatrix for addition
    /// <param name=B> HyperMatrix for addition 
    static HyperMatrix<N> Add(HyperMatrix<N> A, HyperMatrix<N> B);

    /// <summary> Multiply Matrix A by Scalar scalar s
    /// <param name=A> HyperMatrix to be multiplied by s
    /// <param name=s> Scalar to be multiplied by A
    static HyperMatrix<N> ScalarMultiply(HyperMatrix<N> A, double s);

    /// <summary> Multiply two N dimensional Hyper Matrices
    /// <param name=A> HyperMatrix for multiplication
    /// <param name=B> HyperMatrix for multiplication 
    static HyperMatrix<N> MatrixMultiply(HyperMatrix<N> A, HyperMatrix<N> B);

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
HyperMatrix<N>::HyperMatrix(std::array<int, N> shape)
{
    this->shape = shape;
    this->calculateStride();
}

template<unsigned int N>
HyperMatrix<N>::HyperMatrix(std::array<int, N> shape, std::vector<double> values)
{
    this->shape = shape;
    this->values = values;
    this->calculateStride();
}

template<unsigned int N>
HyperMatrix<N>::~HyperMatrix()
{}

template<unsigned int N>
HyperMatrix<N> HyperMatrix<N>::Zeros(std::array<int, N> shape)
{
    int length = 1;
    for (int i = 0; i < N; i++)
        length *= shape[i];
    std::vector<double> values(length, 0.0);
    return HyperMatrix<N>(shape, values);
}

template<unsigned int N>
HyperMatrix<N> HyperMatrix<N>::Ones(std::array<int, N> shape)
{
    int length = 1;
    for (int i = 0; i < N; i++)
        length *= shape[i];
    std::vector<double> values(length, 1.0);
    return HyperMatrix<N>(shape, values);
}

template<unsigned int N>
HyperMatrix<N> HyperMatrix<N>::Identity(std::array<int, N> shape)
{
    // Identity must be "square" so all dimensions must be equal
    for (int i = 0; i < N-1; i++)
        if (shape[i] != shape[i+1])
            throw;

    HyperMatrix<N> identity = HyperMatrix<N>::Zeros(shape);
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
HyperMatrix<N> HyperMatrix<N>::Add(HyperMatrix<N> A, HyperMatrix<N> B)
{
    for (int i = 0; i < N; i++)
    {
        if (A.shape[i] != B.shape[i])
        {
            std::cout << "Cannot Add Matrices of Different Shapes!" << std::endl;
            throw;
        }
    }

    HyperMatrix<N> result = HyperMatrix::Zeros(A.getShape());
    for (int i = 0; i < A.values.size(); i++)
        result.values[i] = A.values[i] + B.values[i];

    return result;
}

template<unsigned int N>
HyperMatrix<N> HyperMatrix<N>::ScalarMultiply(HyperMatrix<N> A, double s)
{
    HyperMatrix<N> result = HyperMatrix<N>::Zeros(A.shape);
    for (int i = 0; i < result.values.size(); i++)
        result.values[i] = s*A.values[i];

    return result;
}

template<unsigned int N>
HyperMatrix<N> HyperMatrix<N>::MatrixMultiply(HyperMatrix<N> A, HyperMatrix<N> B)
{
    if (N != 2)
    {
        std::cout << "Higher Dimension Multiplication Not Implimented" << std::endl;
        throw;
    }

    // TODO: Try to do this without index translations
    std::array<int, N> new_shape = {A.shape[0], B.shape[1]};
    std::vector<double> new_values(A.shape[0] * B.shape[1]);
    std::array<int, N> strides = HyperMatrix<N>::CalculateStride(new_shape);
    int k_dim = A.shape[1];
    
    for (int i = 0; i < new_shape[1]; i++)
    for (int j = 0; j < new_shape[0]; j++)
    {
        double value = 0;
        for (int k = 0; k < k_dim; k++)
            value += A.At({i,k}) * B.At({k,j});

        int idx = HyperMatrix<N>::ConvertIndex({i,j}, strides);
        new_values[idx] = value;
    }

    HyperMatrix<N> C(new_shape, new_values);

    return C;
}

template<unsigned int N>
int HyperMatrix<N>::GetDims()
{
    return this->dims;
}

template<unsigned int N>
double HyperMatrix<N>::At(std::array<int, N> indices)
{
    int idx = this->convertIndex(indices);
    return this->values[idx];
}

template<unsigned int N>
void HyperMatrix<N>::calculateStride()
{
    this->strides = HyperMatrix<N>::CalculateStride(this->shape);
}

template<unsigned int N>
std::array<int, N> HyperMatrix<N>::CalculateStride(std::array<int, N> shape)
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
int HyperMatrix<N>::ConvertIndex(std::array<int, N> indices, std::array<int, N> strides)
{
    int idx = 0;
    for (int i = 0; i < N; i++)
        idx += indices[i] * strides[i];
    return idx;
}

template<unsigned int N>
int HyperMatrix<N>::convertIndex(std::array<int, N> indices)
{
    return HyperMatrix<N>::ConvertIndex(indices, this->strides);
}

// TODO: Try to print values similar to numpy style...or anything really?
template<unsigned int N>
std::string HyperMatrix<N>::toString() const
{
    std::stringstream ss;

    if (N == 2)
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
std::array<int,N> HyperMatrix<N>::getShape()
{
    return this->shape;
}

template<unsigned int N>
std::array<int, N> HyperMatrix<N>::getStrides()
{
    return this->strides;
}

template<unsigned int N>
std::vector<double> HyperMatrix<N>::getValues()
{
    return this->values;
}

template<unsigned int N>
inline std::ostream &operator<<(std::ostream &os, HyperMatrix<N> const &M)
{
    return os << std::string(M);
}

template<unsigned int N>
inline HyperMatrix<N> operator-(const HyperMatrix<N>A)
{
    return HyperMatrix<N>::ScalarMultiply(A,-1);
}

template<unsigned int N>
inline HyperMatrix<N> operator+(const HyperMatrix<N> A, const HyperMatrix<N> B)
{
    return HyperMatrix<N>::Add(A,B);
}

template<unsigned int N>
inline HyperMatrix<N> operator-(const HyperMatrix<N> A, const HyperMatrix<N> B)
{
    return HyperMatrix<N>::Add(A, -B);
}

template<unsigned int N>
inline HyperMatrix<N> operator*(const HyperMatrix<N> A, const HyperMatrix<N> B)
{
    return HyperMatrix<N>::MatrixMultiply(A,B);
}