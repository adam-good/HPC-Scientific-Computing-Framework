#include <iostream>
#include <vector>
#include <array>
#include <sstream>
#include <string>
#include <functional>

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

    /// <summary> Tests A and B for equality
    /// <param name=A> HyperMatrix for equality test
    /// <param name=B> HyperMatrix for equality test
    static bool Equals(HyperMatrix<N> A, HyperMatrix<N> B);

    /// <summary> Add two N dimensional Hyper Matrices element-wise
    /// <param name=A> HyperMatrix for addition
    /// <param name=B> HyperMatrix for addition 
    static HyperMatrix<N> Add(HyperMatrix<N> A, HyperMatrix<N> B);

    /// <summary> Add two N dimensional Hyper Matrices element-wise
    /// <param name=A> HyperMatrix for subtraction
    /// <param name=B> HyperMatrix for subtraction 
    static HyperMatrix<N> Subtract(HyperMatrix<N> A, HyperMatrix<N> B);

    /// <summary> Multiply Matrix A by Scalar scalar s
    /// <param name=A> HyperMatrix to be multiplied by s
    /// <param name=s> Scalar to be multiplied by A
    static HyperMatrix<N> ScalarMultiply(HyperMatrix<N> A, double s);

    /// <summary> Standard Dot Product
    /// <param name=A> Vector for multiplication
    /// <param name=B> Vector for multiplication 
    static HyperMatrix<N> VectorProduct(HyperMatrix<N> A, HyperMatrix<N> B);

    /// <summary> Multiply two N dimensional Hyper Matrices
    /// <param name=A> Right HyperMatrix for multiplication
    /// <param name=B> Left HyperMatrix for multiplication 
    static HyperMatrix MatrixProduct(HyperMatrix<N> A, HyperMatrix<N> B);

    /// <summary> Add all elements of the matrix for a total sum
    /// <param name=A> HyperMatrix to total
    static double Sum(HyperMatrix<N> A);

    /// <summary> Compare two N dimensional Hyper Matrices and return the larger one
    /// <param name=A> HyperMatrix to compare
    /// <param name=B> HyperMatrix to compare
    static double LargerSum(HyperMatrix<N> A, HyperMatrix<N> B);

    /// <summary> Compare two N dimensional Hyper Matrices and return the smaller one
    /// <param name=A> HyperMatrix to compare
    /// <param name=B> HyperMatrix to compare
    static double SmallerSum(HyperMatrix<N> A, HyperMatrix<N> B);

    /// <summary> Apply the function func to every value in the matrix. Returns matrix of new values. </summary>
    /// <param name=func> Function to be applied to values in Matrix </param>
    HyperMatrix<N> Apply(std::function<double(double)> func);

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
bool HyperMatrix<N>::Equals(HyperMatrix<N> A, HyperMatrix<N> B)
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
HyperMatrix<N> HyperMatrix<N>::Subtract(HyperMatrix<N> A, HyperMatrix<N> B)
{
    for (int i = 0; i < N; i++)
    {
        if (A.shape[i] != B.shape[i])
        {
            std::cout << "Cannot Subtract Matrices of Different Shapes!" << std::endl;
            throw;
        }
    }

    HyperMatrix<N> result = HyperMatrix::Zeros(A.getShape());
    for (int i = 0; i < A.values.size(); i++)
        result.values[i] = A.values[i] - B.values[i];

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
HyperMatrix<N> HyperMatrix<N>::VectorProduct(HyperMatrix<N> A, HyperMatrix<N> B)
{
    if (N != 1)
        throw;
    else if (A.shape[0] != B.shape[0])
    {
        std::cout << "Vectors must be same length for dot product!" << std::endl;
        throw;
    }
    int size = A.shape[0];

    double result = 0;
    for (int i = 0; i < size; i++)
        result += A.values[i] * B.values[i];

    return HyperMatrix<N>({1}, {result});
}

template<unsigned int N>
HyperMatrix<N> HyperMatrix<N>::MatrixProduct(HyperMatrix<N> A, HyperMatrix<N> B)
{
    if (N == 1)
    {
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
        
        // Determine New Shape
        std::vector<int> shape_vec(A.shape.begin(), A.shape.end());
        shape_vec[shape_vec.size()-1] = B.shape[N-1];
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
        int a_mat_size = A.shape[N-1] * A.shape[N-2];
        int b_mat_size = A.shape[N-1] * A.shape[N-2];

        // Initialize new values
        std::vector<double> new_values(new_size);
        int newval_idx = 0;
        int shared_dim = A.shape[N-1];
        for (int M = 0; M < num_matrices; M++)
        {
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

            for (int i = 0; i < a_vals.size() / shared_dim; i++)
            for (int j = 0; j < b_vals.size() / shared_dim; j++)
            {
                int dotprod = 0;
                for (int k = 0; k < shared_dim; k++)
                    dotprod += a_vals[i*A.strides[N-2] + k] * b_vals[j*A.strides[N-2] + k];//b_vals[k*B.strides[N-2] + j];
                
                new_values[newval_idx] = dotprod;
                newval_idx += 1;
            }
        }


        return HyperMatrix(new_shape, new_values);
    }
}

template<unsigned int N>
double HyperMatrix<N>::Sum(HyperMatrix<N> A)
{
    int result = 0;

    for (int i = 0; i < A.values.size(); i++)
        result += A.values[i];

    return result;
}

template<unsigned int N>
double HyperMatrix<N>::LargerSum(HyperMatrix<N> A, HyperMatrix<N> B)
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
double HyperMatrix<N>::SmallerSum(HyperMatrix<N> A, HyperMatrix<N> B)
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
HyperMatrix<N> HyperMatrix<N>::Apply(std::function<double(double)> func)
{
    int size = this->values.size();
    std::vector<double> newValues(size);
    for (int i = 0; i < size; i++)
        newValues[i] = func(this->values[i]);

    HyperMatrix<N> newMatrix(this->shape, newValues);
    return newMatrix;
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
inline bool operator==(const HyperMatrix<N> A, const HyperMatrix<N> B)
{
    return HyperMatrix<N>::Equals(A,B);
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
    return HyperMatrix<N>::Subtract(A,B);
}

template<unsigned int N>
inline HyperMatrix<N> operator*(const HyperMatrix<N> A, const HyperMatrix<N> B)
{
    return HyperMatrix<N>::MatrixProduct(A,B);
}
