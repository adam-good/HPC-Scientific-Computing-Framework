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

    /// <summary> Return the number of dimensions N this matrix has </summary>
    int GetDims();

    /// <summary> returns value found at specified indices </summary>
    /// <param name=indices> Array containing values of indices for the requested value </param>
    double At(std::array<int, N> indices);

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

// TODO: Check for N dimensional iteration
// https://stackoverflow.com/questions/14040260/how-to-iterate-over-n-dimensions
template<unsigned int N>
HyperMatrix<N> HyperMatrix<N>::Identity(std::array<int, N> shape)
{
    // Not implemented. Will fix later
    throw;

    int length = 1;
    for (int i = 0; i < N; i++)
        length *= shape[i];
    std::vector<double> values(length, 0.0);


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
    // for (int k = 0; k < N; k++)
    // {
    //     strides[k] = 1;
    //     for (int j = k+1; j < N-1; j++)
    //         strides[k] *= this->shape[j];
    // }

    // for (int k = 0; k < N; k++) 
    // {
    //     strides[k] = 1;
    //     for (int j = 0; j < k-1; j++)
    //         strides[k] *= shape[j];
    // }

    this->strides[0] = 1;
    for (int i = 1; i < N; i++)
        strides[i] = shape[i-1]*strides[i-1];

    for (int i = 0; i < N; i++)
        std::cout << strides[i] << ',';
    std::cout << std::endl;
}

template<unsigned int N>
int HyperMatrix<N>::convertIndex(std::array<int, N> indices)
{
    int idx = 0;
    for (int i = 0; i < N; i++)
        idx += indices[i] * this->strides[i];
    return idx;
}

// TODO: Try to print values similar to numpy style...or anything really?
template<unsigned int N>
std::string HyperMatrix<N>::toString() const
{
    std::stringstream ss;
    ss << "<HyperMatrix ";

    ss << "shape=[";
    for (int i = 0; i < N-1; i++)
        ss << this->shape[i] << ',';
    ss << this->shape[N-1] << ']';

    // int length = 1;
    // for (int i = 0; i < N; i++)
    //     length *= this->shape[i];

    // for (int i = 0; i < length; i++)
    //     if ( (i+1) % this->shape[N-1] == 0)
    //         ss << this->values[i] << '\n';
    //     else
    //         ss << this->values[i] << ' ';

    ss << '>';
    return ss.str();
}

template<unsigned int N>
inline std::ostream &operator<<(std::ostream &os, HyperMatrix<N> const &M)
{
    return os << std::string(M);
}