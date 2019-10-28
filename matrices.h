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
    std::array<int, N> shape;
    int dims = N;
    std::vector<double> values;

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

    /// <summary> Creates an N dimensional array of 0s with shape `shape` </summary>
    /// <param name=shape> Array containing the sizes of each dimenstion of the Martix</param>
    static HyperMatrix<N> Zeros(std::array<int, N> shape);
    int GetDims();

    operator std::string() const { return this->toString(); }
};

template<unsigned int N>
HyperMatrix<N>::HyperMatrix(std::array<int, N> shape)
{
    this->shape = shape;
}

template<unsigned int N>
HyperMatrix<N>::HyperMatrix(std::array<int, N> shape, std::vector<double> values)
{
    this->shape = shape;
    this->values = values;
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
int HyperMatrix<N>::GetDims()
{
    return this->dims;
}

// TODO: Make this look better for higher dimensions
template<unsigned int N>
std::string HyperMatrix<N>::toString() const
{
    std::stringstream ss;

    ss << '\n';

    int length = 1;
    for (int i = 0; i < N; i++)
        length *= this->shape[i];

    for (int i = 0; i < length; i++)
        if ( (i+1) % this->shape[N-1] == 0)
            ss << this->values[i] << '\n';
        else
            ss << this->values[i] << ' ';

    ss << '\n';

    return ss.str();
}

template<unsigned int N>
inline std::ostream &operator<<(std::ostream &os, HyperMatrix<N> const &M)
{
    return os << std::string(M);
}