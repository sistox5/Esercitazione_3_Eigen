#include <iostream>
#include "Eigen/Eigen"
using namespace std;
using namespace Eigen;
Vector2d Palu(const Matrix2d& A,const Vector2d& b)
{
    Vector2d x;
    Vector2d y;
    Matrix2d L, P;
    PartialPivLU<Matrix2d> lu(A);
    L = Matrix2d::Identity();
    L(1,0) = lu.matrixLU()(1,0);
    P = lu.permutationP();
    y = L.triangularView<Lower>().solve(P * b);
    x = lu.matrixLU().triangularView<Upper>().solve(y);
    return x;
}
Vector2d QR(const Matrix2d& A,const Vector2d& b)
{ Vector2d x;
    Matrix2d Q, R;
    HouseholderQR<Matrix2d> qr(A);
    Q = qr.householderQ();
    R = qr.matrixQR().triangularView<Upper>();
    x = R.triangularView<Upper>().solve(Q.transpose() * b);
    return x;
}
double ErroreRelativo(const Vector2d& x,const Vector2d& xcorretto)
{
    double errore = (x - xcorretto).norm()
                   /(xcorretto).norm();
    return errore;
}
int main()
{
Matrix2d A2 ;    
Matrix2d A1 ;
Matrix2d A3 ;
A1 << 5.547001962252291e-01, -3.770900990025203e-02,
      8.320502943378437e-01, -9.992887623566787e-01;

A2 << 5.547001962252291e-01, -5.540607316466765e-01,
      8.320502943378437e-01, -8.324762492991313e-01;

A3 << 5.547001962252291e-01, -5.547001955851905e-01,
      8.320502943378437e-01, -8.320502947645361e-01;
Vector2d B1 ;
Vector2d B2 ;
Vector2d B3 ;

B1 << -5.169911863249772e-01, 
       1.672384680188350e-01;
B2 << -6.394645785530173e-04,
       4.259549612877223e-04;
B3 << -6.400391328043042e-10,
       4.266924591433963e-10;     
Vector2d x1;
Vector2d x2;
Vector2d x3;
Vector2d x;
x << -1.000000000000000e+00,
     -1.000000000000000e+00;

x1 = Palu(A1, B1);
x2 = Palu(A2, B2);   
x3 = Palu(A3, B3);
    cout << "errori relativi rispetto a A1,A2,A3 con PaLu" << endl << endl;
    cout << "Errore di A1 : " << ErroreRelativo(x1,x) << endl << endl;
    cout << "Errore di A2 : " << ErroreRelativo(x2,x) << endl << endl;
    cout << "Errore di A3 : " << ErroreRelativo(x3,x) << endl << endl;
cout << "errori relativi rispetto a A1,A2,A3 con QR" << endl << endl;
x1 = QR(A1, B1);
x2 = QR(A2, B2);
x3 = QR(A3, B3);
cout << "Errore di A1 : " << ErroreRelativo(x1,x) << endl << endl;
cout << "Errore di A2 : " << ErroreRelativo(x2,x) << endl << endl;
cout << "Errore di A3 : " << ErroreRelativo(x3,x) << endl << endl;
    return 0;
}
