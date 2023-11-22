#include <iostream>
#include <vector>

using namespace std;

class Complex {

    private:
        float real;
        float imag;


    public:

        Complex(){
            this->real = 0;
            this->imag = 0;
        }

        Complex(float real, float imag) {
            this->real = real;
            this->imag = imag;
        }


        void setReal(float real) {
            this->real = real;
        }

        void setImag(float imag) {
            this->imag = imag;
        }

        float getReal() const {
            return this->real;
        }

        float getImag() const {
            return this->imag;
        }


        Complex operator+(Complex const &obj) {
            Complex res;
            res.real = this->real + obj.real;
            res.imag = this->imag + obj.imag;
            return res;
        }

        Complex operator-(Complex const &obj) {
            Complex res;
            res.real = this->real - obj.real;
            res.imag = this->imag - obj.imag;
            return res;
        }

        Complex operator*(Complex const &obj) {
            Complex res;
            res.real = this->real * obj.real - this->imag * obj.imag;
            res.imag = this->real * obj.imag + this->imag * obj.real;
            return res;
        }

        friend bool operator!=(Complex const &obj1, Complex const &obj2) {
            return (obj1.real != obj2.real || obj1.imag != obj2.imag);
        }

        friend bool operator==(Complex const &obj1, Complex const &obj2) {
            return (obj1.real == obj2.real && obj1.imag == obj2.imag);
        }

        friend ostream& operator<<(ostream& os, const Complex& obj) {
            os << obj.real << " + " << obj.imag << "i";
            return os;
        }

        Complex complex_mult(Complex const &obj) const{
            return Complex(this->real * obj.real - this->imag * obj.imag, 
                            this->real * obj.imag + this->imag * obj.real);
        }


};