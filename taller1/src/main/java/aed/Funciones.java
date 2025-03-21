package aed;

class Funciones {
    int cuadrado(int x) {
        return x * x ;
    }

    double distancia(double x, double y) {
        return Math.sqrt(x * x + y * y);
    }

    boolean esPar(int n) {
        // COMPLETAR
        int mod = n % 2;
        return mod == 0;
    }

    boolean esBisiesto(int n) {
        return n % 4 == 0 && n % 100 != 0 | n % 400 == 0;
    }

    int factorialIterativo(int n) {
        // COMPLETAR
        int count = 1;
        for (int i = 1; i <= n; i++){
            count = count*i;
        }

        return count;
    }

    int factorialRecursivo(int n) {
        if (n == 0 ){
            return 1;
        }
        return factorialRecursivo(n - 1) * n;
    }

    boolean esPrimo(int n) {
        if (n == 0 | n == 1){
            return false;
        }
        for (int i = 2; i < n; i++){
            if (n % i == 0){
                return false;
            }
        }
        return true;
    }

    int sumatoria(int[] numeros) {
        int count = 0;
        for (int elem : numeros){
            count += elem;
        }
        return count;
    }

    int busqueda(int[] numeros, int buscado) {
        for (int i = 0; i < numeros.length;i++){
            if (numeros[i] == buscado){
                return i;
            }
        }
        return 0;
    }

    boolean tienePrimo(int[] numeros) {
        for (int elem : numeros){
            if (esPrimo(elem)){
                return true;
            }
        }
        return false;
    }

    boolean todosPares(int[] numeros) {
        for (int elem : numeros){
            if (elem % 2 != 0){
                return false;
            }
        }
        return true;
    }

    boolean esPrefijo(String s1, String s2) {
        // COMPLETAR
        if (s1.length() > s2.length()){
            return false;
        }
       for (int i = 0; i < s1.length() && i < s2.length(); i ++){
            if (s1.charAt(i) != s2.charAt(i)){
                return false;
            }
       }
        return true;
    }

    boolean esSufijo(String s1, String s2) {
        // COMPLETAR
        if (s1.length() > s2.length()){
            return false;
        }
        for (int i = 0; i < s1.length() && i < s2.length(); i ++){
            if(s1.charAt(s1.length() - 1-i) != s2.charAt(s2.length() - 1-i)){
                return false;
            }
        }
        return true;
    }
}
