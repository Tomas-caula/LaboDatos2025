package aed;

public class Fecha {
    public int dia = 0;
    public int mes = 0; 


    public Fecha(int dia, int mes) {
        this.dia = dia;
        this.mes = mes;
    }

    public Fecha(Fecha fecha) {
        this(fecha.dia, fecha.mes);
    }

    public Integer dia() {
        return dia;
    }

    public Integer mes() {
        // Implementar
        return mes;
    }

    public String toString() {
        // Implementar
        return dia + "/" + mes;
    }

    @Override
    public boolean equals(Object otra) {
        if (otra != null && this.getClass() == otra.getClass()){
            Fecha otraCopia = (Fecha) otra;
            if (otraCopia.dia == this.dia && otraCopia.mes == this.mes){
                return true;
            }
        }
        return false;
    }

    public void incrementarDia() {
        if (dia + 1 <= diasEnMes(mes)){
            dia += 1;
        }
        else {
            mes += 1;
            dia = 1;
        }
        if (mes == 13){
            mes = 1;
        }
    }

    private int diasEnMes(int mes) {
        int dias[] = {
                // ene, feb, mar, abr, may, jun
                31, 28, 31, 30, 31, 30,
                // jul, ago, sep, oct, nov, dic
                31, 31, 30, 31, 30, 31
        };
        return dias[mes - 1];
    }

}
