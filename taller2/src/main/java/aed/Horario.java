package aed;

public class Horario {

    public int hora = 0;
    public int minutos = 0;

    public Horario(int hora, int minutos) {
        this.hora = hora;
        this.minutos = minutos;
    }

    public int hora() {
        // Implementar
        return this.hora;
    }

    public int minutos() {
        // Implementar
        return this.minutos;
    }

    @Override
    public String toString() {
        // Implementar
        return hora +  ":" + minutos;
    }

    @Override
    public boolean equals(Object otro) {
        if(otro != null && this.getClass() == otro.getClass()){
            Horario otroCopia = (Horario) otro;
            if(otroCopia.hora == this.hora && otroCopia.minutos == this.minutos){
                return true;
            }
        }
        return false;
    }

}
